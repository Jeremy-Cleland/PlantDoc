# Path: plantdoc/core/data/prepare_data.py


# Path: plantdoc/core/data/prepare_data.py
# Description: Consolidates dataset validation, analysis, and visualization tasks.

import json
import os
import random
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig
from PIL import Image, UnidentifiedImageError
from plantdoc.core.data.transforms import get_transforms

# Assuming these utilities and transforms are available in your project structure
from plantdoc.utils.logging import get_logger
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Import basic plotting functions if they are separate, otherwise define inline or import
# Example: from plantdoc.utils.visualization import basic_plots, advanced_plots
# For simplicity here, plotting functions are included below where needed.

logger = get_logger(__name__)

# --- Helper Functions (from validation script) ---

def fix_filename(filename: str) -> str:
    """Fix problematic filenames by removing spaces and special characters."""
    clean_name = filename.replace(" ", "_")
    special_chars = r"()[]{}!@#$%^&*+="
    for char in special_chars:
        clean_name = clean_name.replace(char, "")
    while "__" in clean_name:
        clean_name = clean_name.replace("__", "_")
    return clean_name

def check_folder_name(folder_name: str) -> Tuple[bool, str, List[str]]:
    """Check folder name best practices."""
    issues = []
    suggested_name = folder_name
    has_issues = False
    if "," in folder_name:
        has_issues = True
        issues.append("Contains comma")
        suggested_name = suggested_name.replace(",", "")
    if " " in folder_name:
        has_issues = True
        issues.append("Contains spaces")
        suggested_name = suggested_name.replace(" ", "_")
    special_chars = set("!@#$%^&*()+={}[]|\\:;\"'<>?/")
    found_special = [c for c in folder_name if c in special_chars]
    if found_special:
        has_issues = True
        issues.append(f"Contains special characters: {''.join(found_special)}")
        for char in found_special:
            suggested_name = suggested_name.replace(char, "")
    if "__" in folder_name:
        has_issues = True
        issues.append("Contains consecutive underscores")
        while "__" in suggested_name:
            suggested_name = suggested_name.replace("__", "_")
    return has_issues, suggested_name, issues

def validate_image(file_path: Path) -> bool:
    """Verify that an image file can be opened and is valid."""
    try:
        with Image.open(file_path) as img:
            img.verify() # Verify image header and structure
        # Optionally, try to fully load to catch more subtle issues
        # with Image.open(file_path) as img:
        #     img.load()
        return True
    except (UnidentifiedImageError, OSError, IOError, SyntaxError) as e:
        logger.debug(f"Invalid image {file_path}: {e}")
        return False
    except Exception as e:
        logger.warning(f"Unexpected error validating image {file_path}: {e}")
        return False # Treat unexpected errors as invalid

def rename_file_with_unique_name(file_path: Path, new_file_path: Path) -> Path:
    """Rename a file, adding timestamp if the target exists."""
    if new_file_path.exists() and file_path.resolve() != new_file_path.resolve():
        timestamp = int(time.time() * 1000)
        new_file_path = new_file_path.with_name(f"{new_file_path.stem}_{timestamp}{new_file_path.suffix}")
    return new_file_path

# --- Validation Core Function ---

def run_validation(cfg: DictConfig) -> Dict[str, Any]:
    """
    Scan the dataset directory, report/fix issues based on config.

    Args:
        cfg: Hydra configuration object.

    Returns:
        Dictionary containing the scan results.
    """
    prep_cfg = cfg.prepare_data
    data_dir = Path(cfg.paths.raw_dir) # Validate the raw data
    output_dir = Path(prep_cfg.output_dir) / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    dry_run = prep_cfg.dry_run
    fix_extensions = prep_cfg.fix_extensions
    verify_images = prep_cfg.verify_images
    fix_folders = prep_cfg.fix_folders

    logger.info(f"Starting dataset validation in: {data_dir}")
    if dry_run:
        logger.info("DRY RUN MODE: No changes will be made.")

    stats = {
        "total_files": 0, "renamed_files": 0, "problematic_files": 0,
        "class_stats": {}, "extension_stats": Counter(),
        "folder_issues": 0, "folder_suggestions": {},
    }
    valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')

    if not data_dir.is_dir():
        logger.error(f"Data directory not found: {data_dir}")
        return {"error": f"Data directory not found: {data_dir}"}

    class_dirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    class_dirs.sort(key=lambda x: x.name)
    logger.info(f"Found {len(class_dirs)} potential class directories.")

    # --- Folder Name Check ---
    logger.info("Checking folder names...")
    folder_renames = {}
    current_class_dirs = list(class_dirs) # Copy to modify list if renaming
    for i, folder_path in enumerate(class_dirs):
        folder_name = folder_path.name
        has_issues, suggested_name, issues = check_folder_name(folder_name)
        if has_issues:
            stats["folder_issues"] += 1
            stats["folder_suggestions"][folder_name] = {
                "suggested_name": suggested_name, "issues": issues
            }
            if fix_folders and not dry_run:
                new_path = data_dir / suggested_name
                try:
                    if new_path.exists():
                        timestamp = int(time.time() * 1000)
                        suggested_name = f"{suggested_name}_{timestamp}"
                        new_path = data_dir / suggested_name
                    logger.info(f"Renaming folder: {folder_path} -> {new_path}")
                    folder_path.rename(new_path)
                    folder_renames[folder_name] = suggested_name
                    current_class_dirs[i] = new_path # Update path in the list for file iteration
                except Exception as e:
                    logger.error(f"Error renaming folder {folder_name} to {suggested_name}: {e}")
            else:
                 logger.warning(f"Folder '{folder_name}' has issues: {issues}. Suggested: '{suggested_name}'")

    # --- File Processing ---
    logger.info("Processing files within class directories...")
    for class_dir_path in tqdm(current_class_dirs, desc="Processing classes"):
        class_name = class_dir_path.name
        stats["class_stats"][class_name] = {"total": 0, "renamed": 0, "problematic": 0}
        class_file_count = 0
        class_renamed_count = 0
        class_problem_count = 0

        try:
            for file_path in class_dir_path.iterdir():
                if file_path.is_file():
                    stats["total_files"] += 1
                    class_file_count += 1
                    filename = file_path.name
                    ext = file_path.suffix.lower()
                    stats["extension_stats"][ext] += 1

                    is_valid_ext = ext in valid_extensions
                    if not is_valid_ext:
                        logger.debug(f"Skipping non-image file: {file_path}")
                        continue # Skip non-image files entirely

                    # Image Verification (Do this BEFORE potential rename)
                    is_problematic = False
                    if verify_images:
                        if not validate_image(file_path):
                            logger.warning(f"Problematic image detected: {file_path}")
                            stats["problematic_files"] += 1
                            class_problem_count += 1
                            is_problematic = True
                            # Decide whether to skip fixing problematic files
                            # continue # Uncomment to skip fixing problematic files

                    # File Renaming Logic
                    has_spaces_or_special = any(c in filename for c in " ()[]{}!@#$%^&*+=")
                    needs_ext_fix = ext != ".jpg"
                    needs_rename = has_spaces_or_special or needs_ext_fix

                    if needs_rename and fix_extensions:
                        if is_problematic and not dry_run:
                            logger.warning(f"Skipping rename for problematic file: {file_path}")
                            continue

                        base_name = file_path.stem
                        if has_spaces_or_special:
                            base_name = fix_filename(base_name)

                        new_filename = f"{base_name}.jpg"
                        new_file_path = class_dir_path / new_filename
                        new_file_path = rename_file_with_unique_name(file_path, new_file_path)

                        if file_path != new_file_path:
                            if not dry_run:
                                try:
                                    # Use copy and remove for potentially safer operation across filesystems
                                    shutil.copy2(file_path, new_file_path)
                                    file_path.unlink()
                                    logger.debug(f"Renamed: {filename} -> {new_file_path.name}")
                                    stats["renamed_files"] += 1
                                    class_renamed_count += 1
                                except Exception as e:
                                    logger.error(f"Error renaming {file_path} to {new_file_path}: {e}")
                            else:
                                logger.info(f"[Dry Run] Would rename: {filename} -> {new_file_path.name}")
                                stats["renamed_files"] += 1
                                class_renamed_count += 1

        except Exception as e:
            logger.error(f"Error processing directory {class_dir_path}: {e}")
            continue # Skip to the next class directory

        stats["class_stats"][class_name]["total"] = class_file_count
        stats["class_stats"][class_name]["renamed"] = class_renamed_count
        stats["class_stats"][class_name]["problematic"] = class_problem_count

    # --- Reporting ---
    report_path = output_dir / "validation_report.json"
    logger.info(f"Saving validation report to {report_path}")
    try:
        with open(report_path, "w") as f:
            # Convert Counter to dict for JSON serialization
            stats["extension_stats"] = dict(stats["extension_stats"])
            json.dump(stats, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save validation report: {e}")

    logger.info("--- Validation Summary ---")
    logger.info(f"Total Classes Found: {len(current_class_dirs)}")
    logger.info(f"Total Files Scanned: {stats['total_files']}")
    logger.info(f"Files Flagged for Rename: {stats['renamed_files']}")
    logger.info(f"Problematic/Invalid Images: {stats['problematic_files']}")
    logger.info(f"Folders with Naming Issues: {stats['folder_issues']}")
    logger.info("Extension Counts:")
    for ext, count in sorted(stats["extension_stats"].items()):
         logger.info(f"  {ext}: {count}")
    if dry_run and (stats['renamed_files'] > 0 or (stats['folder_issues'] > 0 and fix_folders)):
        logger.info("Run with prepare_data.dry_run=False to apply fixes.")
    logger.info("--- Validation Complete ---")

    return stats


# --- Helper Functions (from analysis script) ---

def extract_image_metadata(image_path: Path) -> Optional[Dict[str, Any]]:
    """Extract metadata from an image file."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            aspect_ratio = width / height if height > 0 else 0
            img_format = img.format
            img_mode = img.mode
            try:
                file_size_kb = image_path.stat().st_size / 1024
            except Exception:
                 file_size_kb = -1 # Indicate error

            # Basic color info (requires loading image data)
            try:
                img_array = np.array(img.convert("RGB"))
                mean_rgb = np.mean(img_array, axis=(0, 1))
                std_rgb = np.std(img_array, axis=(0, 1))
                brightness = mean_rgb.mean() # Simple brightness estimate
            except Exception as e:
                 logger.debug(f"Could not get color info for {image_path}: {e}")
                 mean_rgb = [-1,-1,-1]
                 std_rgb = [-1,-1,-1]
                 brightness = -1

            return {
                "path": str(image_path), "class": image_path.parent.name,
                "width": width, "height": height, "aspect_ratio": aspect_ratio,
                "format": img_format, "mode": img_mode, "file_size_kb": file_size_kb,
                "mean_r": mean_rgb[0], "mean_g": mean_rgb[1], "mean_b": mean_rgb[2],
                "std_r": std_rgb[0], "std_g": std_rgb[1], "std_b": std_rgb[2],
                "brightness": brightness,
            }
    except Exception as e:
        logger.error(f"Error processing metadata for {image_path}: {e}")
        return None

def create_summary_report(
    df: pd.DataFrame, class_counts: Dict[str, int], file_extensions: Dict[str, int], output_dir: Path
) -> None:
    """Generate a summary report (Markdown) of dataset statistics."""
    report_path = output_dir / "analysis_summary_report.md"
    logger.info(f"Generating analysis summary report: {report_path}")
    total_images = sum(class_counts.values())

    with open(report_path, "w") as f:
        f.write("# Dataset Analysis Report\n\n")
        f.write("## Overview\n\n")
        f.write(f"- **Total Classes:** {len(class_counts)}\n")
        f.write(f"- **Total Images:** {total_images:,}\n")
        if not df.empty:
            f.write(f"- **Images Analyzed (Sample):** {len(df):,}\n")

        f.write("\n### File Formats (Overall)\n\n")
        total_ext = sum(file_extensions.values())
        for ext, count in sorted(file_extensions.items()):
            perc = count / total_ext * 100 if total_ext > 0 else 0
            f.write(f"- **{ext}:** {count:,} ({perc:.1f}%)\n")

        f.write("\n## Class Distribution\n\n")
        f.write("| Class | Image Count | Percentage |\n")
        f.write("|---|---|---|\n")
        for class_name, count in sorted(class_counts.items(), key=lambda item: item[1], reverse=True):
            perc = count / total_images * 100 if total_images > 0 else 0
            f.write(f"| {class_name} | {count:,} | {perc:.2f}% |\n")

        if not df.empty:
            f.write("\n## Image Properties (Sampled)\n\n")
            f.write("### Dimensions\n")
            f.write(f"- **Width:** Min={df['width'].min():,}, Median={df['width'].median():,.0f}, Max={df['width'].max():,}\n")
            f.write(f"- **Height:** Min={df['height'].min():,}, Median={df['height'].median():,.0f}, Max={df['height'].max():,}\n")
            f.write(f"- **Aspect Ratio:** Min={df['aspect_ratio'].min():.2f}, Median={df['aspect_ratio'].median():.2f}, Max={df['aspect_ratio'].max():.2f}\n")

            f.write("\n### File Size\n")
            f.write(f"- **Size (KB):** Min={df['file_size_kb'].min():.2f}, Median={df['file_size_kb'].median():.2f}, Max={df['file_size_kb'].max():.2f}\n")

            f.write("\n### Color & Brightness\n")
            f.write(f"- **Avg RGB:** ({df['mean_r'].mean():.1f}, {df['mean_g'].mean():.1f}, {df['mean_b'].mean():.1f})\n")
            f.write(f"- **Avg Brightness:** {df['brightness'].mean():.1f}\n")

    logger.info("Analysis summary report generated.")

# --- Analysis Core Function ---

def run_analysis(cfg: DictConfig) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Analyze the dataset, generate statistics, and optional plots.

    Args:
        cfg: Hydra configuration object.

    Returns:
        Tuple containing:
            - DataFrame with image metadata (or None if error).
            - Dictionary with dataset statistics.
    """
    prep_cfg = cfg.prepare_data
    data_dir = Path(cfg.paths.raw_dir) # Analyze the raw (potentially fixed) data
    output_dir = Path(prep_cfg.output_dir) / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_size = prep_cfg.sample_size
    random_seed = prep_cfg.random_seed
    create_plots = prep_cfg.create_plots
    plots_output_dir = output_dir / "plots"
    if create_plots:
        plots_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting dataset analysis for: {data_dir}")
    random.seed(random_seed)
    np.random.seed(random_seed)

    # --- Collect Basic Stats & File Paths ---
    class_counts = {}
    file_extensions = Counter()
    all_image_paths = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')

    try:
        class_dirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        class_dirs.sort(key=lambda x: x.name)
        logger.info(f"Found {len(class_dirs)} classes for analysis.")

        for class_dir in tqdm(class_dirs, desc="Scanning for analysis"):
            count = 0
            for item in class_dir.iterdir():
                if item.is_file() and item.suffix.lower() in valid_extensions:
                    all_image_paths.append(item)
                    file_extensions[item.suffix.lower()] += 1
                    count += 1
            class_counts[class_dir.name] = count

    except Exception as e:
         logger.error(f"Error scanning directories for analysis: {e}")
         return None, {"error": "Failed to scan directories."}

    total_images = len(all_image_paths)
    logger.info(f"Found {total_images} total image files for potential analysis.")
    if total_images == 0:
        logger.warning("No images found to analyze.")
        return pd.DataFrame(), {"num_classes": len(class_counts), "total_images": 0}

    # --- Sample Images for Detailed Metadata Extraction ---
    if total_images > sample_size:
        logger.info(f"Sampling {sample_size} images for detailed analysis (seed={random_seed}).")
        sampled_paths = random.sample(all_image_paths, sample_size)
    else:
        logger.info(f"Analyzing all {total_images} images.")
        sampled_paths = all_image_paths

    # --- Extract Metadata ---
    metadata_list = []
    logger.info(f"Extracting metadata from {len(sampled_paths)} images...")
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_path = {executor.submit(extract_image_metadata, path): path for path in sampled_paths}
        for future in tqdm(as_completed(future_to_path), total=len(sampled_paths), desc="Extracting metadata"):
            result = future.result()
            if result:
                metadata_list.append(result)

    if not metadata_list:
         logger.warning("No metadata could be extracted from sampled images.")
         df = pd.DataFrame()
    else:
        df = pd.DataFrame(metadata_list)
        metadata_csv_path = output_dir / "image_metadata_sample.csv"
        try:
            df.to_csv(metadata_csv_path, index=False)
            logger.info(f"Saved sampled image metadata to {metadata_csv_path}")
        except Exception as e:
            logger.error(f"Failed to save metadata CSV: {e}")


    # --- Calculate Overall Statistics ---
    dataset_stats = {
        "num_classes": len(class_counts),
        "total_images": total_images,
        "class_counts": class_counts,
        "file_extensions": dict(file_extensions),
        "sampled_images_analyzed": len(df),
        # Add stats derived from the sampled DataFrame
        "image_dimensions_stats (sample)": {
            "min_width": int(df["width"].min()) if not df.empty else None,
            "max_width": int(df["width"].max()) if not df.empty else None,
            "median_width": float(df["width"].median()) if not df.empty else None,
            "min_height": int(df["height"].min()) if not df.empty else None,
            "max_height": int(df["height"].max()) if not df.empty else None,
            "median_height": float(df["height"].median()) if not df.empty else None,
        },
         "aspect_ratio_stats (sample)": {
            "min": float(df["aspect_ratio"].min()) if not df.empty else None,
            "max": float(df["aspect_ratio"].max()) if not df.empty else None,
            "median": float(df["aspect_ratio"].median()) if not df.empty else None,
        },
        "file_size_stats (sample)": {
            "min_kb": float(df["file_size_kb"].min()) if not df.empty else None,
            "max_kb": float(df["file_size_kb"].max()) if not df.empty else None,
            "median_kb": float(df["file_size_kb"].median()) if not df.empty else None,
        },
        "color_info_stats (sample)": {
            "avg_mean_rgb": [df['mean_r'].mean(), df['mean_g'].mean(), df['mean_b'].mean()] if not df.empty else None,
            "avg_std_rgb": [df['std_r'].mean(), df['std_g'].mean(), df['std_b'].mean()] if not df.empty else None,
            "avg_brightness": float(df["brightness"].mean()) if not df.empty else None,
        },
    }

    # Save statistics
    stats_path = output_dir / "dataset_analysis_stats.json"
    logger.info(f"Saving analysis statistics to {stats_path}")
    try:
        with open(stats_path, "w") as f:
            # Convert numpy types for JSON
            sanitized_stats = json.loads(json.dumps(dataset_stats, default=lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else str(x)))
            json.dump(sanitized_stats, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save analysis stats JSON: {e}")

    # Generate Markdown Report
    create_summary_report(df, class_counts, file_extensions, output_dir)

    # --- Generate Plots (if enabled) ---
    if create_plots and not df.empty:
        logger.info(f"Generating analysis plots in {plots_output_dir}")
        try:
            # Plot Class Distribution
            plt.figure(figsize=(max(12, len(class_counts) * 0.5), 8))
            sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()), palette="viridis")
            plt.xticks(rotation=90, fontsize=10)
            plt.title("Class Distribution")
            plt.ylabel("Number of Images")
            plt.tight_layout()
            plt.savefig(plots_output_dir / "class_distribution.png", dpi=150)
            plt.close()

            # Plot Image Dimensions (Width vs Height)
            plt.figure(figsize=(10, 8))
            sns.scatterplot(data=df, x="width", y="height", hue="class", alpha=0.6, legend=False, s=20)
            plt.title("Image Dimensions (Width vs Height) - Sampled")
            plt.xlabel("Width (pixels)")
            plt.ylabel("Height (pixels)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_output_dir / "dimensions_scatter.png", dpi=150)
            plt.close()

            # Plot Aspect Ratio Distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(df["aspect_ratio"], kde=True, bins=50)
            plt.title("Aspect Ratio Distribution - Sampled")
            plt.xlabel("Aspect Ratio (Width / Height)")
            plt.tight_layout()
            plt.savefig(plots_output_dir / "aspect_ratio_hist.png", dpi=150)
            plt.close()

            # Plot File Size Distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(df["file_size_kb"], kde=True, bins=50)
            plt.title("File Size Distribution (KB) - Sampled")
            plt.xlabel("File Size (KB)")
            plt.tight_layout()
            plt.savefig(plots_output_dir / "file_size_hist.png", dpi=150)
            plt.close()

            logger.info("Analysis plots generated.")

        except Exception as e:
            logger.error(f"Error generating analysis plots: {e}", exc_info=True)

    logger.info("--- Dataset Analysis Complete ---")
    return df, dataset_stats


# --- Helper Functions (from visualization script) ---

def extract_features_for_viz(image_path: Path, target_size=(224, 224)) -> Optional[Tuple[List[float], str]]:
    """Extract simple features from an image for visualization (t-SNE, clustering)."""
    try:
        img = cv2.imread(str(image_path))
        if img is None: return None, None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)

        features = []
        # Color stats
        for i in range(3):
            features.extend([np.mean(img[:, :, i]), np.std(img[:, :, i])])
        # Gradient stats
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        features.extend([np.mean(magnitude), np.std(magnitude)])
        # Simple histogram
        hist = cv2.calcHist([gray], [0], None, [8], [0, 256])
        features.extend( (hist.flatten() / (hist.sum() + 1e-6)).tolist() ) # Normalize safely

        return features, image_path.parent.name
    except Exception as e:
        logger.error(f"Error extracting features for viz from {image_path}: {e}")
        return None, None

def sample_images_for_viz(data_dir: Path, n_per_class: int = 30, max_classes: Optional[int] = None, random_seed: int = 42) -> Tuple[List[Path], List[str]]:
    """Sample images for visualization tasks like t-SNE."""
    random.seed(random_seed)
    all_image_paths = []
    labels_list = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')

    class_dirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    class_dirs.sort(key=lambda x: x.name)

    if max_classes and len(class_dirs) > max_classes:
        logger.info(f"Sampling {max_classes} classes out of {len(class_dirs)} for visualization.")
        class_dirs = random.sample(class_dirs, max_classes)

    logger.info(f"Sampling up to {n_per_class} images per class for visualization...")
    for class_dir in class_dirs:
        image_files = [item for item in class_dir.iterdir() if item.is_file() and item.suffix.lower() in valid_extensions]
        sampled_images = random.sample(image_files, min(n_per_class, len(image_files)))
        all_image_paths.extend(sampled_images)
        labels_list.extend([class_dir.name] * len(sampled_images))

    logger.info(f"Sampled {len(all_image_paths)} images from {len(class_dirs)} classes for visualization.")
    return all_image_paths, labels_list


# --- Visualization Core Function ---

def run_visualization(cfg: DictConfig):
    """
    Create advanced visualizations for the dataset.

    Args:
        cfg: Hydra configuration object.
    """
    prep_cfg = cfg.prepare_data
    data_dir = Path(cfg.paths.raw_dir)
    output_dir = Path(prep_cfg.output_dir) / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sample size for t-SNE/clustering often different from metadata analysis
    # Use a specific config or reuse sample_size
    n_per_class_viz = prep_cfg.get("n_per_class_viz", 30)
    max_classes_viz = prep_cfg.get("max_classes_viz", None) # Limit classes for viz if needed
    random_seed = prep_cfg.random_seed

    logger.info(f"Starting dataset visualization generation in: {output_dir}")
    logger.info(f"Sampling: {n_per_class_viz} images/class, Max classes: {max_classes_viz}, Seed: {random_seed}")

    # --- Sample Images and Extract Features ---
    sampled_paths, sampled_labels = sample_images_for_viz(
        data_dir, n_per_class=n_per_class_viz, max_classes=max_classes_viz, random_seed=random_seed
    )

    if not sampled_paths:
        logger.warning("No images sampled for visualization. Skipping feature extraction and related plots.")
        features = None
        labels = None
    else:
        logger.info("Extracting features for visualization...")
        features_list = []
        labels_list = []
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_path = {executor.submit(extract_features_for_viz, path): path for path in sampled_paths}
            for future in tqdm(as_completed(future_to_path), total=len(sampled_paths), desc="Extracting viz features"):
                feature_set, class_name = future.result()
                if feature_set is not None:
                    features_list.append(feature_set)
                    labels_list.append(class_name)

        if not features_list:
             logger.warning("Feature extraction for visualization failed. Skipping related plots.")
             features = None
             labels = None
        else:
            features = np.array(features_list)
            labels = np.array(labels_list)
            logger.info(f"Extracted features matrix shape: {features.shape}")

    # --- Generate Visualizations ---
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("paper", font_scale=1.5)

    # 1. Image Grid
    try:
        logger.info("Generating image grid...")
        n_grid_classes = min(len(np.unique(sampled_labels)) if labels is not None else 10, 10)
        create_image_grid_viz(data_dir, output_dir, n_per_class=5, max_classes=n_grid_classes)
    except Exception as e:
        logger.error(f"Failed to create image grid: {e}", exc_info=True)

    # 2. t-SNE Plot (if features extracted)
    if features is not None and labels is not None:
        try:
             logger.info("Generating t-SNE plot...")
             create_tsne_viz(features, labels, output_dir, random_seed)
        except Exception as e:
             logger.error(f"Failed to create t-SNE plot: {e}", exc_info=True)

        # 3. Hierarchical Clustering (if features extracted)
        try:
            logger.info("Generating hierarchical clustering plot...")
            create_hierarchical_clustering_viz(features, labels, output_dir)
        except Exception as e:
            logger.error(f"Failed to create hierarchical clustering plot: {e}", exc_info=True)

        # 4. Similarity Matrix (if features extracted)
        try:
            logger.info("Generating class similarity matrix...")
            create_similarity_matrix_viz(features, labels, output_dir)
        except Exception as e:
            logger.error(f"Failed to create similarity matrix: {e}", exc_info=True)

    # 5. Augmentation Visualization
    try:
        logger.info("Generating augmentation visualization...")
        # Need the augmentation config part from main cfg
        create_augmentation_viz(cfg, data_dir, output_dir, num_samples=3, random_seed=random_seed)
    except Exception as e:
        logger.error(f"Failed to create augmentation visualization: {e}", exc_info=True)

    logger.info(f"--- Visualization Complete --- Plots saved in {output_dir}")


# --- Visualization Plotting Functions (Internal Helpers for run_visualization) ---

def create_image_grid_viz(data_dir: Path, output_dir: Path, n_per_class: int = 5, max_classes: int = 10):
    """Internal helper to create image grid plot."""
    class_dirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    class_dirs.sort(key=lambda x: x.name)
    if len(class_dirs) > max_classes: class_dirs = random.sample(class_dirs, max_classes)
    if not class_dirs: return

    fig = plt.figure(figsize=(n_per_class * 2.5, len(class_dirs) * 2.5))
    valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')

    for i, class_dir in enumerate(class_dirs):
        image_files = [item for item in class_dir.iterdir() if item.is_file() and item.suffix.lower() in valid_extensions]
        selected_images = random.sample(image_files, min(n_per_class, len(image_files)))

        for j, img_path in enumerate(selected_images):
            ax = fig.add_subplot(len(class_dirs), n_per_class, i * n_per_class + j + 1)
            try:
                img = Image.open(img_path)
                ax.imshow(np.array(img))
            except Exception as e:
                 logger.warning(f"Could not load image {img_path} for grid: {e}")
                 ax.text(0.5, 0.5, 'Error', horizontalalignment='center', verticalalignment='center')
            ax.axis("off")
            if j == 0:
                 simple_class = class_dir.name.replace("___", " - ").replace("_", " ")[:25]
                 ax.set_ylabel(simple_class, fontsize=10, rotation=0, labelpad=40, va='center', ha='right')

    plt.suptitle("Sample Images by Class", fontsize=16, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(output_dir / "viz_image_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

def create_tsne_viz(features: np.ndarray, labels: np.ndarray, output_dir: Path, random_seed: int):
    """Internal helper to create t-SNE plot."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Optional PCA pre-reduction
    n_pca = min(50, X_scaled.shape[0] - 1, X_scaled.shape[1])
    if n_pca > 2:
        logger.debug(f"Applying PCA with {n_pca} components before t-SNE.")
        pca = PCA(n_components=n_pca, random_state=random_seed)
        X_pca = pca.fit_transform(X_scaled)
    else:
        X_pca = X_scaled # Skip PCA if too few samples/features

    logger.info("Running t-SNE...")
    perplexity_val = min(30, X_pca.shape[0] - 1)
    if perplexity_val <= 1:
        logger.warning(f"Perplexity ({perplexity_val}) too low for t-SNE, skipping plot.")
        return

    tsne = TSNE(n_components=2, random_state=random_seed, perplexity=perplexity_val, n_iter=300)
    X_tsne = tsne.fit_transform(X_pca)

    df_tsne = pd.DataFrame({"x": X_tsne[:, 0], "y": X_tsne[:, 1], "class": labels})
    unique_classes = sorted(df_tsne["class"].unique())

    plt.figure(figsize=(14, 10))
    colors = sns.color_palette("viridis", len(unique_classes))
    class_to_color = dict(zip(unique_classes, colors))

    for class_name, group in df_tsne.groupby("class"):
        simple_name = class_name.replace("___", "-").replace("_", " ")[:30]
        plt.scatter(group["x"], group["y"], label=simple_name, color=class_to_color[class_name], alpha=0.7, s=50)

    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.title("t-SNE Visualization of Image Features (Sampled)", fontsize=16)
    plt.xlabel("t-SNE Feature 1")
    plt.ylabel("t-SNE Feature 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust for legend
    plt.savefig(output_dir / "viz_tsne.png", dpi=150, bbox_inches="tight")
    plt.close()

def create_hierarchical_clustering_viz(features: np.ndarray, labels: np.ndarray, output_dir: Path):
    """Internal helper to create hierarchical clustering plot."""
    class_features = {}
    for i, label in enumerate(labels):
        class_features.setdefault(label, []).append(features[i])
    class_mean_features = {cls: np.mean(feats, axis=0) for cls, feats in class_features.items() if feats}
    if not class_mean_features: return

    class_names = list(class_mean_features.keys())
    feature_matrix = np.array([class_mean_features[cls] for cls in class_names])

    try:
        linked = linkage(feature_matrix, "ward")
    except Exception as e:
        logger.error(f"Hierarchical clustering linkage failed: {e}. Skipping plot.")
        return

    plt.figure(figsize=(12, max(8, len(class_names) * 0.3)))
    simple_class_names = [name.replace("___", "-").replace("_", " ")[:40] for name in class_names]
    dendrogram(linked, orientation="right", labels=simple_class_names, leaf_font_size=10)
    plt.title("Hierarchical Clustering of Class Mean Features", fontsize=16)
    plt.xlabel("Distance")
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "viz_hierarchical_clustering.png", dpi=150, bbox_inches="tight")
    plt.close()

def create_similarity_matrix_viz(features: np.ndarray, labels: np.ndarray, output_dir: Path):
    """Internal helper to create similarity matrix plot."""
    class_features = {}
    for i, label in enumerate(labels):
        class_features.setdefault(label, []).append(features[i])
    class_mean_features = {cls: np.mean(feats, axis=0) for cls, feats in class_features.items() if feats}
    if not class_mean_features: return

    class_names = list(class_mean_features.keys())
    feature_matrix = np.array([class_mean_features[cls] for cls in class_names])

    similarity_matrix = cosine_similarity(feature_matrix)
    simple_class_names = [name.replace("___", "-").replace("_", " ")[:30] for name in class_names]
    df_similarity = pd.DataFrame(similarity_matrix, index=simple_class_names, columns=simple_class_names)

    plt.figure(figsize=(max(12, len(class_names)*0.5), max(10, len(class_names)*0.5)))
    sns.heatmap(df_similarity, cmap="viridis", annot=False, fmt=".2f", linewidths=.5, square=True)
    plt.title("Class Similarity Matrix (Cosine Similarity of Mean Features)", fontsize=14)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "viz_similarity_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

def create_augmentation_viz(cfg: DictConfig, data_dir: Path, output_dir: Path, num_samples: int = 3, random_seed: int = 42):
    """Internal helper to visualize augmentations."""
    random.seed(random_seed)
    # Use get_transforms to get the configured training augmentations
    try:
        # Pass only relevant parts of config if needed, or full cfg if get_transforms handles it
        train_transforms = get_transforms(cfg, split='train')
        # Extract individual transforms for display (this might be tricky depending on Compose structure)
        # Or define a fixed set of augmentations to showcase here:
        showcase_augmentations = {
            "Original": A.Compose([A.Resize(256, 256)]), # Just resize
            "HorizontalFlip": A.Compose([A.Resize(256, 256), A.HorizontalFlip(p=1.0)]),
            "Rotate": A.Compose([A.Resize(256, 256), A.Rotate(limit=45, p=1.0)]),
            "BrightnessContrast": A.Compose([A.Resize(256, 256), A.RandomBrightnessContrast(p=1.0)]),
            "ShiftScaleRotate": A.Compose([A.Resize(256, 256), A.ShiftScaleRotate(p=1.0)]),
            "CoarseDropout": A.Compose([A.Resize(256, 256), A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=1.0)])
        }
        num_augments = len(showcase_augmentations)
    except Exception as e:
        logger.error(f"Could not get/define transforms for augmentation viz: {e}. Skipping.")
        return

    class_dirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not class_dirs: return
    selected_classes = random.sample(class_dirs, min(num_samples, len(class_dirs)))
    valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')

    fig, axes = plt.subplots(len(selected_classes), num_augments, figsize=(num_augments * 2.5, len(selected_classes) * 2.5))
    if len(selected_classes) == 1: axes = np.expand_dims(axes, axis=0) # Handle single sample case

    for i, class_dir in enumerate(selected_classes):
        image_files = [item for item in class_dir.iterdir() if item.is_file() and item.suffix.lower() in valid_extensions]
        if not image_files: continue
        img_path = random.choice(image_files)

        try:
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.warning(f"Could not load image {img_path} for augmentation viz: {e}")
            continue

        for j, (aug_name, aug_func) in enumerate(showcase_augmentations.items()):
            ax = axes[i, j]
            try:
                augmented = aug_func(image=image)["image"]
                ax.imshow(augmented)
            except Exception as e:
                logger.warning(f"Error applying {aug_name} to {img_path}: {e}")
                ax.imshow(image) # Show original on error
                ax.set_title(f"{aug_name}\n(Error)", fontsize=8, color='red')
            ax.axis("off")
            if i == 0: ax.set_title(aug_name, fontsize=10)
            if j == 0: ax.set_ylabel(class_dir.name[:20], fontsize=10, rotation=0, labelpad=40, va='center', ha='right')

    plt.suptitle("Sample Data Augmentations", fontsize=16, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    plt.savefig(output_dir / "viz_augmentations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --- Main Orchestration Function ---

def run_prepare_data(cfg: DictConfig):
    """
    Runs the full data preparation pipeline: validation, analysis, visualization.

    Args:
        cfg: Hydra configuration object.
    """
    start_time = time.time()
    logger.info("===== Starting Data Preparation Pipeline =====")

    prep_cfg = cfg.prepare_data
    main_output_dir = Path(prep_cfg.output_dir)
    main_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Main output directory: {main_output_dir}")

    # --- Step 1: Validation ---
    logger.info("--- Running Step 1: Dataset Validation ---")
    validation_results = run_validation(cfg)
    # Check for critical errors from validation if needed

    # --- Step 2: Analysis (Conditional) ---
    analysis_results_df = None
    analysis_stats = {}
    if prep_cfg.run_analysis_after_validation:
        logger.info("--- Running Step 2: Dataset Analysis ---")
        analysis_results_df, analysis_stats = run_analysis(cfg)
    else:
        logger.info("--- Skipping Step 2: Dataset Analysis (run_analysis_after_validation=False) ---")

    # --- Step 3: Visualization (Conditional) ---
    if prep_cfg.run_visualization_after_analysis:
        if analysis_stats.get("error"):
             logger.warning("Skipping visualization because analysis failed.")
        else:
            logger.info("--- Running Step 3: Dataset Visualization ---")
            run_visualization(cfg) # Pass df/stats if needed by viz funcs
    else:
        logger.info("--- Skipping Step 3: Dataset Visualization (run_visualization_after_analysis=False) ---")

    # --- Step 4: Combined Report (Optional) ---
    if prep_cfg.generate_combined_report:
        logger.info("--- Running Step 4: Generating Combined Report (Placeholder) ---")
        # TODO: Implement combined report generation
        # - Could involve merging the validation JSON, analysis JSON/Markdown,
        #   and adding links/references to the visualization PNGs.
        # - Example: Create a main README.md in main_output_dir summarizing all findings.
        combined_report_path = main_output_dir / "DATA_PREPARATION_REPORT.md"
        try:
            with open(combined_report_path, "w") as f:
                f.write("# Data Preparation Pipeline Report\n\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Data Source: {cfg.paths.raw_dir}\n\n")
                f.write("## Summary\n\n")
                if validation_results.get("error"): f.write("- Validation: FAILED\n")
                else: f.write("- Validation: COMPLETED\n")
                if prep_cfg.run_analysis_after_validation:
                    if analysis_stats.get("error"): f.write("- Analysis: FAILED\n")
                    else: f.write("- Analysis: COMPLETED\n")
                if prep_cfg.run_visualization_after_analysis and not analysis_stats.get("error"):
                     f.write("- Visualization: COMPLETED\n")

                f.write("\n## Details\n\n")
                f.write("- **Validation Report:** See `validation/validation_report.json`\n")
                if prep_cfg.run_analysis_after_validation and not analysis_stats.get("error"):
                     f.write("- **Analysis Stats:** See `analysis/dataset_analysis_stats.json`\n")
                     f.write("- **Analysis Summary:** See `analysis/analysis_summary_report.md`\n")
                     if prep_cfg.create_plots: f.write("- **Analysis Plots:** See `analysis/plots/`\n")
                if prep_cfg.run_visualization_after_analysis and not analysis_stats.get("error"):
                     f.write("- **Visualizations:** See `visualizations/`\n")
            logger.info(f"Generated combined report stub: {combined_report_path}")
        except Exception as e:
            logger.error(f"Failed to generate combined report: {e}")


    elapsed_time = time.time() - start_time
    logger.info(f"===== Data Preparation Pipeline Finished in {elapsed_time:.2f} seconds =====")

