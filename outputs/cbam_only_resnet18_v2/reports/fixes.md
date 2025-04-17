# Main Page fixes (index.html)

The main content is not centered in the page, please fix this.

The dataset overview is not connected to anything please use th info belwo to create it

## Dataset Overview

- **Total number of classes:** 39
- **Total number of images:** 61,486

### File Format Distribution

- **.jpg:** 61,486 (100.00%)

## Class Distribution

| Class | Number of Images | Percentage |
|-------|----------------:|-----------:|
| Orange_Haunglongbing_Citrus_greening | 5,507 | 8.96% |
| Tomato_Tomato_Yellow_Leaf_Curl_Virus | 5,357 | 8.71% |
| Soybean_healthy | 5,090 | 8.28% |
| Peach_Bacterial_spot | 2,297 | 3.74% |
| Tomato_Bacterial_spot | 2,127 | 3.46% |
| Tomato_Late_blight | 1,909 | 3.10% |
| Squash_Powdery_mildew | 1,835 | 2.98% |
| Tomato_Septoria_leaf_spot | 1,771 | 2.88% |
| Tomato_Spider_mites_Two-spotted_spider_mite | 1,676 | 2.73% |
| Apple_healthy | 1,645 | 2.68% |
| Tomato_healthy | 1,591 | 2.59% |
| Blueberry_healthy | 1,502 | 2.44% |
| Pepper_bell_healthy | 1,478 | 2.40% |
| Tomato_Target_Spot | 1,404 | 2.28% |
| Grape_Esca_Black_Measles | 1,383 | 2.25% |
| Corn_Common_rust | 1,192 | 1.94% |
| Grape_Black_rot | 1,180 | 1.92% |
| Corn_healthy | 1,162 | 1.89% |
| Background_without_leaves | 1,143 | 1.86% |
| Strawberry_Leaf_scorch | 1,109 | 1.80% |
| Grape_Leaf_blight_Isariopsis_Leaf_Spot | 1,076 | 1.75% |
| Cherry_Powdery_mildew | 1,052 | 1.71% |
| Apple_Apple_scab | 1,000 | 1.63% |
| Apple_Black_rot | 1,000 | 1.63% |
| Apple_Cedar_apple_rust | 1,000 | 1.63% |
| Cherry_healthy | 1,000 | 1.63% |
| Corn_Cercospora_leaf_spot_Gray_leaf_spot | 1,000 | 1.63% |
| Corn_Northern_Leaf_Blight | 1,000 | 1.63% |
| Grape_healthy | 1,000 | 1.63% |
| Peach_healthy | 1,000 | 1.63% |
| Pepper_bell_Bacterial_spot | 1,000 | 1.63% |
| Potato_Early_blight | 1,000 | 1.63% |
| Potato_Late_blight | 1,000 | 1.63% |
| Potato_healthy | 1,000 | 1.63% |
| Raspberry_healthy | 1,000 | 1.63% |
| Strawberry_healthy | 1,000 | 1.63% |
| Tomato_Early_blight | 1,000 | 1.63% |
| Tomato_Leaf_Mold | 1,000 | 1.63% |
| Tomato_Tomato_mosaic_virus | 1,000 | 1.63% |

## Image Properties

### Image Dimensions

- **Width (min, median, max):** 204, 256, 350 pixels
- **Height (min, median, max):** 192, 256, 350 pixels

### Aspect Ratio

- **Aspect ratio (min, median, max):** 1.00, 1.00, 1.33
- **Approximately square images:** 97.84%

### File Sizes

- **File size (min, median, max):** 4.11, 14.96, 28.60 KB

### Color Information

- **Average RGB values:** R=118.45, G=124.62, B=104.63
- **Average brightness:** 115.90
- **RGB standard deviation:** R=45.51, G=38.65, B=49.59

## Recommendations for Model Training

- **Class Imbalance:** High class imbalance detected (ratio of largest to smallest class: 5.51). Consider using techniques like oversampling, undersampling, or weighted loss functions.
- **Image Sizes:** Relatively consistent image dimensions. Standard preprocessing should work well.
- **Suggested Input Size:** 256x256 pixels based on median dimensions.

### Data Augmentation Recommendations

- **Color Augmentation:** Low color variation detected. Recommend using color jittering, hue/saturation adjustments, and brightness/contrast modifications.

- **Underrepresented Classes:** The following classes have few samples and may benefit from extra augmentation or oversampling:
  - Apple_Apple_scab
  - Apple_Black_rot
  - Apple_Cedar_apple_rust
  - Background_without_leaves
  - Cherry_Powdery_mildew
  - Cherry_healthy
  - Corn_Cercospora_leaf_spot_Gray_leaf_spot
  - Corn_Common_rust
  - Corn_Northern_Leaf_Blight
  - Corn_healthy
  - Grape_Black_rot
  - Grape_Leaf_blight_Isariopsis_Leaf_Spot
  - Grape_healthy
  - Peach_healthy
  - Pepper_bell_Bacterial_spot
  - Potato_Early_blight
  - Potato_Late_blight
  - Potato_healthy
  - Raspberry_healthy
  - Strawberry_Leaf_scorch
  - Strawberry_healthy
  - Tomato_Early_blight
  - Tomato_Leaf_Mold
  - Tomato_Tomato_mosaic_virus

- **Recommended Augmentation Techniques for Plant Disease Dataset:**
  - Random rotations (0-360 degrees)
  - Random flips (horizontal and vertical)
  - Random cropping (ensuring the disease area is preserved)
  - Minimal zoom and scale variations
  - Brightness and contrast adjustments
  - Light color jittering

### Model Architecture Recommendations

- **Baseline Models:** EfficientNet-B0/B1, ResNet50, or MobileNetV3 are good starting points
- **Training Strategy:** Transfer learning from ImageNet pre-trained weights
- **Loss Function:** Categorical Cross-Entropy (or weighted version if addressing class imbalance)
- **Optimization:** SGD with momentum or Adam optimizer
- **Regularization:** Dropout (0.2-0.5) and batch normalization

### Validation Strategy

- **Data Split:** 80% training, 10% validation, 10% test
- **Stratification:** Ensure balanced class representation in all splits
- **Metrics:** Accuracy, F1-score (weighted and per-class), confusion matrix

### Advanced Techniques to Consider

- **Grad-CAM Visualization:** For model interpretability and disease localization
- **Test-Time Augmentation:** Apply multiple augmentations during inference and average results
- **Ensemble Methods:** Combine predictions from multiple model architectures
- **Semi-Supervised Learning:** If additional unlabeled data is available


/Users/jeremy/plantdoc/outputs/dataset_analysis_report/analysis_dashboard.png

/Users/jeremy/plantdoc/outputs/dataset_analysis_report/class_distribution_pie.png

/Users/jeremy/plantdoc/outputs/dataset_analysis_report/correlation_matrix.png

/Users/jeremy/plantdoc/outputs/dataset_analysis_report/file_size_by_class.png



# Training Report Fixes

For both training reprots (v1 and v2) the Data Analysis in the sidebr would move to the section Data Analysis:

Make a section based off this info:

# Plant Disease Dataset Analysis Report

## Dataset Overview

- **Total number of classes:** 39
- **Total number of images:** 61,486

### File Format Distribution

- **.jpg:** 61,486 (100.00%)

## Class Distribution

| Class | Number of Images | Percentage |
|-------|----------------:|-----------:|
| Orange_Haunglongbing_Citrus_greening | 5,507 | 8.96% |
| Tomato_Tomato_Yellow_Leaf_Curl_Virus | 5,357 | 8.71% |
| Soybean_healthy | 5,090 | 8.28% |
| Peach_Bacterial_spot | 2,297 | 3.74% |
| Tomato_Bacterial_spot | 2,127 | 3.46% |
| Tomato_Late_blight | 1,909 | 3.10% |
| Squash_Powdery_mildew | 1,835 | 2.98% |
| Tomato_Septoria_leaf_spot | 1,771 | 2.88% |
| Tomato_Spider_mites_Two-spotted_spider_mite | 1,676 | 2.73% |
| Apple_healthy | 1,645 | 2.68% |
| Tomato_healthy | 1,591 | 2.59% |
| Blueberry_healthy | 1,502 | 2.44% |
| Pepper_bell_healthy | 1,478 | 2.40% |
| Tomato_Target_Spot | 1,404 | 2.28% |
| Grape_Esca_Black_Measles | 1,383 | 2.25% |
| Corn_Common_rust | 1,192 | 1.94% |
| Grape_Black_rot | 1,180 | 1.92% |
| Corn_healthy | 1,162 | 1.89% |
| Background_without_leaves | 1,143 | 1.86% |
| Strawberry_Leaf_scorch | 1,109 | 1.80% |
| Grape_Leaf_blight_Isariopsis_Leaf_Spot | 1,076 | 1.75% |
| Cherry_Powdery_mildew | 1,052 | 1.71% |
| Apple_Apple_scab | 1,000 | 1.63% |
| Apple_Black_rot | 1,000 | 1.63% |
| Apple_Cedar_apple_rust | 1,000 | 1.63% |
| Cherry_healthy | 1,000 | 1.63% |
| Corn_Cercospora_leaf_spot_Gray_leaf_spot | 1,000 | 1.63% |
| Corn_Northern_Leaf_Blight | 1,000 | 1.63% |
| Grape_healthy | 1,000 | 1.63% |
| Peach_healthy | 1,000 | 1.63% |
| Pepper_bell_Bacterial_spot | 1,000 | 1.63% |
| Potato_Early_blight | 1,000 | 1.63% |
| Potato_Late_blight | 1,000 | 1.63% |
| Potato_healthy | 1,000 | 1.63% |
| Raspberry_healthy | 1,000 | 1.63% |
| Strawberry_healthy | 1,000 | 1.63% |
| Tomato_Early_blight | 1,000 | 1.63% |
| Tomato_Leaf_Mold | 1,000 | 1.63% |
| Tomato_Tomato_mosaic_virus | 1,000 | 1.63% |

## Image Properties

### Image Dimensions

- **Width (min, median, max):** 204, 256, 350 pixels
- **Height (min, median, max):** 192, 256, 350 pixels

### Aspect Ratio

- **Aspect ratio (min, median, max):** 1.00, 1.00, 1.33
- **Approximately square images:** 97.84%

### File Sizes

- **File size (min, median, max):** 4.11, 14.96, 28.60 KB

### Color Information

- **Average RGB values:** R=118.45, G=124.62, B=104.63
- **Average brightness:** 115.90
- **RGB standard deviation:** R=45.51, G=38.65, B=49.59

## Recommendations for Model Training

- **Class Imbalance:** High class imbalance detected (ratio of largest to smallest class: 5.51). Consider using techniques like oversampling, undersampling, or weighted loss functions.
- **Image Sizes:** Relatively consistent image dimensions. Standard preprocessing should work well.
- **Suggested Input Size:** 256x256 pixels based on median dimensions.

### Data Augmentation Recommendations

- **Color Augmentation:** Low color variation detected. Recommend using color jittering, hue/saturation adjustments, and brightness/contrast modifications.

- **Underrepresented Classes:** The following classes have few samples and may benefit from extra augmentation or oversampling:
  - Apple_Apple_scab
  - Apple_Black_rot
  - Apple_Cedar_apple_rust
  - Background_without_leaves
  - Cherry_Powdery_mildew
  - Cherry_healthy
  - Corn_Cercospora_leaf_spot_Gray_leaf_spot
  - Corn_Common_rust
  - Corn_Northern_Leaf_Blight
  - Corn_healthy
  - Grape_Black_rot
  - Grape_Leaf_blight_Isariopsis_Leaf_Spot
  - Grape_healthy
  - Peach_healthy
  - Pepper_bell_Bacterial_spot
  - Potato_Early_blight
  - Potato_Late_blight
  - Potato_healthy
  - Raspberry_healthy
  - Strawberry_Leaf_scorch
  - Strawberry_healthy
  - Tomato_Early_blight
  - Tomato_Leaf_Mold
  - Tomato_Tomato_mosaic_virus

- **Recommended Augmentation Techniques for Plant Disease Dataset:**
  - Random rotations (0-360 degrees)
  - Random flips (horizontal and vertical)
  - Random cropping (ensuring the disease area is preserved)
  - Minimal zoom and scale variations
  - Brightness and contrast adjustments
  - Light color jittering

### Model Architecture Recommendations

- **Baseline Models:** EfficientNet-B0/B1, ResNet50, or MobileNetV3 are good starting points
- **Training Strategy:** Transfer learning from ImageNet pre-trained weights
- **Loss Function:** Categorical Cross-Entropy (or weighted version if addressing class imbalance)
- **Optimization:** SGD with momentum or Adam optimizer
- **Regularization:** Dropout (0.2-0.5) and batch normalization

### Validation Strategy

- **Data Split:** 80% training, 10% validation, 10% test
- **Stratification:** Ensure balanced class representation in all splits
- **Metrics:** Accuracy, F1-score (weighted and per-class), confusion matrix

### Advanced Techniques to Consider

- **Grad-CAM Visualization:** For model interpretability and disease localization
- **Test-Time Augmentation:** Apply multiple augmentations during inference and average results
- **Ensemble Methods:** Combine predictions from multiple model architectures
- **Semi-Supervised Learning:** If additional unlabeled data is available


/Users/jeremy/plantdoc/outputs/dataset_analysis_report/analysis_dashboard.png

/Users/jeremy/plantdoc/outputs/dataset_analysis_report/class_distribution_pie.png

/Users/jeremy/plantdoc/outputs/dataset_analysis_report/correlation_matrix.png

/Users/jeremy/plantdoc/outputs/dataset_analysis_report/file_size_by_class.png


**On the side the first confidence analysis is not connected to anything please remove it.**


**Nex the Mdoel analysis is does not connect to anything, please review the report and create a analyis on the model**


**Next: Key insights is not completed, please review the report and create a key insights section.**