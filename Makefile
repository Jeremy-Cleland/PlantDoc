EXPERIMENT ?= cbam_default
CLI=PYTHONPATH=. python cli/main.py
CONFIG=configs/config.yaml
RAW_DIR ?= data/raw
OUTPUT_DIR ?= outputs/data_quality


clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +
	find . -name '.pytest_cache' -exec rm -rf {} +
	find . -name '*.egg-info' -exec rm -rf {} +
	find . -name '*.egg' -exec rm -f {} +

	
train:
	$(CLI) train

eval:
	$(CLI) eval

tune:
	$(CLI) tune

prepare-data:
	$(CLI) prepare --raw-dir $(RAW_DIR) $(if $(DRY_RUN),--dry-run,)

prepare-data-dry:
	$(MAKE) prepare-data DRY_RUN=1

report:
	PYTHONPATH=. python reports/generate_report.py --experiment $(EXPERIMENT)

plots:
	PYTHONPATH=. python reports/generate_plots.py --experiment $(EXPERIMENT)

clean:
	rm -rf outputs/* logs/* .cache/*
