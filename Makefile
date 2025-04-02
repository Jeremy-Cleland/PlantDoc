EXPERIMENT ?= cbam
CLI=PYTHONPATH=. python cli/main.py
CONFIG=configs/config.yaml
RAW_DIR ?= data/raw
OUTPUT_DIR ?= outputs/data_quality

# PlantDoc Project Makefile

# Constants
LOG_DIR := outputs/cmd_logs
TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)

# Extract experiment name from config for logging
EXPERIMENT_NAME := $(shell grep "experiment_name" $(CONFIG) | awk -F'"' '{print $$2}')
ifeq ($(EXPERIMENT_NAME),)
EXPERIMENT_NAME := default
endif

# Ensure log directory exists
$(shell mkdir -p $(LOG_DIR))

# Default target
.PHONY: help
help:
	@echo "PlantDoc Project Makefile"
	@echo "-------------------------"
	@echo "Available targets:"
	@echo "  train        Run training with experiment=$(EXPERIMENT_NAME)"
	@echo "  eval         Run evaluation"
	@echo "  prepare      Run data preparation"
	@echo "  clean        Clean outputs"
	@echo "  help         Show this help message"
	@echo ""
	@echo "All commands log to $(LOG_DIR)/<command>_<experiment>.log"
	@echo "Logs are also copied to the experiment output directory"

# Logging function with color preservation and experiment name
define log_cmd
	@mkdir -p $(LOG_DIR)
	@LOGFILE=$(LOG_DIR)/$(2)_$(EXPERIMENT_NAME).log; \
	echo "=== Command: $(1) ===" > $$LOGFILE; \
	echo "=== Started: $(shell date -u '+%Y-%m-%dT%H:%M:%SZ') ===" >> $$LOGFILE; \
	echo "=== Experiment: $(EXPERIMENT_NAME) ===" >> $$LOGFILE; \
	echo "" >> $$LOGFILE; \
	echo "Running: $(1) (logging to $$LOGFILE)"; \
	(FORCE_COLOR=1 PYTHONUNBUFFERED=1 $(1) | tee -a $$LOGFILE) || (echo "Command failed with exit code $$?" >> $$LOGFILE); \
	echo "" >> $$LOGFILE; \
	echo "=== Ended: $(shell date -u '+%Y-%m-%dT%H:%M:%SZ') ===" >> $$LOGFILE; \
	rm -f $(LOG_DIR)/latest.log; \
	ln -s $$(basename $$LOGFILE) $(LOG_DIR)/latest.log; \
	echo "Checking for experiment output directory..."; \
	sleep 3; \
	if [ -d "outputs/$(EXPERIMENT_NAME)" ] || ls -d outputs/$(EXPERIMENT_NAME)_v* >/dev/null 2>&1; then \
		NEWEST_DIR=$$(ls -td outputs/$(EXPERIMENT_NAME)* | head -n1); \
		if [ -n "$$NEWEST_DIR" ]; then \
			mkdir -p $$NEWEST_DIR/logs; \
			cp $$LOGFILE $$NEWEST_DIR/logs/; \
			echo "Log copied to $$NEWEST_DIR/logs/$$(basename $$LOGFILE)"; \
		fi; \
	fi;
endef

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +
	find . -name '.pytest_cache' -exec rm -rf {} +
	find . -name '*.egg-info' -exec rm -rf {} +
	find . -name '*.egg' -exec rm -f {} +

lint:
	ruff check .

lint-fix:
	ruff check --fix .

format:
	black .
	isort .

typecheck:
	mypy .

check-all: lint format typecheck

	
train:
	$(call log_cmd,$(CLI) train,python_train)

eval:
	$(call log_cmd,$(CLI) eval,python_eval)

tune:
	$(call log_cmd,$(CLI) tune,python_tune)

prepare-data:
	$(call log_cmd,$(CLI) prepare --raw-dir $(RAW_DIR) $(if $(DRY_RUN),--dry-run,),python_prepare)

prepare-data-dry:
	$(MAKE) prepare-data DRY_RUN=1

report:
	$(call log_cmd,PYTHONPATH=. python reports/generate_report.py --experiment $(EXPERIMENT),python_report)

plots:
	$(call log_cmd,PYTHONPATH=. python reports/generate_plots.py --experiment $(EXPERIMENT),python_plots)

# Clean outputs but preserve the log directory
clean:
	@echo "Cleaning outputs and temporary files..."
	@rm -rf outputs/* logs/* .cache/*
	@mkdir -p $(LOG_DIR)
	@find . -name '*.pyc' -exec rm -f {} +
	@find . -name '*.pyo' -exec rm -f {} +
	@find . -name '*~' -exec rm -f {} +
	@find . -name '__pycache__' -exec rm -rf {} +
	@find . -name '.pytest_cache' -exec rm -rf {} +
	@find . -name '*.egg-info' -exec rm -rf {} +
	@find . -name '*.egg' -exec rm -f {} +
	@echo "Cleaned successfully. Log directory recreated at $(LOG_DIR)"
