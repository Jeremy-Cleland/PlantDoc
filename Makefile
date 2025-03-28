EXPERIMENT ?= cbam_default
CLI=python cli/main.py
CONFIG=configs/config.yaml

train:
	$(CLI) train

eval:
	$(CLI) eval

tune:
	$(CLI) tune

report:
	python reports/generate_report.py --experiment $(EXPERIMENT)

plots:
	python reports/generate_plots.py --experiment $(EXPERIMENT)

clean:
	rm -rf outputs/* logs/* .cache/*
