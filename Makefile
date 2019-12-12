PYTHON = python

help:
	@echo "Please use \`make <target>\` where <target> is one of"
	@echo "  fetch         fetch all testing datasets"
	@echo "  sensor        run sensor space processing"
	@echo "  source        run source space processing"
	@echo "  all           fetch data and run full pipeline"

fetch:
	$(PYTHON) ./tests/download_test_data.py --dataset=${DATASET}

sensor:
	$(PYTHON) 01-import_and_maxfilter.py
	$(PYTHON) 02-frequency_filter.py
	$(PYTHON) 03-extract_events.py
	$(PYTHON) 04-make_epochs.py
	$(PYTHON) 05a-run_ica.py
	$(PYTHON) 05b-run_ssp.py
	$(PYTHON) 06a-apply_ica.py
	$(PYTHON) 06b-apply_ssp.py
	$(PYTHON) 07-make_evoked.py
	$(PYTHON) 08-group_average_sensors.py
	$(PYTHON) 09-sliding_estimator.py
	$(PYTHON) 10-time_frequency.py

source:
	$(PYTHON) 11-make_forward.py
	$(PYTHON) 12-make_cov.py
	$(PYTHON) 13-make_inverse.py
	$(PYTHON) 14-group_average_source.py

report:
	$(PYTHON) 99-make_reports.py

test: fetch sensor source

all: fetch sensor source report
