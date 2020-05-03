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
	$(PYTHON) 03-make_epochs.py
	$(PYTHON) 04a-run_ica.py
	$(PYTHON) 04b-run_ssp.py
	$(PYTHON) 05a-apply_ica.py
	$(PYTHON) 05b-apply_ssp.py
	$(PYTHON) 06-make_evoked.py
	$(PYTHON) 07-group_average_sensors.py
	$(PYTHON) 08-sliding_estimator.py
	$(PYTHON) 09-time_frequency.py

source:
	$(PYTHON) 10-make_forward.py
	$(PYTHON) 11-make_cov.py
	$(PYTHON) 12-make_inverse.py
	$(PYTHON) 13-group_average_source.py

report:
	$(PYTHON) 99-make_reports.py

test: fetch sensor source

all: fetch sensor source report
