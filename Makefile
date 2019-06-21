PYTHON = python

help:
	@echo "Please use \`make <target>\` where <target> is one of"
	@echo "  fetch         to fetch the data"
	@echo "  sensor        run sensor space processing"
	@echo "  source        run source space processing"
	@echo "  profile       to profile memory consumption"
	@echo "  all           fetch data and run full pipeline

fetch_test: # make only one subject for testing
	$(PYTHON) -c "import mne; mne.datasets.sample.data_path(update_path=True)"

fetch: fetch_test

sensor:
	$(PYTHON) 01-import_and_filter.py
	$(PYTHON) 02-apply_maxwell_filter.py
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

test: fetch_test sensor source

all: fetch sensor source report
