PYTHON = python

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  fetch         to fetch the data"
	@echo "  sensor        run sensor space processing"
	@echo "  source        run source space processing"
	@echo "  profile       to profile memory consumption"
	@echo "  all           fetch data and run full pipeline

fetch_test: # make only one subject for testing
	mkdir -p data/system_calibration_files/ && wget https://osf.io/prnzb/download -cO data/system_calibration_files/ct_sparse_nspn.fif
	mkdir -p data/system_calibration_files/ && wget https://osf.io/hyg8k/download -cO data/system_calibration_files/sss_cal_nspn.dat
	mkdir -p data/MEG/SB01/ && wget https://osf.io/k9bth/download -cO data/MEG/SB01/SB01_Localizer_raw.fif

fetch: fetch_test
	mkdir -p data/MEG/SB02/ && wget https://osf.io/4rbpd/download -cO data/MEG/SB02/SB02_Localizer_raw.fif
	mkdir -p data/MEG/SB04/ && wget https://osf.io/3nxyv/download -cO data/MEG/SB04/SB04_Localizer_raw.fif
	mkdir -p data/MEG/SB05/ && wget https://osf.io/57qwd/download -cO data/MEG/SB05/SB05_Localizer_raw.fif
	mkdir -p data/MEG/SB06/ && wget https://osf.io/z7ybc/download -cO data/MEG/SB06/SB06_Localizer_raw.fif
	mkdir -p data/MEG/SB07/ && wget https://osf.io/m6wpf/download -cO data/MEG/SB07/SB07_Localizer_raw.fif
	mkdir -p data/MEG/SB08/ && wget https://osf.io/xrhqu/download -cO data/MEG/SB08/SB08_Localizer_raw.fif
	mkdir -p data/MEG/SB09/ && wget https://osf.io/vkftn/download -cO data/MEG/SB09/SB09_Localizer_raw.fif
	mkdir -p data/MEG/SB10/ && wget https://osf.io/29xtm/download -cO data/MEG/SB10/SB10_Localizer_raw.fif
	mkdir -p data/MEG/SB11/ && wget https://osf.io/2zst8/download -cO data/MEG/SB11/SB11_Localizer_raw.fif
	mkdir -p data/MEG/SB12/ && wget https://osf.io/u7cj2/download -cO data/MEG/SB12/SB12_Localizer_raw.fif

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

test: fetch_test sensor source report

all: fetch sensor source report
