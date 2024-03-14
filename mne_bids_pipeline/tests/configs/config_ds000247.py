"""OMEGA Resting State Sample Data."""

import numpy as np

bids_root = "~/mne_data/ds000247"
deriv_root = "~/mne_data/derivatives/mne-bids-pipeline/ds000247"

subjects = ["0002"]
sessions = ["01"]
task = "rest"
task_is_rest = True

crop_runs = (0, 100)  # to speed up computations

ch_types = ["meg"]
spatial_filter = "ssp"

l_freq = 1.0
h_freq = 40.0

rest_epochs_duration = 10
rest_epochs_overlap = 0

epochs_tmin = 0
baseline = None

time_frequency_conditions = ["rest"]
time_frequency_freq_min = 1.0
time_frequency_freq_max = 30.0
time_frequency_cycles = np.arange(time_frequency_freq_min, time_frequency_freq_max) / 4
time_frequency_subtract_evoked = True
