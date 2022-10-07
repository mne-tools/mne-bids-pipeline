"""
SRM Resting-state EEG
"""

study_name = 'ds003775'
bids_root = '~/mne_data/ds003775'
deriv_root = '~/mne_data/derivatives/mne-bids-pipeline/ds003775'

# To get all subjects for example:
# from mne_bids import get_entity_vals
# subjects = sorted(get_entity_vals(bids_root, entity_key='subject'))
subjects = ['010']

reader_extra_params = {"units": "uV"}

sessions = ["t1"]

run_source_estimation = False

ch_types = ['eeg']

baseline = None
reject = None
spatial_filter = None

h_freq = 40
l_freq = None

task = "resteyesc"
task_is_rest = True
epochs_tmin = 0.
epochs_tmax = 10.
rest_epochs_overlap = 0.
rest_epochs_duration = 10.
baseline = None

parallel_backend = 'loky'
dask_open_dashboard = True

log_level = 'info'

N_JOBS = 1
