import pathlib
from mne_bids import get_entity_vals

study_name = 'ds003775'
bids_root = pathlib.Path('/storage/store2/data/ds003775')
deriv_root = pathlib.Path('/storage/store2/derivatives/ds003775/mne-bids-pipeline')

subjects = sorted(get_entity_vals(bids_root, entity_key='subject'))
subjects = subjects[:1]  # take only the first subject

sessions = ["t1"]

run_source_estimation = False

ch_types = ['eeg']

baseline = None
reject = None
spatial_filter = None

h_freq = 110
l_freq = None

task = "resteyesc"
epochs_tmin = 0.
epochs_tmax = 10.
rest_epochs_overlap = 0.
rest_epochs_duration = 10.
baseline = None

parallel_backend = 'loky'
dask_open_dashboard = True

on_error = 'continue'
# on_error = 'abort'
# on_error = 'debug'

# log_level = 'debug'
log_level = 'info'

N_JOBS = 1
