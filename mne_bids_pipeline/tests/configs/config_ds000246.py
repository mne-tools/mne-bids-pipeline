"""
Brainstorm - Auditory Dataset.

See https://openneuro.org/datasets/ds000246/versions/1.0.0 for more
information.
"""

study_name = 'ds000246'
bids_root = '~/mne_data/ds000246'
deriv_root = '~/mne_data/derivatives/mne-bids-pipeline/ds000246'

runs = ['01']
crop_runs = (0, 120)  # Reduce memory usage on CI system
l_freq = 0.3
h_freq = 100
epochs_decim = 4
subjects = ['0001']
ch_types = ['meg']
reject = dict(mag=4e-12, eog=250e-6)
conditions = ['standard', 'deviant', 'button']
contrasts = [('deviant', 'standard')]
decode = True
decoding_time_generalization = True
decoding_time_generalization_decim = 4
on_error = 'abort'
plot_psd_for_runs = []  # too much memory on CIs

parallel_backend = 'dask'
dask_worker_memory_limit = '2G'
dask_temp_dir = "./.dask-worker-space"
dask_open_dashboard = True
N_JOBS = 2
