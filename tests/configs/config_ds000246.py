"""
Auditory MEG
"""

study_name = 'ds000246'
bids_root = '~/mne_data/ds000246'
deriv_root = '~/mne_data/derivatives/mne-bids-pipeline/ds000246'

runs = ['01']
l_freq = 0.3
h_freq = 100
decim = 4
subjects = ['0001']
ch_types = ['meg']
reject = dict(mag=4e-12, eog=250e-6)
conditions = ['standard', 'deviant', 'button']
contrasts = [('deviant', 'standard')]
decode = True
on_error = 'abort'
parallel_backend = 'dask'
dask_worker_memory_limit = '3.5G'
dask_temp_dir = "./.dask-worker-space"
dask_open_dashboard = True
N_JOBS = 2
