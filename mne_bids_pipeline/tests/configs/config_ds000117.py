"""
Faces dataset
"""

study_name = 'ds000117'
bids_root = '~/mne_data/ds000117'
deriv_root = '~/mne_data/derivatives/mne-bids-pipeline/ds000117'

task = 'facerecognition'
ch_types = ['meg']
runs = ['01', '02']
sessions = ['meg']
interactive = False
acq = None
subjects = ['01']

raw_resample_sfreq = 125.
crop_runs = (0, 300)  # Reduce memory usage on CI system

find_flat_channels_meg = False
find_noisy_channels_meg = False
use_maxwell_filter = True

mf_reference_run = '01'
mf_cal_fname = bids_root + '/derivatives/meg_derivatives/sss_cal.dat'
mf_ctc_fname = bids_root + '/derivatives/meg_derivatives/ct_sparse.fif'

reject = {'grad': 4000e-13, 'mag': 4e-12}
conditions = ['Famous', 'Unfamiliar', 'Scrambled']
contrasts = [('Famous', 'Scrambled'),
             ('Unfamiliar', 'Scrambled'),
             ('Famous', 'Unfamiliar')]

decode = True
decoding_time_generalization = True

run_source_estimation = False
