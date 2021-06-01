"""
Faces dataset
"""

study_name = 'ds000117'
bids_root = '~/mne_data/ds000117'
task = 'facerecognition'
ch_types = ['meg']
runs = ['01', '02']
sessions = ['meg']
interactive = False
acq = None
subjects = ['01']

resample_sfreq = 125.
crop = (0, 350)

find_flat_channels_meg = True
find_noisy_channels_meg = True
use_maxwell_filter = True

mf_cal_fname = bids_root + '/derivatives/meg_derivatives/sss_cal.dat'
mf_ctc_fname = bids_root + '/derivatives/meg_derivatives/ct_sparse.fif'

reject = {'grad': 4000e-13, 'mag': 4e-12}
conditions = ['Famous', 'Unfamiliar', 'Scrambled']
contrasts = [('Famous', 'Scrambled'),
             ('Unfamiliar', 'Scrambled'),
             ('Famous', 'Unfamiliar')]
decode = True
