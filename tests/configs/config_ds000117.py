"""
Faces dataset
"""

study_name = 'ds000117'
bids_root = '~/mne_data/ds000117'
task = 'facerecognition'
ch_types = ['meg']
runs = ['01']
sessions = ['meg']
interactive = False
acq = None
subjects = ['01']

# use_maxwell_filter = True
# subjects_dir = op.join(bids_root, 'derivatives', 'freesurfer', 'subjects')

reject = {'grad': 4000e-13, 'mag': 4e-12}
conditions = ['Famous', 'Unfamiliar', 'Scrambled']
contrasts = [('Famous', 'Scrambled'),
             ('Unfamiliar', 'Scrambled'),
             ('Famous', 'Unfamiliar')]
decode = True
use_ssp = False
use_ica = False
