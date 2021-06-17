"""
MNE Sample Data
"""

study_name = 'ds000248'
bids_root = '~/mne_data/ds000248'
deriv_root = '~/mne_data/ds000248/derivatives/mne-bids-pipeline-no-mri'

subjects = ['01']
rename_events = {'Smiley': 'Emoji',
                 'Button': 'Switch'}
conditions = ['Auditory', 'Visual', 'Auditory/Left', 'Auditory/Right']
contrasts = [('Auditory/Right', 'Auditory/Left')]

ch_types = ['meg']
use_maxwell_filter = False
process_er = False

use_mri_template = True
