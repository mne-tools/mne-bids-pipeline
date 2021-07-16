"""
MNE Sample Data: BEM fromm T1 images
"""

study_name = 'ds000248'
bids_root = '~/mne_data/ds000248'
deriv_root = '~/mne_data/mne-bids-pipeline/tests/ds000248_T1_BEM'
subjects_dir = f'{bids_root}/derivatives/freesurfer/subjects'

subjects = ['01']
conditions = ['Auditory']

ch_types = ['meg']

bem_mri_images = 'T1'
recreate_bem = True
