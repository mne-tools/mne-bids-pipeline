"""
MNE Sample Data: BEM from T1 images
"""

study_name = 'ds000248'
bids_root = '~/mne_data/ds000248'
deriv_root = '~/mne_data/derivatives/mne-bids-pipeline/ds000248_T1_BEM'
subjects_dir = f'{bids_root}/derivatives/freesurfer/subjects'

subjects = ['01']
conditions = ['Auditory']

ch_types = ['meg']

bem_mri_images = 'T1'
recreate_bem = True
freesurfer_verbose = True  # Prevent the CI from canceling the job prematurely
