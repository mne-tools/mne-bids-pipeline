"""
MNE Sample Data: Head surfaces from FreeSurfer surfaces for coregistration step
"""

study_name = 'ds000248'
bids_root = '~/mne_data/ds000248'
deriv_root = '~/mne_data/mne-bids-pipeline/tests/ds000248_coreg_surfaces'
subjects_dir = f'{bids_root}/derivatives/freesurfer/subjects'

subjects = ['01']
conditions = ['Auditory']
ch_types = ['meg']
