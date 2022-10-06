"""
MNE Sample Data: Head surfaces from FreeSurfer surfaces for coregistration step
"""

study_name = 'ds000248'
bids_root = '~/mne_data/ds000248'
deriv_root = '~/mne_data/derivatives/mne-bids-pipeline/ds000248_coreg_surfaces'
subjects_dir = f'{bids_root}/derivatives/freesurfer/subjects'

subjects = ['01']
conditions = ['Auditory']
ch_types = ['meg']

recreate_scalp_surface = True
