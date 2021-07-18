"""Somato
"""
study_name = 'MNE-somato-data-anonymized'
bids_root = '~/mne_data/ds003104'
deriv_root = '~/mne_data/derivatives/mne-bids-pipeline/ds003104'
subjects_dir = f'{bids_root}/derivatives/freesurfer/subjects'

conditions = ['somato_event1']
ch_types = ['meg']
