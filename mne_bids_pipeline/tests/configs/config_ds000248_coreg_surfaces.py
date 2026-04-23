"""MNE Sample Data: Head surfaces from FreeSurfer surfaces for coregistration step."""

bids_root = "~/mne_data/ds000248"
deriv_root = "~/mne_data/derivatives/mne-bids-pipeline/ds000248_coreg_surfaces"
subjects_dir = f"{bids_root}/derivatives/freesurfer/subjects"
ignore_warnings = [
    r"Surface outer skin\s+has topological defects",
    r"Surface outer skin\s+is not complete",
]

subjects = ["01"]
conditions = ["Auditory"]
ch_types = ["meg"]

recreate_scalp_surface = True
