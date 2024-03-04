"""Definition of the testing datasets."""

from typing import TypedDict


# If not supplied below, the effective defaults are listed in comments
class DATASET_OPTIONS_T(TypedDict, total=False):
    """A container for sources, hash, include and excludes of a dataset."""

    git: str  # ""
    openneuro: str  # ""
    osf: str  # ""
    web: str  # ""
    mne: str  # ""
    include: list[str]  # []
    exclude: list[str]  # []
    hash: str  # ""


DATASET_OPTIONS: dict[str, DATASET_OPTIONS_T] = {
    "ERP_CORE": {
        # original dataset: "osf": "9f5w7"
        "web": "https://osf.io/3zk6n/download?version=2",
        "hash": "sha256:ddc94a7c9ba1922637f2770592dd51c019d341bf6bc8558e663e1979a4cb002f",  # noqa: E501
    },
    "eeg_matchingpennies": {
        "web": "https://osf.io/download/8rbfk?version=1",
        "hash": "sha256:06bfbe52c50b9343b6b8d2a5de3dd33e66ad9303f7f6bfbe6868c3c7c375fafd",  # noqa: E501
    },
    "ds003104": {  # Anonymized "somato" dataset.
        "openneuro": "ds003104",
        "include": ["sub-01", "derivatives/freesurfer/subjects"],
        "exclude": [
            "derivatives/freesurfer/subjects/01/mri/aparc+aseg.mgz",
            "derivatives/freesurfer/subjects/01/mri/aparc.DKTatlas+aseg.mgz",
            "derivatives/freesurfer/subjects/01/mri/aparc.a2009s+aseg.mgz",
        ],
    },
    "ds000246": {
        "openneuro": "ds000246",
        "include": [
            "sub-0001/meg/sub-0001_task-AEF_run-01_meg.ds",
            "sub-0001/meg/sub-0001_task-AEF_run-01_meg.json",
            "sub-0001/meg/sub-0001_task-AEF_run-01_channels.tsv",
        ],
    },
    "ds000247": {
        "openneuro": "ds000247",
        "include": ["sub-0002/ses-01/meg"],
    },
    "ds000248": {
        "openneuro": "ds000248",
        "include": ["sub-01", "sub-emptyroom", "derivatives/freesurfer/subjects"],
        "exclude": [
            "derivatives/freesurfer/subjects/fsaverage/mri/aparc.a2005s+aseg.mgz",  # noqa: E501
            "derivatives/freesurfer/subjects/fsaverage/mri/aparc+aseg.mgz",
            "derivatives/freesurfer/subjects/fsaverage/mri/aparc.a2009s+aseg.mgz",  # noqa: E501
            "derivatives/freesurfer/subjects/fsaverage/xhemi/mri/aparc+aseg.mgz",  # noqa: E501
            "derivatives/freesurfer/subjects/sub-01/mri/aparc+aseg.mgz",
            "derivatives/freesurfer/subjects/sub-01/mri/aparc.DKTatlas+aseg.mgz",  # noqa: E501
            "derivatives/freesurfer/subjects/sub-01/mri/aparc.DKTatlas+aseg.mgz",  # noqa: E501
            "derivatives/freesurfer/subjects/sub-01/mri/aparc.a2009s+aseg.mgz",
        ],
    },
    "ds000117": {
        "openneuro": "ds000117",
        "include": [
            "sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_*",  # noqa: E501
            "sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_*",  # noqa: E501
            "sub-01/ses-meg/meg/sub-01_ses-meg_headshape.pos",
            "sub-01/ses-meg/*.tsv",
            "sub-01/ses-meg/*.json",
            "sub-emptyroom/ses-20090409",
            "derivatives/meg_derivatives/ct_sparse.fif",
            "derivatives/meg_derivatives/sss_cal.dat",
        ],
    },
    "ds003775": {
        "openneuro": "ds003775",
        "include": ["sub-010"],
        # See https://github.com/OpenNeuroOrg/openneuro/issues/2976
        "exclude": ["sub-010/ses-t1/sub-010_ses-t1_scans.tsv"],
    },
    "ds001810": {
        "openneuro": "ds001810",
        "include": ["sub-01"],
    },
    "ds001971": {
        "openneuro": "ds001971",
        "include": [
            "sub-001/eeg/sub-001_task-AudioCueWalkingStudy_run-01_events.tsv",
            "sub-001/eeg/sub-001_task-AudioCueWalkingStudy_run-01_eeg.set",
            "sub-001/eeg/sub-001_task-AudioCueWalkingStudy_run-01_eeg.fdt",
            "sub-001/eeg/sub-001_task-AudioCueWalkingStudy_run-01_eeg.json",
            "sub-001/eeg/sub-001_task-AudioCueWalkingStudy_run-01_electrodes.tsv",  # noqa: E501
            "sub-001/eeg/sub-001_task-AudioCueWalkingStudy_run-01_coordsystem.json",  # noqa: E501
            "sub-001/eeg/sub-001_task-AudioCueWalkingStudy_run-01_channels.tsv",  # noqa: E501
        ],
    },
    "ds003392": {
        "openneuro": "ds003392",
        "include": ["sub-01", "sub-emptyroom/ses-19111211"],
    },
    "ds004107": {
        "openneuro": "ds004107",
        "include": [
            "sub-mind002/ses-01/meg/*coordsystem*",
            "sub-mind002/ses-01/meg/*auditory*",
        ],
    },
    "ds004229": {
        "openneuro": "ds004229",
        "include": [
            "sub-102",
            "sub-emptyroom/ses-20000101",
        ],
    },
    "MNE-phantom-KIT-data": {
        "mne": "phantom_kit",
    },
}
