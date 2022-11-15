"""Definition of the testing datasets."""

from typing import Dict, List, TypedDict


class DATASET_OPTIONS_T(TypedDict):
    git: str
    openneuro: str
    osf: str
    web: str
    include: List[str]
    exclude: List[str]


DATASET_OPTIONS: Dict[str, DATASET_OPTIONS_T] = {
    'ERP_CORE': {
        'git': '',
        'openneuro': '',
        'osf': '',  # original dataset: '9f5w7'
        'web': 'https://osf.io/3zk6n/download',
        'include': [],
        'exclude': []
    },
    'eeg_matchingpennies': {
        'git': 'https://gin.g-node.org/sappelhoff/eeg_matchingpennies',
        'openneuro': '',
        'osf': '',  # original dataset: 'cj2dr'
        'web': '',
        'include': ['sub-05'],
        'exclude': []
    },
    'ds003104': {  # Anonymized "somato" dataset.
        'git': '',
        'openneuro': 'ds003104',
        'osf': '',
        'web': '',
        'include': [
            'sub-01',
            'derivatives/freesurfer/subjects'
        ],
        'exclude': [
            'derivatives/freesurfer/subjects/01/mri/aparc+aseg.mgz',
            'derivatives/freesurfer/subjects/01/mri/aparc.DKTatlas+aseg.mgz',
            'derivatives/freesurfer/subjects/01/mri/aparc.a2009s+aseg.mgz'
        ]
    },
    'ds000246': {
        'git': '',
        'openneuro': 'ds000246',
        'osf': '',
        'web': '',
        'include': [
            'sub-0001/meg/sub-0001_task-AEF_run-01_meg.ds',
            'sub-0001/meg/sub-0001_task-AEF_run-01_meg.json',
            'sub-0001/meg/sub-0001_task-AEF_run-01_channels.tsv'
        ],
        'exclude': []
    },
    'ds000247': {
        'git': '',
        'openneuro': 'ds000247',
        'osf': '',
        'web': '',
        'include': ['sub-0002/ses-01/meg'],
        'exclude': []
    },
    'ds000248': {
        'git': '',
        'openneuro': 'ds000248',
        'osf': '',
        'web': '',
        'include': [
            'sub-01',
            'sub-emptyroom',
            'derivatives/freesurfer/subjects'
        ],
        'exclude': [
            'derivatives/freesurfer/subjects/fsaverage/mri/aparc.a2005s+aseg.mgz',  # noqa: E501
            'derivatives/freesurfer/subjects/fsaverage/mri/aparc+aseg.mgz',
            'derivatives/freesurfer/subjects/fsaverage/mri/aparc.a2009s+aseg.mgz',  # noqa: E501
            'derivatives/freesurfer/subjects/fsaverage/xhemi/mri/aparc+aseg.mgz',  # noqa: E501
            'derivatives/freesurfer/subjects/sub-01/mri/aparc+aseg.mgz',
            'derivatives/freesurfer/subjects/sub-01/mri/aparc.DKTatlas+aseg.mgz',  # noqa: E501
            'derivatives/freesurfer/subjects/sub-01/mri/aparc.DKTatlas+aseg.mgz',  # noqa: E501
            'derivatives/freesurfer/subjects/sub-01/mri/aparc.a2009s+aseg.mgz'
        ],
    },
    'ds000117': {
        'git': '',
        'openneuro': 'ds000117',
        'osf': '',
        'web': '',
        'include': [
            'sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_*',  # noqa: E501
            'sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_*',  # noqa: E501
            'sub-01/ses-meg/meg/sub-01_ses-meg_headshape.pos',
            'sub-01/ses-meg/*.tsv',
            'sub-01/ses-meg/*.json',
            'sub-emptyroom/ses-20090409',
            'derivatives/meg_derivatives/ct_sparse.fif',
            'derivatives/meg_derivatives/sss_cal.dat',
        ],
        'exclude': []
    },
    'ds003775': {
        'git': '',
        'openneuro': 'ds003775',
        'osf': '',
        'web': '',
        'include': ['sub-01'],
        'exclude': []
    },
    'ds001810': {
        'git': '',
        'openneuro': 'ds001810',
        'osf': '',
        'web': '',
        'include': ['sub-01'],
        'exclude': []
    },
    'ds001971': {
        'git': '',
        'openneuro': 'ds001971',
        'osf': '',
        'web': '',
        'include': [
            'sub-001/eeg/sub-001_task-AudioCueWalkingStudy_run-01_events.tsv',
            'sub-001/eeg/sub-001_task-AudioCueWalkingStudy_run-01_eeg.set',
            'sub-001/eeg/sub-001_task-AudioCueWalkingStudy_run-01_eeg.fdt',
            'sub-001/eeg/sub-001_task-AudioCueWalkingStudy_run-01_eeg.json',
            'sub-001/eeg/sub-001_task-AudioCueWalkingStudy_run-01_electrodes.tsv',  # noqa: E501
            'sub-001/eeg/sub-001_task-AudioCueWalkingStudy_run-01_coordsystem.json',  # noqa: E501
            'sub-001/eeg/sub-001_task-AudioCueWalkingStudy_run-01_channels.tsv'  # noqa: E501
        ],
        'exclude': []
    },
    'ds003392': {
        'git': '',
        'openneuro': 'ds003392',
        'osf': '',
        'web': '',
        'include': [
            'sub-01',
            'sub-emptyroom/ses-19111211'
        ],
        'exclude': []
    },
    'ds004107': {
        'git': '',
        'openneuro': 'ds004107',
        'osf': '',
        'web': '',
        'include': [
            'sub-mind002/ses-01/meg/*coordsystem*',
            'sub-mind002/ses-01/meg/*auditory*',
        ],
        'exclude': []
    },
    'ds004229': {
        'git': '',
        'openneuro': 'ds004229',
        'osf': '',
        'web': '',
        'include': [
            'sub-102',
            'sub-emptyroom/ses-20000101',
            'derivatives/meg_derivatives/ct_sparse.fif',
            'derivatives/meg_derivatives/sss_cal.dat',
        ],
        'exclude': []
    },
}
