"""Definition of the testing datasets."""

import sys
from typing import Dict, List
if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


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
        'git': 'https://github.com/sappelhoff/eeg_matchingpennies',
        'openneuro': '',
        'osf': '',
        'web': '',
        'include': ['sub-05'],
        'exclude': []
    },
    'ds003104': {  # Anonymized "somato" dataset.
        'git': '',
        'openneuro': 'ds003104',
        'osf': '',
        'web': '',
        'include': ['sub-01',
                    'derivatives/freesurfer/subjects'],
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
        'include': ['sub-0001/meg/sub-0001_task-AEF_run-01_meg.ds',
                    'sub-0001/meg/sub-0001_task-AEF_run-01_meg.json',
                    'sub-0001/meg/sub-0001_task-AEF_run-01_channels.tsv'],
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
        'include': ['sub-01', 'sub-emptyroom',
                    'derivatives/freesurfer/subjects'],
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
    'ds000248_ica': {
        'git': '',
        'openneuro': 'ds000248',
        'osf': '',
        'web': '',
        'include': ['sub-01'],
        'exclude': []
    },
    'ds000248_T1_BEM': {
        'git': '',
        'openneuro': 'ds000248',
        'osf': '',
        'web': '',
        'include': ['derivatives/freesurfer/subjects/sub-01'],
        'exclude': [
            'derivatives/freesurfer/subjects/sub-01/mri/aparc+aseg.mgz',
            'derivatives/freesurfer/subjects/sub-01/mri/aparc.DKTatlas+aseg.mgz',  # noqa: E501
            'derivatives/freesurfer/subjects/sub-01/mri/aparc.DKTatlas+aseg.mgz',  # noqa: E501
            'derivatives/freesurfer/subjects/sub-01/mri/aparc.a2009s+aseg.mgz'
        ],
    },
    'ds000248_FLASH_BEM': {
        'git': '',
        'openneuro': 'ds000248',
        'osf': '',
        'web': '',
        'include': ['derivatives/freesurfer/subjects/sub-01'],
        'exclude': [
            'derivatives/freesurfer/subjects/sub-01/mri/aparc+aseg.mgz',
            'derivatives/freesurfer/subjects/sub-01/mri/aparc.DKTatlas+aseg.mgz',  # noqa: E501
            'derivatives/freesurfer/subjects/sub-01/mri/aparc.DKTatlas+aseg.mgz',  # noqa: E501
            'derivatives/freesurfer/subjects/sub-01/mri/aparc.a2009s+aseg.mgz'
        ],
    },
    'ds000248_no_mri': {
        'git': '',
        'openneuro': 'ds000248',
        'osf': '',
        'web': '',
        'include': ['sub-01'],
        'exclude': ['sub-01/anat'],
    },
    'ds000117': {
        'git': '',
        'openneuro': 'ds000117',
        'osf': '',
        'web': '',
        'include': [
            'sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_coordsystem.json',  # noqa: E501
            'sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_events.tsv',  # noqa: E501
            'sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_meg.fif',  # noqa: E501
            'sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_events.tsv',  # noqa: E501
            'sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_meg.fif',  # noqa: E501
            'sub-01/ses-meg/meg/sub-01_ses-meg_headshape.pos',
            'derivatives/meg_derivatives/ct_sparse.fif',
            'derivatives/meg_derivatives/sss_cal.dat'
        ],
        'exclude': []
    },
    'ds001810': {
        'git': '',
        'openneuro': 'ds001810',
        'osf': '',
        'web': '',
        'include': ['sub-01/ses-anodalpre',
                    'sub-02/ses-anodalpre',
                    'sub-03/ses-anodalpre',
                    'sub-04/ses-anodalpre',
                    'sub-05/ses-anodalpre'],
        'exclude': []
    },
    'ds001971': {
        'git': '',
        'openneuro': 'ds001971',
        'osf': '',
        'web': '',
        'include': [
            'sub-001/eeg/sub-001_task-AudioCueWalkingStudy_run-01_events.tsv'
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
        'include': ['sub-01',
                    'sub-emptyroom/ses-19111211'],
        'exclude': []
    }
}
