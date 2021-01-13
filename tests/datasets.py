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
    include: List[str]
    exclude: List[str]


DATASET_OPTIONS: Dict[str, DATASET_OPTIONS_T] = {
    'eeg_matchingpennies': {
        'git': 'https://github.com/sappelhoff/eeg_matchingpennies',
        'openneuro': '',
        'include': ['sub-05'],
        'exclude': []
    },
    'ds003104': {  # Anonymized "somato" dataset.
        'git': '',
        'openneuro': 'ds003104',
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
        'include': ['sub-0001/meg/sub-0001_task-AEF_run-01_meg.ds',
                    'sub-0001/meg/sub-0001_task-AEF_run-01_meg.json',
                    'sub-0001/meg/sub-0001_task-AEF_run-01_channels.tsv'],
        'exclude': []
    },
    'ds000248': {
        'git': '',
        'openneuro': 'ds000248',
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
        'include': ['sub-01'],
        'exclude': []
    },
    'ds000248_T1_BEM': {
        'git': '',
        'openneuro': 'ds000248',
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
        'include': ['derivatives/freesurfer/subjects/sub-01'],
        'exclude': [
            'derivatives/freesurfer/subjects/sub-01/mri/aparc+aseg.mgz',
            'derivatives/freesurfer/subjects/sub-01/mri/aparc.DKTatlas+aseg.mgz',  # noqa: E501
            'derivatives/freesurfer/subjects/sub-01/mri/aparc.DKTatlas+aseg.mgz',  # noqa: E501
            'derivatives/freesurfer/subjects/sub-01/mri/aparc.a2009s+aseg.mgz'
        ],
    },
    'ds000117': {
        'git': '',
        'openneuro': 'ds000117',
        'include': [
            'sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_coordsystem.json',  # noqa: E501
            'sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_events.tsv',  # noqa: E501
            'sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_meg.fif',  # noqa: E501
            'sub-01/ses-meg/meg/sub-01_ses-meg_headshape.pos'
        ],
        'exclude': []
    },
    'ds001810': {
        'git': '',
        'openneuro': 'ds001810',
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
        'include': ['sub-01',
                    'sub-emptyroom/ses-19111211'],
        'exclude': []
    }
}
