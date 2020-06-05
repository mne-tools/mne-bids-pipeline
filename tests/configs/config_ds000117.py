"""Configuration file for the ds000117 dataset.

Set the `MNE_BIDS_STUDY_CONFIG` environment variable to
"config_ds000117" to overwrite `config.py` with the values specified
below.

Download ds000117 from OpenNeuro: https://github.com/OpenNeuroDatasets/ds000117

export MNE_BIDS_STUDY_CONFIG=config_ds000117
export BIDS_ROOT=~/mne_data/ds000117

"""
import os.path as op

study_name = 'ds000117'
task = 'facerecognition'
ch_types = ['meg']
runs = ['01']
sessions = ['meg']
interactive = False
acq = None
bids_root = op.join(op.expanduser('~'), 'data', 'ds000117')
subjects_dir = op.join(bids_root, 'derivatives', 'freesurfer', 'subjects')
subjects_list = ['01']

use_maxwell_filter = True
mf_ctc_fname = None
mf_cal_fname = None

reject = {'grad': 4000e-13, 'mag': 4e-12}
conditions = ['face', 'scrambled']
contrasts = [('face', 'scrambled')]
decode = True
use_ssp = False
use_ica = False
