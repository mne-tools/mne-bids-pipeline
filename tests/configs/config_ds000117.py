"""Configuration file for the ds000117 dataset.

Set the `MNE_BIDS_STUDY_CONFIG` environment variable to
"config_ds000117" to overwrite `config.py` with the values specified
below.

Download ds000117 from OpenNeuro: https://github.com/OpenNeuroDatasets/ds000117

export MNE_BIDS_STUDY_CONFIG=config_ds000117
export BIDS_ROOT=~/data/ds000117

"""
from bids import BIDSLayout

study_name = 'ds000117'
task = 'VisualFaces'
kind = 'meg'
ch_types = ['meg']
runs = ['06']
sessions = ['01']
plot = False
acq = None
bids_root = '~/data/ds000117'
subjects_dir = bids_root + '/derivatives/freesurfer/subjects'
layout = BIDSLayout(bids_root)
# subjects_list = layout.get(return_type='id', target='subject')
subjects_list = ['01']

use_maxwell_filter = True
mf_ctc_fname = None
mf_cal_fname = None

reject = {'grad': 4000e-13, 'mag': 4e-12}
conditions = ['face', 'scrambled']
decoding_conditions = [('face', 'scrambled')]
use_ssp = False
# use_ssp = True
use_ica = True
use_ica = False
