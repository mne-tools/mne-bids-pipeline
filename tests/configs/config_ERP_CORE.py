"""
ERP-CORE: Word pair judgment.
"""
import os

study_name = 'ERP-CORE'
bids_root = '~/mne_data/ERP_CORE'

task =  os.environ.get('MNE_BIDS_STUDY_TASK')
sessions = [task]

subjects = ['015', '016', '017', '018', '019']


ch_types = ['eeg']
interactive = False

resample_sfreq = 256

eeg_template_montage = 'standard_1005'
eeg_reference = ['P9', 'P10']
eeg_bipolar_channels = {'HEOG': ('HEOG_left', 'HEOG_right'),
                        'VEOG': ('VEOG_lower', 'FP2')}
drop_channels = ['HEOG_left', 'HEOG_right', 'VEOG_lower']

l_freq = 0.1
h_freq = None

decode = True

use_ssp = False
use_ica = True
ica_n_components = 30 - len(eeg_reference)
ica_max_iterations = 1000
ica_eog_threshold = 2

run_source_estimation = False

on_error = 'abort'
N_JOBS = 10

if task == 'N400':
    rename_events = {
        'response/201': 'response/correct',
        'response/202': 'response/error',

        'stimulus/111': 'stimulus/prime/related',
        'stimulus/112': 'stimulus/prime/related',
        'stimulus/121': 'stimulus/prime/unrelated',
        'stimulus/122': 'stimulus/prime/unrelated',

        'stimulus/211': 'stimulus/target/related',
        'stimulus/212': 'stimulus/target/related',
        'stimulus/221': 'stimulus/target/unrelated',
        'stimulus/222': 'stimulus/target/unrelated',
    }

    epochs_tim = -0.2
    epochs_tmax = 0.8
    epochs_metadata_tmin = 0
    epochs_metadata_tmax = 1.5
    epochs_metadata_keep_first = ['stimulus/target', 'response']
    baseline = (-0.2, 0)

    conditions = {
        'related': '`first_stimulus/target` == "related" and '
                'first_response == "correct"',
        'unrelated': '`first_stimulus/target` == "unrelated" and '
                    'first_response == "correct"'
    }

    contrasts = [('unrelated', 'related')]
else:
    raise RuntimeError(f'Task {task} not currently supported')
