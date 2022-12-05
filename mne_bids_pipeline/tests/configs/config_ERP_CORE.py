"""
ERP CORE

This example demonstrate how to process 5 participants from the
[ERP CORE](https://erpinfo.org/erp-core) dataset. It shows how to obtain 7 ERP
components from a total of 6 experimental tasks:

- N170 (face perception)
- MMN (passive auditory oddball)
- N2pc (visual search)
- N400 (word pair judgment)
- P3b (active visual oddball)
- LRP and ERN (flankers task)

## Dataset information

- **Authors:** Emily S. Kappenman, Jaclyn L. Farrens, Wendy Zhang,
                       Andrew X. Stewart, and Steven J. Luck
- **License:** CC-BY-4.0
- **URL:** [https://erpinfo.org/erp-core](https://erpinfo.org/erp-core)
- **Citation:** Kappenman, E., Farrens, J., Zhang, W., Stewart, A. X.,
                & Luck, S. J. (2021). ERP CORE: An open resource for human
                event-related potential research. *NeuroImage* 225: 117465.
                [https://doi.org/10.1016/j.neuroimage.2020.117465](https://doi.org/10.1016/j.neuroimage.2020.117465)
"""
import argparse
import mne
import sys

study_name = 'ERP-CORE'
bids_root = '~/mne_data/ERP_CORE'
deriv_root = '~/mne_data/derivatives/mne-bids-pipeline/ERP_CORE'

# Find the --task option
args = [arg for arg in sys.argv
        if arg.startswith('--task') or not arg.startswith('-')]
parser = argparse.ArgumentParser()
parser.add_argument('ignored', nargs='*')
parser.add_argument(
    '--task', choices=('N400', 'ERN', 'LRP', 'MMN', 'N2pc', 'N170', 'P3'),
    required=True)
task = parser.parse_args(args).task
sessions = [task]

subjects = ['015', '016', '017', '018', '019']

ch_types = ['eeg']
interactive = False

raw_resample_sfreq = 128

eeg_template_montage = mne.channels.make_standard_montage('standard_1005')
eeg_bipolar_channels = {'HEOG': ('HEOG_left', 'HEOG_right'),
                        'VEOG': ('VEOG_lower', 'FP2')}
drop_channels = ['HEOG_left', 'HEOG_right', 'VEOG_lower']
eog_channels = ['HEOG', 'VEOG']

l_freq = 0.1
h_freq = 40

decode = True
decoding_time_generalization = True
decoding_time_generalization_decim = 2

find_breaks = True
min_break_duration = 10
t_break_annot_start_after_previous_event = 3.0
t_break_annot_stop_before_next_event = 1.5

ica_reject = dict(eeg=350e-6, eog=500e-6)
reject = 'autoreject_global'

spatial_filter = 'ica'
ica_max_iterations = 1000
ica_eog_threshold = 2
ica_decim = 2  # speed up ICA fitting

run_source_estimation = False

on_rename_missing_events = 'ignore'

parallel_backend = 'dask'
dask_worker_memory_limit = '2G'
N_JOBS = 2

if task == 'N400':
    dask_open_dashboard = True

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

    eeg_reference = ['P9', 'P10']
    ica_n_components = 30 - len(eeg_reference)
    epochs_tmin = -0.2
    epochs_tmax = 0.8
    epochs_metadata_tmin = 0
    epochs_metadata_tmax = 1.5
    epochs_metadata_keep_first = ['stimulus/target', 'response']
    baseline = (None, 0)

    conditions = {
        'related': '`first_stimulus/target` == "related" and '
                   'first_response == "correct"',
        'unrelated': '`first_stimulus/target` == "unrelated" and '
                     'first_response == "correct"'
    }
    contrasts = [('unrelated', 'related')]
    cluster_forming_t_threshold = 1.5      # Only for testing!
    cluster_permutation_p_threshold = 0.2  # Only for testing!
elif task == 'ERN':
    rename_events = {
        'stimulus/11': 'compatible/left',
        'stimulus/12': 'compatible/right',
        'stimulus/21': 'incompatible/left',
        'stimulus/22': 'incompatible/right',

        'response/111': 'response/correct',
        'response/112': 'response/incorrect',
        'response/121': 'response/correct',
        'response/122': 'response/incorrect',
        'response/211': 'response/incorrect',
        'response/212': 'response/correct',
        'response/221': 'response/incorrect',
        'response/222': 'response/correct',
    }

    eeg_reference = ['P9', 'P10']
    ica_n_components = 30 - len(eeg_reference)
    epochs_tmin = -0.6
    epochs_tmax = 0.4
    baseline = (-0.4, -0.2)
    conditions = ['response/correct', 'response/incorrect']
    contrasts = [('response/incorrect', 'response/correct')]
    cluster_forming_t_threshold = 5        # Only for testing!
    cluster_permutation_p_threshold = 0.2  # Only for testing!
    decoding_csp = True
    decoding_csp_freqs = {
        'theta': [4, 7],
        'alpha': [8, 12],
        'beta': [13, 20, 30],
    }
    decoding_csp_times = [-0.2, 0., 0.2, 0.4]
elif task == 'LRP':
    rename_events = {
        'stimulus/11': 'compatible/left',
        'stimulus/12': 'compatible/right',
        'stimulus/21': 'incompatible/left',
        'stimulus/22': 'incompatible/right',

        'response/111': 'response/left/correct',
        'response/112': 'response/left/incorrect',
        'response/121': 'response/left/correct',
        'response/122': 'response/left/incorrect',
        'response/211': 'response/right/incorrect',
        'response/212': 'response/right/correct',
        'response/221': 'response/right/incorrect',
        'response/222': 'response/right/correct',
    }

    eeg_reference = ['P9', 'P10']
    ica_n_components = 30 - len(eeg_reference)
    epochs_tmin = -0.8
    epochs_tmax = 0.2
    baseline = (None, -0.6)
    conditions = ['response/left', 'response/right']
    contrasts = [('response/right', 'response/left')]  # contralateral vs ipsi
elif task == 'MMN':
    rename_events = {
        'stimulus/70': 'stimulus/deviant',
        'stimulus/80': 'stimulus/standard'
    }

    eeg_reference = ['P9', 'P10']
    ica_n_components = 30 - len(eeg_reference)
    epochs_tmin = -0.2
    epochs_tmax = 0.8
    baseline = (None, 0)
    conditions = ['stimulus/standard', 'stimulus/deviant']
    contrasts = [('stimulus/deviant', 'stimulus/standard')]
elif task == 'N2pc':
    rename_events = {
        'response/201': 'response/correct',
        'response/202': 'response/error',

        'stimulus/111': 'stimulus/blue/left',
        'stimulus/112': 'stimulus/blue/left',
        'stimulus/121': 'stimulus/blue/right',
        'stimulus/122': 'stimulus/blue/right',
        'stimulus/211': 'stimulus/pink/left',
        'stimulus/212': 'stimulus/pink/left',
        'stimulus/221': 'stimulus/pink/right',
        'stimulus/222': 'stimulus/pink/right'
    }

    eeg_reference = ['P9', 'P10']
    ica_n_components = 30 - len(eeg_reference)
    epochs_tmin = -0.2
    epochs_tmax = 0.8
    baseline = (None, 0)
    conditions = ['stimulus/right', 'stimulus/left']
    contrasts = [('stimulus/right', 'stimulus/left')]  # Contralteral vs ipsi
elif task == 'N170':
    rename_events = {
        'response/201': 'response/correct',
        'response/202': 'response/error'
    }

    eeg_reference = 'average'
    ica_n_components = 30 - 1
    for i in range(1, 180+1):
        orig_name = f'stimulus/{i}'

        if 1 <= i <= 40:
            new_name = 'stimulus/face/normal'
        elif 41 <= i <= 80:
            new_name = 'stimulus/car/normal'
        elif 101 <= i <= 140:
            new_name = 'stimulus/face/scrambled'
        elif 141 <= i <= 180:
            new_name = 'stimulus/car/scrambled'
        else:
            continue

        rename_events[orig_name] = new_name

    epochs_tmin = -0.2
    epochs_tmax = 0.8
    baseline = (None, 0)
    conditions = ['stimulus/face/normal', 'stimulus/car/normal']
    contrasts = [('stimulus/face/normal', 'stimulus/car/normal')]
elif task == 'P3':
    rename_events = {
        'response/201': 'response/correct',
        'response/202': 'response/incorrect',

        'stimulus/11': 'stimulus/target/11',
        'stimulus/22': 'stimulus/target/22',
        'stimulus/33': 'stimulus/target/33',
        'stimulus/44': 'stimulus/target/44',
        'stimulus/55': 'stimulus/target/55',
        'stimulus/21': 'stimulus/non-target/21',
        'stimulus/31': 'stimulus/non-target/31',
        'stimulus/41': 'stimulus/non-target/41',
        'stimulus/51': 'stimulus/non-target/51',
        'stimulus/12': 'stimulus/non-target/12',
        'stimulus/32': 'stimulus/non-target/32',
        'stimulus/42': 'stimulus/non-target/42',
        'stimulus/52': 'stimulus/non-target/52',
        'stimulus/13': 'stimulus/non-target/13',
        'stimulus/23': 'stimulus/non-target/23',
        'stimulus/43': 'stimulus/non-target/43',
        'stimulus/53': 'stimulus/non-target/53',
        'stimulus/14': 'stimulus/non-target/14',
        'stimulus/24': 'stimulus/non-target/24',
        'stimulus/34': 'stimulus/non-target/34',
        'stimulus/54': 'stimulus/non-target/54',
        'stimulus/15': 'stimulus/non-target/15',
        'stimulus/25': 'stimulus/non-target/25',
        'stimulus/35': 'stimulus/non-target/35',
        'stimulus/45': 'stimulus/non-target/45'
    }

    eeg_reference = ['P9', 'P10']
    ica_n_components = 30 - len(eeg_reference)
    epochs_tmin = -0.2
    epochs_tmax = 0.8
    baseline = (None, 0)
    conditions = ['stimulus/target', 'stimulus/non-target']
    contrasts = [('stimulus/target', 'stimulus/non-target')]
    cluster_forming_t_threshold = 0.8      # Only for testing!
    cluster_permutation_p_threshold = 0.2  # Only for testing!
else:
    raise RuntimeError(f'Task {task} not currently supported')
