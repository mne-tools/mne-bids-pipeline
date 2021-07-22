"""
MNE Sample Data
"""
import numpy as np

study_name = 'ds000248'
bids_root = '~/mne_data/ds000248'

subjects = ['01']
rename_events = {'Smiley': 'Emoji',
                 'Button': 'Switch'}
conditions = ['Auditory', 'Visual', 'Auditory/Left', 'Auditory/Right']
contrasts = [('Visual', 'Auditory'),
             ('Auditory/Right', 'Auditory/Left')]

time_frequency_conditions = ['Auditory', 'Visual']

# We have to deal with the filter warning
# RuntimeWarning: filter_length (455) is longer than the signal (421), distortion is likely. Reduce filter length or filter a longer signal.
# that's why we do not use csp_freqs = np.linspace(10, 40, 5) but:
csp_freqs = np.linspace(20, 40, 3)  # ok.

ch_types = ['meg']
mf_reference_run = '01'
find_flat_channels_meg = True
find_noisy_channels_meg = True
use_maxwell_filter = True
process_er = True
noise_cov = 'emptyroom'

spatial_filter = 'ssp'
n_proj_eog = dict(n_mag=1, n_grad=1, n_eeg=1)
n_proj_ecg = dict(n_mag=1, n_grad=1, n_eeg=0)
ecg_proj_from_average = True
eog_proj_from_average = False

bem_mri_images = 'FLASH'
recreate_bem = True
recreate_scalp_surface = True


def mri_t1_path_generator(bids_path):
    # don't really do any modifications – just for testing!
    return bids_path
