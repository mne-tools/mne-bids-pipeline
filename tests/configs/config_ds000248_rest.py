"""
MNE Sample Data
"""

study_name = 'ds000248'
bids_root = '~/mne_data/ds000248'

subjects = ['01']
rename_events = {'Smiley': 'Emoji',
                 'Button': 'Switch'}
conditions = ['Auditory', 'Visual', 'Auditory/Left', 'Auditory/Right']
contrasts = [('Visual', 'Auditory'),
             ('Auditory/Right', 'Auditory/Left')]

time_frequency_conditions = ['Auditory', 'Visual']

ch_types = ['meg']
mf_reference_run = '01'
find_flat_channels_meg = True
find_noisy_channels_meg = True
use_maxwell_filter = True
process_er = True
noise_cov = 'emptyroom'

spatial_filter = 'ssp'

bem_mri_images = 'FLASH'
recreate_bem = True
recreate_scalp_surface = True


def mri_t1_path_generator(bids_path):
    # don't really do any modifications – just for testing!
    return bids_path
