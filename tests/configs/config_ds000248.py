study_name = 'ds000248'
subjects_list = ['01']
rename_events = {'Smiley': 'Emoji',
                 'Button': 'Switch'}
conditions = ['Auditory', 'Visual', 'Auditory/Left', 'Auditory/Right']
contrasts = [('Visual', 'Auditory'),
             ('Auditory/Right', 'Auditory/Left')]

ch_types = ['meg']
mf_ctc_fname = None
mf_cal_fname = None
mf_reference_run = '01'
find_flat_channels_meg = True
find_noisy_channels_meg = True
use_maxwell_filter = True
noise_cov = 'emptyroom'
