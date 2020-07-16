study_name = 'ds003129'
runs = ['01']
l_freq = .3
h_freq = 100.
decim = 4
subjects_list = ['0001']
ch_types = ['meg']
reject = dict(mag=4e-12, eog=250e-6)
conditions = ['standard', 'deviant', 'button']
contrasts = [('deviant', 'standard')]
decode = True
