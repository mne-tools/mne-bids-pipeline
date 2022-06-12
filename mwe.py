from run import process


process(
    config='~/Development/mne-bids-pipeline/tests/configs/config_ERP_CORE.py',
    task="N400",
    # config='~/Development/mne-bids-pipeline/tests/configs/config_ds000117.py',
    # config='~/Development/mne-bids-pipeline/tests/configs/config_ds003104.py',
    # steps=('sensor/decoding_full', 'sensor/decoding_time', 'sensor/group_average',)
    # steps=('sensor/group_average', 'report',),
    # steps=('sensor/decoding_entire_epochs',)
    steps=('report',),
    n_jobs='8',
    # subject='015',
    # steps=('sensor/decoding_time_by_time', 'sensor/group_average')
    # steps=('sensor/sliding_estimator', 'sensor/group_average', 'report'),
    # steps=('sensor/group_average', 'report'),
    # task='ERN'
    # steps=('sensor/group_average', 'report'),
    # config='~/Development/mne-bids-pipeline/tests/configs/config_ds000248_no_mri.py',
)
