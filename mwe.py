from run import process

process(
    config='~/Development/mne-bids-pipeline/tests/configs/config_ERP_CORE.py',
    # steps='sensor/group_average',
    # steps='sensor/time_frequency_csp',
    steps='report',
    task='N400',
    n_jobs=1
)
