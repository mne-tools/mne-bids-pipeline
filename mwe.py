from run import process

process(
    # config='~/Development/mne-bids-pipeline/tests/configs/config_ds000247.py',
    # config='~/Development/mne-bids-pipeline/tests/configs/config_ds000248_no_mri.py',
    config='/storage/store2/data/time_in_wm/code/bids_pipeline_config.py',
    subject='220',
    steps='sensor/time_frequency_csp',
    n_jobs='1'
)
