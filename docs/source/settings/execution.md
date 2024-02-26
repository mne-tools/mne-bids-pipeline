These options control how the pipeline is executed but should not affect
what outputs get produced.

::: mne_bids_pipeline._config
    options:
      members:
        - n_jobs
        - parallel_backend
        - dask_open_dashboard
        - dask_temp_dir
        - dask_worker_memory_limit
        - log_level
        - mne_log_level
        - on_error
        - memory_location
        - memory_subdir
        - memory_file_method
        - memory_verbose
        - config_validation
