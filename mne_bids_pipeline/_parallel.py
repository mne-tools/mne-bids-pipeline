"""Parallelization."""

import os
from typing import Literal

import joblib

from ._logging import logger
from ._config_utils import get_deriv_root


def get_n_jobs(config) -> int:
    env = os.environ
    if config.interactive:
        n_jobs = 1
    elif env.get('MNE_BIDS_STUDY_NJOBS'):
        n_jobs = int(env['MNE_BIDS_STUDY_NJOBS'])
    else:
        n_jobs = config.N_JOBS

    if n_jobs < 0:
        n_cores = joblib.cpu_count()
        n_jobs = min(n_cores + n_jobs + 1, n_cores)

    return n_jobs


dask_client = None


def setup_dask_client(config):
    global dask_client

    import dask
    from dask.distributed import Client

    if dask_client is not None:
        return

    # n_workers = multiprocessing.cpu_count()  # FIXME should use N_JOBS
    n_workers = get_n_jobs(config)
    logger.info(f'👾 Initializing Dask client with {n_workers} workers …')

    if config.dask_temp_dir is None:
        this_dask_temp_dir = get_deriv_root(config) / ".dask-worker-space"
    else:
        this_dask_temp_dir = config.dask_temp_dir

    logger.info(f'📂 Temporary directory is: {this_dask_temp_dir}')
    dask.config.set(
        {
            'temporary-directory': this_dask_temp_dir,
            'distributed.worker.memory.pause': 0.8,
            # fraction of memory that can be utilized before the nanny
            # process will terminate the worker
            'distributed.worker.memory.terminate': 1.0,
            # TODO spilling to disk currently doesn't work reliably for us,
            # as Dask cannot spill "unmanaged" memory – and most of what we
            # see currently is, in fact, "unmanaged". Needs thorough
            # investigation.
            'distributed.worker.memory.spill': False
        }
    )
    client = Client(  # noqa: F841
        memory_limit=config.dask_worker_memory_limit,
        n_workers=n_workers,
        threads_per_worker=1,
        name='mne-bids-pipeline'
    )
    client.auto_restart = False  # don't restart killed workers

    dashboard_url = client.dashboard_link
    logger.info(
        f'⏱  The Dask client is ready. Open {dashboard_url} '
        f'to monitor the workers.\n'
    )

    if config.dask_open_dashboard:
        import webbrowser
        webbrowser.open(url=dashboard_url, autoraise=True)

    # Update global variable
    dask_client = client


def get_parallel_backend_name(config) -> Literal['dask', 'loky']:
    if config.parallel_backend == 'loky' or get_n_jobs(config) == 1:
        return 'loky'
    elif config.parallel_backend == 'dask':
        # Disable interactive plotting backend
        import matplotlib
        matplotlib.use('Agg')
        return 'dask'
    else:
        raise ValueError(
            f'Unknown parallel backend: {config.parallel_backend}')


def get_parallel_backend(config):
    import joblib

    backend = get_parallel_backend_name(config)
    kwargs = {
        'n_jobs': get_n_jobs(config)
    }

    if backend == "loky":
        kwargs['inner_max_num_threads'] = 1
    else:
        setup_dask_client(config)

    return joblib.parallel_backend(
        backend,
        **kwargs
    )


def parallel_func(func, *, config):
    if get_parallel_backend_name(config) == 'loky':
        if get_n_jobs(config) == 1:
            my_func = func
            parallel = list
        else:
            from joblib import Parallel, delayed
            parallel = Parallel()
            my_func = delayed(func)
    else:  # Dask
        from joblib import Parallel, delayed
        parallel = Parallel()
        my_func = delayed(func)

    return parallel, my_func
