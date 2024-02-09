"""Parallelization."""

from types import SimpleNamespace
from typing import Callable, Literal

import joblib
from mne.utils import logger as mne_logger
from mne.utils import use_log_level

from ._logging import _is_testing, gen_log_kwargs, logger


def get_n_jobs(*, exec_params: SimpleNamespace, log_override: bool = False) -> int:
    n_jobs = exec_params.n_jobs
    if n_jobs < 0:
        n_cores = joblib.cpu_count()
        n_jobs = min(n_cores + n_jobs + 1, n_cores)

    # Shim to allow overriding n_jobs for specific steps
    if _is_testing() and hasattr(exec_params, "_n_jobs"):
        from ._run import _get_step_path, _short_step_path

        step_path = _short_step_path(_get_step_path())
        orig_n_jobs = n_jobs
        n_jobs = exec_params._n_jobs.get(step_path, n_jobs)
        if log_override and n_jobs != orig_n_jobs:
            msg = f"Overriding n_jobs: {orig_n_jobs}→{n_jobs}"
            logger.info(**gen_log_kwargs(message=msg, emoji="override"))
    return n_jobs


dask_client = None


def setup_dask_client(*, exec_params: SimpleNamespace) -> None:
    global dask_client

    import dask
    from dask.distributed import Client

    if dask_client is not None:
        return

    n_workers = get_n_jobs(exec_params=exec_params)
    msg = f"Dask initializing with {n_workers} workers …"
    logger.info(**gen_log_kwargs(message=msg, emoji="👾"))

    if exec_params.dask_temp_dir is None:
        this_dask_temp_dir = exec_params.deriv_root / ".dask-worker-space"
    else:
        this_dask_temp_dir = exec_params.dask_temp_dir

    msg = f"Dask temporary directory: {this_dask_temp_dir}"
    logger.info(**gen_log_kwargs(message=msg, emoji="📂"))
    dask.config.set(
        {
            "temporary-directory": this_dask_temp_dir,
            "distributed.worker.memory.pause": 0.8,
            # fraction of memory that can be utilized before the nanny
            # process will terminate the worker
            "distributed.worker.memory.terminate": 1.0,
            # TODO spilling to disk currently doesn't work reliably for us,
            # as Dask cannot spill "unmanaged" memory – and most of what we
            # see currently is, in fact, "unmanaged". Needs thorough
            # investigation.
            "distributed.worker.memory.spill": False,
        }
    )
    client = Client(  # noqa: F841
        memory_limit=exec_params.dask_worker_memory_limit,
        n_workers=n_workers,
        threads_per_worker=1,
        name="mne-bids-pipeline",
    )
    client.auto_restart = False  # don't restart killed workers

    dashboard_url = client.dashboard_link
    msg = "Dask client dashboard: " f"[link={dashboard_url}]{dashboard_url}[/link]"
    logger.info(**gen_log_kwargs(message=msg, emoji="🌎"))

    if exec_params.dask_open_dashboard:
        import webbrowser

        webbrowser.open(url=dashboard_url, autoraise=True)

    # Update global variable
    dask_client = client


def get_parallel_backend_name(
    *,
    exec_params: SimpleNamespace,
) -> Literal["dask", "loky"]:
    if (
        exec_params.parallel_backend == "loky"
        or get_n_jobs(exec_params=exec_params) == 1
    ):
        backend = "loky"
    elif exec_params.parallel_backend == "dask":
        # Disable interactive plotting backend
        import matplotlib

        matplotlib.use("Agg")
        backend = "dask"
    else:
        # TODO: Move to value validation step
        raise ValueError(f"Unknown parallel backend: {exec_params.parallel_backend}")

    return backend


def get_parallel_backend(exec_params: SimpleNamespace) -> joblib.parallel_backend:
    import joblib

    backend = get_parallel_backend_name(exec_params=exec_params)
    kwargs = {
        "n_jobs": get_n_jobs(
            exec_params=exec_params,
            log_override=True,
        )
    }

    if backend == "loky":
        kwargs["inner_max_num_threads"] = 1
    else:
        setup_dask_client(exec_params=exec_params)

    return joblib.parallel_backend(backend, **kwargs)


def parallel_func(func: Callable, *, exec_params: SimpleNamespace):
    if (
        get_parallel_backend_name(exec_params=exec_params) == "loky"
        and get_n_jobs(exec_params=exec_params) == 1
    ):
        my_func = func
        parallel = list
    else:  # Dask or n_jobs > 1
        from joblib import Parallel, delayed

        parallel = Parallel()

        def run_verbose(*args, verbose=mne_logger.level, **kwargs):
            with use_log_level(verbose=verbose):
                return func(*args, **kwargs)

        my_func = delayed(run_verbose)

    return parallel, my_func
