"""Parallelization."""

from typing import Literal, Callable
from types import SimpleNamespace

import joblib

from ._logging import logger, gen_log_kwargs, _is_testing


def get_n_jobs(*, exec_params: SimpleNamespace) -> int:
    n_jobs = exec_params.n_jobs
    if n_jobs < 0:
        n_cores = joblib.cpu_count()
        n_jobs = min(n_cores + n_jobs + 1, n_cores)

    # Shim to allow overriding n_jobs for specific steps
    if _is_testing() and hasattr(exec_params, "_n_jobs"):
        from ._run import _get_step_path, _short_step_path

        step_path = _short_step_path(_get_step_path())
        n_jobs = exec_params._n_jobs.get(step_path, n_jobs)
    return n_jobs


dask_client = None


def setup_dask_client(*, exec_params: SimpleNamespace) -> None:
    global dask_client

    import dask
    from dask.distributed import Client

    if dask_client is not None:
        return

    n_workers = get_n_jobs(exec_params=exec_params)
    logger.info(f"👾 Initializing Dask client with {n_workers} workers …")

    if exec_params.dask_temp_dir is None:
        this_dask_temp_dir = exec_params.deriv_root / ".dask-worker-space"
    else:
        this_dask_temp_dir = exec_params.dask_temp_dir

    logger.info(f"📂 Temporary directory is: {this_dask_temp_dir}")
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
    logger.info(
        f"⏱  The Dask client is ready. Open {dashboard_url} "
        f"to monitor the workers.\n"
    )

    if exec_params.dask_open_dashboard:
        import webbrowser

        webbrowser.open(url=dashboard_url, autoraise=True)

    # Update global variable
    dask_client = client


def get_parallel_backend_name(
    *, exec_params: SimpleNamespace
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

    # Shim to allow changing the backend on a per-step basis for testing
    if _is_testing() and hasattr(exec_params, "_parallel_backend"):
        from ._run import _get_step_path, _short_step_path

        step_path = _short_step_path(_get_step_path())
        old_backend = backend
        backend = exec_params._parallel_backend.get(step_path, backend)
        msg = f"Overriding parallel backend {old_backend}→{backend}"
        logger.info(**gen_log_kwargs(message=msg))
        raise RuntimeError(backend)

    return backend


def get_parallel_backend(exec_params: SimpleNamespace) -> joblib.parallel_backend:
    import joblib

    backend = get_parallel_backend_name(exec_params=exec_params)
    kwargs = {"n_jobs": get_n_jobs(exec_params=exec_params)}

    if backend == "loky":
        kwargs["inner_max_num_threads"] = 1
    else:
        setup_dask_client(exec_params=exec_params)

    return joblib.parallel_backend(backend, **kwargs)


def parallel_func(func: Callable, *, exec_params: SimpleNamespace):
    if get_parallel_backend_name(exec_params=exec_params) == "loky":
        if get_n_jobs(exec_params=exec_params) == 1:
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
