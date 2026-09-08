"""Parallelization."""

import contextlib
import functools
import time
import warnings
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, Literal

import joblib
from mne.utils import _pl, use_log_level
from mne.utils import logger as mne_logger

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
    return int(n_jobs)


dask_client = None


def setup_dask_client(*, exec_params: SimpleNamespace) -> None:
    global dask_client

    import dask
    from dask.distributed import Client

    if dask_client is not None:
        return

    n_workers = get_n_jobs(exec_params=exec_params)
    if exec_params.dask_cluster is not None:
        dask_client = _external_dask_client(
            exec_params=exec_params, n_workers=n_workers
        )
        return

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
    client = Client(  # type: ignore[no-untyped-call]
        memory_limit=exec_params.dask_worker_memory_limit,
        n_workers=n_workers,
        threads_per_worker=1,
        name="mne-bids-pipeline",
    )

    dashboard_url = client.dashboard_link
    msg = f"Dask client dashboard: [link={dashboard_url}]{dashboard_url}[/link]"
    logger.info(**gen_log_kwargs(message=msg, emoji="🌎"))

    if exec_params.dask_open_dashboard:
        import webbrowser

        webbrowser.open(url=dashboard_url, autoraise=True)

    # Update global variable
    dask_client = client


def _external_dask_client(*, exec_params: SimpleNamespace, n_workers: int) -> Any:
    from dask.distributed import Client

    cluster = exec_params.dask_cluster
    if callable(cluster):
        cluster = cluster()
        # workers not requested (e.g., SLURMCluster before scale/adapt)
        if not getattr(cluster, "worker_spec", None) and hasattr(cluster, "scale"):
            cluster.scale(n_workers)
    what = cluster if isinstance(cluster, str) else type(cluster).__name__
    msg = f"Dask connecting to external cluster {what} …"
    logger.info(**gen_log_kwargs(message=msg, emoji="👾"))
    client = Client(cluster, name="mne-bids-pipeline")  # type: ignore[no-untyped-call]
    # joblib runs everything in the driver if only 1 worker core has been granted by
    # the time a step starts, so wait for the workers we asked for (adaptive clusters
    # start with an empty spec and are left to joblib's probe task)
    n_requested = len(getattr(cluster, "worker_spec", ()) or ())
    if n_requested:
        # log every minute so a long queue wait doesn't look like a hang
        timeout = exec_params.dask_worker_startup_timeout
        deadline = time.monotonic() + timeout

        def _n_up() -> int:
            return len(client.scheduler_info()["workers"])  # type: ignore[no-untyped-call]

        while (step_timeout := min(60.0, deadline - time.monotonic())) > 0:
            msg = f"Waiting for {n_requested} worker{_pl(n_requested)} ({_n_up()} up) …"
            logger.info(**gen_log_kwargs(message=msg, emoji="⏳️"))
            with contextlib.suppress(TimeoutError):
                client.wait_for_workers(n_workers=n_requested, timeout=step_timeout)
            if _n_up() >= n_requested:  # checked last so `else` really means timeout
                break
        else:
            raise TimeoutError(
                f"Only {_n_up()}/{n_requested} Dask workers started within "
                f"dask_worker_startup_timeout={timeout} s; increase it if your "
                "queue is slow, or check the cluster's worker job logs"
            )
    dashboard_url = client.dashboard_link
    msg = f"Dask client dashboard: [link={dashboard_url}]{dashboard_url}[/link]"
    logger.info(**gen_log_kwargs(message=msg, emoji="🌎"))
    return client


def get_parallel_backend_name(
    *,
    exec_params: SimpleNamespace,
) -> Literal["dask", "loky"]:
    backend: Literal["dask", "loky"] = "loky"
    if (
        exec_params.parallel_backend == "loky"
        or get_n_jobs(exec_params=exec_params) == 1
    ):
        pass
    elif exec_params.parallel_backend == "dask":
        # Disable interactive plotting backend
        import matplotlib

        matplotlib.use("Agg")
        backend = "dask"
    else:
        # TODO: Move to value validation step
        raise ValueError(f"Unknown parallel backend: {exec_params.parallel_backend}")

    return backend


@functools.cache
def _filter_loky_warnings() -> None:
    # loky warns from its executor manager thread; escalation (e.g. -W error) kills
    # the thread and wedges the pool at 0% CPU forever, so pin these to "always"
    warnings.filterwarnings(
        "always",
        message="A worker (stopped|was restarted) while some jobs were given",
        category=UserWarning,
    )


def get_parallel_backend(exec_params: SimpleNamespace) -> joblib.parallel_backend:
    _filter_loky_warnings()
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
        if exec_params.dask_cluster is not None:
            # queued jobs can take a while to start (joblib default is 10 s)
            kwargs["wait_for_workers_timeout"] = exec_params.dask_worker_startup_timeout

    return joblib.parallel_backend(backend, **kwargs)


def parallel_func(
    func: Callable[..., Any],
    *,
    exec_params: SimpleNamespace,
    n_iter: int,
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    if get_parallel_backend_name(exec_params=exec_params) == "loky" and (
        n_iter <= 1 or get_n_jobs(exec_params=exec_params) == 1
    ):
        my_func = func
        parallel = list
    else:  # Dask or n_jobs > 1
        from joblib import Parallel, delayed

        parallel = Parallel()

        def run_verbose(*args, verbose=mne_logger.level, **kwargs):  # type: ignore
            with use_log_level(verbose=verbose):
                return func(*args, **kwargs)

        my_func = delayed(run_verbose)

    return parallel, my_func
