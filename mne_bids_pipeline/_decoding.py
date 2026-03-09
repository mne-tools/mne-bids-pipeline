from types import SimpleNamespace
from typing import Any

import mne
import numpy as np
from joblib import parallel_backend
from mne.utils import _validate_type, check_version
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from ._config_utils import _get_rank
from ._logging import gen_log_kwargs, logger
from .typing import FloatArrayT


class LogReg(LogisticRegression):  # type: ignore[misc]
    """Logistic Regression with fixed parameters suitable for our internal decoding."""

    def __init__(self, *, random_state: int | None) -> None:
        kwargs = dict(
            random_state=random_state,
            solver="liblinear",  # much faster than the default
        )
        # TODO: Once we require sklearn 1.8 we should drop n_jobs
        if not check_version("sklearn", "1.8"):
            kwargs["n_jobs"] = 1
        super().__init__(**kwargs)

    def fit(self, *args, **kwargs):  # type: ignore
        # Hack to avoid a warning with n_jobs != 1 when using dask
        with parallel_backend("loky"):
            return super().fit(*args, **kwargs)


def _handle_csp_args(
    decoding_csp_times: list[float] | tuple[float, ...] | FloatArrayT | None,
    decoding_csp_freqs: dict[str, Any] | None,
    decoding_metric: str,
    *,
    epochs_tmin: float,
    epochs_tmax: float,
    time_frequency_freq_min: float,
    time_frequency_freq_max: float,
) -> tuple[dict[str, list[tuple[float, float]]], FloatArrayT]:
    _validate_type(
        decoding_csp_times, (None, list, tuple, np.ndarray), "decoding_csp_times"
    )
    if decoding_csp_times is None:
        decoding_csp_times = np.linspace(
            max(0, epochs_tmin), epochs_tmax, num=6, dtype=float
        )
    else:
        decoding_csp_times = np.array(decoding_csp_times, float)
    assert isinstance(decoding_csp_times, np.ndarray)
    if decoding_csp_times.ndim != 1 or len(decoding_csp_times) == 1:
        raise ValueError(
            "decoding_csp_times should be 1 dimensional and contain at least 2 values "
            "to define time intervals, or be empty to disable time-frequency mode, got "
            f"shape {decoding_csp_times.shape}"
        )
    if not np.array_equal(decoding_csp_times, np.sort(decoding_csp_times)):
        ValueError("decoding_csp_times should be sorted.")
    time_bins = np.c_[decoding_csp_times[:-1], decoding_csp_times[1:]]
    assert time_bins.ndim == 2 and time_bins.shape[1] == 2, time_bins.shape

    if decoding_metric != "roc_auc":
        raise ValueError(
            f'CSP decoding currently only supports the "roc_auc" '
            f"decoding metric, but received "
            f'decoding_metric="{decoding_metric}"'
        )
    _validate_type(decoding_csp_freqs, (None, dict), "config.decoding_csp_freqs")
    if decoding_csp_freqs is None:
        decoding_csp_freqs = {
            "custom": (
                time_frequency_freq_min,
                (time_frequency_freq_max + time_frequency_freq_min) / 2,  # noqa: E501
                time_frequency_freq_max,
            ),
        }
    freq_name_to_bins_map = dict()
    for freq_range_name, edges in decoding_csp_freqs.items():
        _validate_type(freq_range_name, str, "config.decoding_csp_freqs key")
        _validate_type(
            edges,
            (list, tuple, np.ndarray),
            f'config.decoding_csp_freqs["{freq_range_name}"]',
        )
        edges = np.array(edges, float)
        if (
            edges.ndim != 1
            or len(edges) < 2
            or edges[0] < 0
            or not np.array_equal(edges, np.sort(edges))
        ):
            raise ValueError(
                f'config.decoding_csp_freqs["{freq_range_name}"] must be '
                "1-dimensional array of at least 2 non-negative values in "
                f"ascending order got: {edges}"
            )

        freq_bins = list(zip(edges[:-1], edges[1:]))
        freq_name_to_bins_map[freq_range_name] = freq_bins
    return freq_name_to_bins_map, time_bins


def _decoding_preproc_steps(
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
    epochs: mne.BaseEpochs,
    pca: bool = True,
) -> list[BaseEstimator]:
    scaler = mne.decoding.Scaler(epochs.info)
    steps = [scaler]
    if pca:
        ranks = _get_rank(
            cfg=cfg, subject=subject, session=session, inst=epochs, log=False
        )
        rank = sum(ranks.values())
        msg = f"Reducing data dimension via PCA; new rank: {rank} (from {ranks})."
        logger.info(**gen_log_kwargs(message=msg))
        steps.append(
            mne.decoding.UnsupervisedSpatialFilter(
                PCA(rank, whiten=True),
                average=False,
            )
        )
    return steps
