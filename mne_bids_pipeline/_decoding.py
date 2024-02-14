from typing import Optional

import mne
import numpy as np
from joblib import parallel_backend
from mne.utils import _validate_type
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from ._logging import gen_log_kwargs, logger


class LogReg(LogisticRegression):
    """Hack to avoid a warning with n_jobs != 1 when using dask."""

    def fit(self, *args, **kwargs):
        with parallel_backend("loky"):
            return super().fit(*args, **kwargs)


def _handle_csp_args(
    decoding_csp_times,
    decoding_csp_freqs,
    decoding_metric,
    *,
    epochs_tmin,
    epochs_tmax,
    time_frequency_freq_min,
    time_frequency_freq_max,
):
    _validate_type(
        decoding_csp_times, (None, list, tuple, np.ndarray), "decoding_csp_times"
    )
    if decoding_csp_times is None:
        decoding_csp_times = np.linspace(max(0, epochs_tmin), epochs_tmax, num=6)
    if len(decoding_csp_times) < 2:
        raise ValueError("decoding_csp_times should contain at least 2 values.")
    if not np.array_equal(decoding_csp_times, np.sort(decoding_csp_times)):
        ValueError("decoding_csp_times should be sorted.")
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
    return freq_name_to_bins_map


def _decoding_preproc_steps(
    subject: str,
    session: Optional[str],
    epochs: mne.Epochs,
    pca: bool = True,
) -> list[BaseEstimator]:
    scaler = mne.decoding.Scaler(epochs.info)
    steps = [scaler]
    if pca:
        ranks = mne.compute_rank(inst=epochs, rank="info")
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
