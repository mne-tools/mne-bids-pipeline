import numpy as np
from sklearn.linear_model import LogisticRegression
from joblib import parallel_backend

from mne.utils import _validate_type


class LogReg(LogisticRegression):
    """Hack to avoid a warning with n_jobs != 1 when using dask."""

    def fit(self, *args, **kwargs):
        with parallel_backend("loky"):
            return super().fit(*args, **kwargs)


def _handle_csp_args(decoding_csp_times, decoding_csp_freqs, decoding_metric):
    _validate_type(
        decoding_csp_times, (list, tuple, np.ndarray), 'decoding_csp_times')
    if len(decoding_csp_times) < 2:
        raise ValueError(
            'decoding_csp_times should contain at least 2 values.'
        )
    if not np.array_equal(decoding_csp_times, np.sort(decoding_csp_times)):
        ValueError("decoding_csp_times should be sorted.")
    if decoding_metric != "roc_auc":
        raise ValueError(
            f'CSP decoding currently only supports the "roc_auc" '
            f'decoding metric, but received '
            f'decoding_metric="{decoding_metric}"'
        )
    _validate_type(decoding_csp_freqs, dict, 'config.decoding_csp_freqs')
    freq_name_to_bins_map = dict()
    for freq_range_name, edges in decoding_csp_freqs.items():
        _validate_type(freq_range_name, str, 'config.decoding_csp_freqs key')
        _validate_type(
            edges, (list, tuple, np.ndarray),
            f'config.decoding_csp_freqs["{freq_range_name}"]')
        edges = np.array(edges, float)
        if edges.ndim != 1 or len(edges) < 2 or edges[0] < 0 or \
                not np.array_equal(edges, np.sort(edges)):
            raise ValueError(
                f'config.decoding_csp_freqs["{freq_range_name}"] must be '
                '1-dimensional array of at least 2 non-negative values in '
                f'ascending order got: {edges}')

        freq_bins = list(zip(edges[:-1], edges[1:]))
        freq_name_to_bins_map[freq_range_name] = freq_bins
    return freq_name_to_bins_map
