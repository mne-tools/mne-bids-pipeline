from sklearn.linear_model import LogisticRegression
from joblib import parallel_backend


class LogReg(LogisticRegression):
    """Hack to avoid a warning with n_jobs != 1 when using dask
    """
    def fit(self, *args, **kwargs):
        with parallel_backend("loky"):
            return super().fit(*args, **kwargs)
