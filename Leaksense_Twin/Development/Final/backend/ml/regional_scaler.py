"""
LeakSense Twin — Operating-Point-Conditioned Feature Scaler

Multiple independent FDI papers (see RESEARCH_NOTES.md) found that
intake/exhaust leak isolability improves significantly when residuals are
evaluated against a regime-specific ("operating-point-conditioned")
healthy baseline instead of one global one — e.g. a residual that's
unremarkable at idle may be a real anomaly at rated speed, and vice versa.

RegionalScaler operationalizes this as a drop-in replacement for
sklearn's StandardScaler: raw sensor columns (indices 0-12) are scaled
globally (their absolute magnitude is itself informative — the model
needs to know actual RPM, not just "RPM relative to its own bucket").
The physics-residual / derived-ratio / statistical columns (indices
13-30) are z-scored within RPM buckets, so "is this residual unusual"
is judged relative to that operating point's own healthy distribution.
"""

import numpy as np

RAW_FEATURE_COUNT = 13  # indices 0-12: RPM, MAF, MAP_*, T_*, fuel_qty (see config.RAW_SENSOR_COLS)


class RegionalScaler:
    def __init__(self, rpm_col_idx: int = 0, n_buckets: int = 4):
        self.rpm_col_idx = rpm_col_idx
        self.n_buckets = n_buckets
        self.global_mean_ = None
        self.global_std_ = None
        self.bucket_edges_ = None
        self.regional_mean_ = None
        self.regional_std_ = None

    def _buckets_for(self, rpm: np.ndarray) -> np.ndarray:
        idx = np.digitize(rpm, self.bucket_edges_[1:-1])
        return np.clip(idx, 0, self.n_buckets - 1)

    def fit(self, X: np.ndarray) -> "RegionalScaler":
        X = np.asarray(X, dtype=np.float64)

        raw = X[:, :RAW_FEATURE_COUNT]
        self.global_mean_ = raw.mean(axis=0)
        gstd = raw.std(axis=0)
        self.global_std_ = np.where(gstd < 1e-8, 1.0, gstd)

        rpm = X[:, self.rpm_col_idx]
        self.bucket_edges_ = np.linspace(rpm.min(), rpm.max(), self.n_buckets + 1)
        buckets = self._buckets_for(rpm)

        rest = X[:, RAW_FEATURE_COUNT:]
        n_rest = rest.shape[1]
        self.regional_mean_ = np.zeros((self.n_buckets, n_rest))
        self.regional_std_ = np.ones((self.n_buckets, n_rest))
        for b in range(self.n_buckets):
            mask = buckets == b
            if mask.sum() > 1:
                self.regional_mean_[b] = rest[mask].mean(axis=0)
                s = rest[mask].std(axis=0)
                self.regional_std_[b] = np.where(s < 1e-8, 1.0, s)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        out = np.zeros_like(X)
        out[:, :RAW_FEATURE_COUNT] = (X[:, :RAW_FEATURE_COUNT] - self.global_mean_) / self.global_std_

        rpm = X[:, self.rpm_col_idx]
        buckets = self._buckets_for(rpm)
        rest = X[:, RAW_FEATURE_COUNT:]
        rest_out = np.zeros_like(rest)
        for b in range(self.n_buckets):
            mask = buckets == b
            if mask.any():
                rest_out[mask] = (rest[mask] - self.regional_mean_[b]) / self.regional_std_[b]
        out[:, RAW_FEATURE_COUNT:] = rest_out
        return out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
