# src/data/outlier_handler.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from scipy import stats
from loguru import logger
from mltools.shared.exceptions import OutlierHandlerError
from typing import List, Optional


class ExpertOutlierHandler(BaseEstimator, TransformerMixin):
    """
    Handler outlier production-grade dengan multiple strategi.
    Sklearn-compatible: fit() pada train, transform() pada semua split.

    Parameters
    ----------
    method : 'iqr' | 'zscore' | 'modified_zscore' | 'isolation_forest' | 'lof'
    treatment : 'cap' | 'remove' | 'flag' | 'cap_and_flag'
    threshold : Multiplier untuk IQR (1.5=mild, 3.0=extreme)
                atau z-score threshold (2.5–3.5)
    contamination : Estimasi fraksi outlier (untuk isolation_forest / lof)
    columns : List kolom yang akan dicek. None = semua numerik
    """

    def __init__(
        self,
        method        : str = "iqr",
        treatment     : str = "cap",
        threshold     : float = 1.5,
        contamination : float = 0.05,
        columns       : Optional[List[str]] = None,
    ):
        self.method        = method
        self.treatment     = treatment
        self.threshold     = threshold
        self.contamination = contamination
        self.columns       = columns

    def fit(self, X: pd.DataFrame, y=None):
        """Belajar batas outlier dari training data.

        Example
        -------
        >>> handler = ExpertOutlierHandler(method="iqr", treatment="cap", threshold=1.5)
        >>> handler.fit(X_train)
        """
        # Tentukan kolom yang akan diproses
        if self.columns:
            self.num_cols_ = [c for c in self.columns if c in X.columns]
        else:
            self.num_cols_ = X.select_dtypes(include=np.number).columns.tolist()

        self.bounds_        = {}   # Simpan lower/upper bounds per kolom
        self.outlier_models_= {}

        logger.info(
            f"Fitting outlier handler: method={self.method}, "
            f"treatment={self.treatment}"
        )
        if self.method == "iqr":
            self._fit_iqr(X)
        elif self.method == "zscore":
            self._fit_zscore(X)
        elif self.method == "modified_zscore":
            self._fit_modified_zscore(X)
        elif self.method == "isolation_forest":
            self._fit_isolation_forest(X)
        elif self.method == "lof":
            self._fit_lof(X)

        logger.success(f"OutlierHandler fitted pada {len(self.num_cols_)} kolom")
        return self

    def _fit_iqr(self, X: pd.DataFrame):
        """IQR Method — Robust, cocok untuk distribusi skewed."""
        for col in self.num_cols_:
            Q1  = X[col].quantile(0.25)
            Q3  = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - self.threshold * IQR
            upper = Q3 + self.threshold * IQR
            self.bounds_[col] = {"lower": lower, "upper": upper}

    def _fit_zscore(self, X: pd.DataFrame):
        """Z-Score Method — Cocok untuk distribusi normal."""
        for col in self.num_cols_:
            mean = X[col].mean()
            std  = X[col].std()
            self.bounds_[col] = {
                "lower": mean - self.threshold * std,
                "upper": mean + self.threshold * std,
                "mean" : mean,
                "std"  : std,
            }

    def _fit_modified_zscore(self, X: pd.DataFrame):
        """
        Modified Z-Score (MAD-based) — Lebih robust dari Z-Score biasa.
        Cocok untuk distribusi dengan outlier ekstrim.
        """
        for col in self.num_cols_:
            median = X[col].median()
            mad    = np.abs(X[col] - median).median()
            # Konversi MAD ke skala std: 0.6745 adalah faktor konversi
            lower = median - (self.threshold * mad / 0.6745)
            upper = median + (self.threshold * mad / 0.6745)
            self.bounds_[col] = {
                "lower" : lower,
                "upper" : upper,
                "median": median,
                "mad"   : mad,
            }

    def _fit_isolation_forest(self, X: pd.DataFrame):
        """
        Isolation Forest — Multivariate, tidak perlu asumsi distribusi.
        Ideal untuk anomaly detection.
        """
        X_num = X[self.num_cols_].copy()
        # Handle missing sebelum fitting (quick fill dengan median)
        for col in self.num_cols_:
            X_num[col] = X_num[col].fillna(X_num[col].median())
        self.iso_forest_ = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=200,
            n_jobs=-1,
        )
        self.iso_forest_.fit(X_num)
        # Untuk treatment=cap: gunakan IQR bounds sebagai fallback
        self._fit_iqr(X)
        logger.info("Isolation Forest fitted (multivariate detection)")

    def _fit_lof(self, X: pd.DataFrame):
        """
        Local Outlier Factor — Deteksi outlier berbasis density.
        Bagus untuk data dengan cluster tidak seragam.
        """
        X_num = X[self.num_cols_].copy()
        for col in self.num_cols_:
            X_num[col] = X_num[col].fillna(X_num[col].median())
        # Normalize dulu untuk LOF
        scaler   = RobustScaler()
        X_scaled = scaler.fit_transform(X_num)
        self.lof_scaler_ = scaler
        self.lof_ = LocalOutlierFactor(
            n_neighbors=20,
            contamination=self.contamination,
            novelty=True,   # novelty=True agar bisa predict pada test
            n_jobs=-1,
        )
        self.lof_.fit(X_scaled)
        # Fallback bounds untuk capping
        self._fit_iqr(X)
        logger.info("LOF fitted (density-based detection)")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply outlier treatment.

        Example
        -------
        >>> X_train_clean = handler.transform(X_train)
        >>> X_val_clean   = handler.transform(X_val)    # gunakan bounds dari train
        >>> X_test_clean  = handler.transform(X_test)
        """
        X = X.copy()
        total_outliers = 0

        if self.method in ["isolation_forest", "lof"]:
            # Multivariate: dapatkan mask outlier dulu
            X_num = X[self.num_cols_].copy()
            for col in self.num_cols_:
                X_num[col] = X_num[col].fillna(X_num[col].median())

            if self.method == "isolation_forest":
                predictions = self.iso_forest_.predict(X_num)
            else:   # lof
                X_scaled    = self.lof_scaler_.transform(X_num)
                predictions = self.lof_.predict(X_scaled)

            outlier_mask    = predictions == -1
            n_outliers      = outlier_mask.sum()
            total_outliers += n_outliers

            if "flag" in self.treatment:
                X["is_outlier_multivariate"] = outlier_mask.astype(int)
            if self.treatment == "remove":
                X = X[~outlier_mask]
            elif "cap" in self.treatment:
                # Cap menggunakan IQR bounds (univariate per kolom)
                for col in self.num_cols_:
                    if col in self.bounds_:
                        bounds = self.bounds_[col]
                        X[col] = X[col].clip(bounds["lower"], bounds["upper"])

        else:
            # Univariate method (iqr / zscore / modified_zscore)
            rows_to_drop = set()
            for col in self.num_cols_:
                if col not in X.columns or col not in self.bounds_:
                    continue
                bounds = self.bounds_[col]
                lower, upper = bounds["lower"], bounds["upper"]
                col_outliers = (X[col] < lower) | (X[col] > upper)
                n_outliers   = col_outliers.sum()
                if n_outliers > 0:
                    total_outliers += n_outliers
                    if "flag" in self.treatment:
                        X[f"{col}_is_outlier"] = col_outliers.astype(int)
                    if "cap" in self.treatment:
                        X[col] = X[col].clip(lower, upper)
                    elif self.treatment == "remove":
                        rows_to_drop.update(X[col_outliers].index.tolist())
            if self.treatment == "remove" and rows_to_drop:
                X = X.drop(index=list(rows_to_drop))
                logger.info(f"Removed {len(rows_to_drop)} outlier rows")

        logger.info(f"Outlier treatment selesai. Total outliers: {total_outliers:,}")
        return X

    def get_outlier_report(self, X: pd.DataFrame) -> pd.DataFrame:
        """Laporan lengkap outlier per kolom.

        Example
        -------
        >>> handler.fit(X_train)
        >>> report = handler.get_outlier_report(X_train)
        >>> print(report.head())
        """
        report = []
        for col in self.num_cols_:
            if col not in self.bounds_:
                continue
            bounds = self.bounds_[col]
            mask   = (X[col] < bounds["lower"]) | (X[col] > bounds["upper"])
            n      = mask.sum()
            pct    = n / len(X) * 100
            report.append({
                "column"      : col,
                "n_outliers"  : n,
                "pct_outliers": f"{pct:.2f}%",
                "lower_bound" : round(bounds["lower"], 4),
                "upper_bound" : round(bounds["upper"], 4),
                "action"      : self.treatment,
            })
        return pd.DataFrame(report).sort_values("n_outliers", ascending=False)


# ── Visualisasi Outlier ───────────────────────────────────────────────────────

def plot_outlier_before_after(
    df_before : pd.DataFrame,
    df_after  : pd.DataFrame,
    columns   : List[str],
    max_cols  : int = 6,
):
    """Visualisasi distribusi sebelum vs sesudah outlier treatment.

    Example
    -------
    >>> plot_outlier_before_after(X_train, X_train_clean,
    ...                          columns=report["column"].head(6).tolist())
    """
    import matplotlib.pyplot as plt
    cols = columns[:max_cols]
    fig, axes = plt.subplots(len(cols), 2, figsize=(14, len(cols) * 3))
    if len(cols) == 1:
        axes = [axes]
    for i, col in enumerate(cols):
        # Before
        axes[i][0].hist(
            df_before[col].dropna(), bins=50,
            color="salmon", edgecolor="black", alpha=0.7,
        )
        axes[i][0].set_title(
            f"{col} — BEFORE\n(skew: {df_before[col].skew():.2f})"
        )
        # After
        axes[i][1].hist(
            df_after[col].dropna(), bins=50,
            color="steelblue", edgecolor="black", alpha=0.7,
        )
        axes[i][1].set_title(
            f"{col} — AFTER\n(skew: {df_after[col].skew():.2f})"
        )
    plt.suptitle("Outlier Treatment: Before vs After", fontsize=14)
    plt.tight_layout()
    plt.savefig("reports/outlier_before_after.png", dpi=150, bbox_inches="tight")
    plt.show()


# ── Contoh Penggunaan ─────────────────────────────────────────────────────────
"""
# Deteksi + Visualisasi dulu
handler = ExpertOutlierHandler(method="iqr", treatment="cap", threshold=1.5)
handler.fit(X_train)
report = handler.get_outlier_report(X_train)
print(report)

# Apply treatment
X_train_clean = handler.transform(X_train)
X_val_clean   = handler.transform(X_val)
X_test_clean  = handler.transform(X_test)

# Visualisasi
plot_outlier_before_after(
    X_train, X_train_clean, columns=report["column"].head(6).tolist()
)

# Untuk fraud/anomaly detection: gunakan isolation_forest + flag
handler_fraud = ExpertOutlierHandler(
    method        = "isolation_forest",
    treatment     = "cap_and_flag",
    contamination = 0.02,   # Estimasi 2% fraud
)
"""