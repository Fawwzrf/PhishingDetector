# src/features/scaler.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    MaxAbsScaler, PowerTransformer, QuantileTransformer,
    FunctionTransformer,
)
from scipy import stats
from loguru import logger
from mltools.shared.exceptions import ScalingError
from typing import List, Optional


class ExpertScalerTransformer(BaseEstimator, TransformerMixin):
    """
    Scaler + Transformer production-grade.

    Fitur:
    - Auto-detect kolom yang perlu transformasi (berdasarkan skewness)
    - Apply transformasi sebelum scaling
    - Hanya proses kolom numerik
    - Sklearn-compatible pipeline

    Parameters
    ----------
    scaler : 'standard' | 'minmax' | 'robust' | 'maxabs' | 'none'
    auto_transform : Otomatis transformasi kolom skewed
    transform_method : 'yeojohnson' | 'boxcox' | 'log1p' | 'sqrt' | 'quantile'
    skew_threshold : Threshold absolute skewness untuk auto-transform (default: 1.0)
    columns : Kolom yang akan di-scale. None = semua numerik
    exclude_cols : Kolom yang dikecualikan dari scaling
    """

    def __init__(
        self,
        scaler           : str = "robust",
        auto_transform   : bool = True,
        transform_method : str = "yeojohnson",
        skew_threshold   : float = 1.0,
        columns          : Optional[List[str]] = None,
        exclude_cols     : Optional[List[str]] = None,
    ):
        self.scaler           = scaler
        self.auto_transform   = auto_transform
        self.transform_method = transform_method
        self.skew_threshold   = skew_threshold
        self.columns          = columns
        self.exclude_cols     = exclude_cols or []

    def fit(self, X: pd.DataFrame, y=None):
        """Fit scaler dan transformer pada training data.

        Example
        -------
        >>> scaler = ExpertScalerTransformer(scaler="robust", auto_transform=True,
        ...                                  exclude_cols=["year", "month"])
        >>> scaler.fit(X_train)
        """
        # Tentukan kolom yang akan diproses
        if self.columns:
            self.num_cols_ = [c for c in self.columns if c in X.columns]
        else:
            self.num_cols_ = X.select_dtypes(include=np.number).columns.tolist()
        # Exclude kolom tertentu (misal: binary features, target)
        self.num_cols_ = [c for c in self.num_cols_ if c not in self.exclude_cols]
        logger.info(f"Fitting scaler ({self.scaler}) pada {len(self.num_cols_)} kolom")

        # ── 1. Auto-detect kolom yang perlu transformasi ──────────────────────
        self.transform_cols_ = []
        if self.auto_transform:
            for col in self.num_cols_:
                # Abaikan kolom yang sudah binary (0/1)
                if set(X[col].dropna().unique()).issubset({0, 1}):
                    continue
                skew = abs(X[col].skew())
                if skew > self.skew_threshold:
                    self.transform_cols_.append(col)
            logger.info(
                f"Kolom yang akan ditransformasi ({self.transform_method}): "
                f"{self.transform_cols_}"
            )

        # ── 2. Fit Transformer ────────────────────────────────────────────────
        if self.transform_cols_:
            if self.transform_method == "yeojohnson":
                self.transformer_ = PowerTransformer(
                    method     = "yeo-johnson",
                    standardize= False,   # Scaling dilakukan terpisah
                )
                self.transformer_.fit(X[self.transform_cols_])

            elif self.transform_method == "boxcox":
                # Box-Cox hanya untuk nilai positif
                self.transformer_ = PowerTransformer(
                    method     = "box-cox",
                    standardize= False,
                )
                # Pastikan semua nilai positif
                for col in self.transform_cols_:
                    if X[col].min() <= 0:
                        logger.warning(
                            f"Kolom '{col}' punya nilai ≤ 0, "
                            f"beralih ke yeo-johnson"
                        )
                        self.transform_method = "yeojohnson"
                        self.transformer_ = PowerTransformer(
                            method="yeo-johnson", standardize=False
                        )
                        break
                self.transformer_.fit(X[self.transform_cols_])

            elif self.transform_method == "log1p":
                # log(1 + x) — untuk data count/positif
                self.transformer_ = FunctionTransformer(
                    func         = np.log1p,
                    inverse_func = np.expm1,
                    validate     = True,
                )
                self.transformer_.fit(X[self.transform_cols_])

            elif self.transform_method == "sqrt":
                self.transformer_ = FunctionTransformer(
                    func         = np.sqrt,
                    inverse_func = np.square,
                    validate     = True,
                )
                self.transformer_.fit(X[self.transform_cols_])

            elif self.transform_method == "quantile":
                self.transformer_ = QuantileTransformer(
                    output_distribution = "normal",
                    n_quantiles         = min(1000, len(X)),
                    random_state        = 42,
                )
                self.transformer_.fit(X[self.transform_cols_])

        # ── 3. Fit Scaler ─────────────────────────────────────────────────────
        if self.scaler != "none":
            scaler_map = {
                "standard": StandardScaler(),
                "minmax"  : MinMaxScaler(),
                "robust"  : RobustScaler(quantile_range=(5, 95)),
                "maxabs"  : MaxAbsScaler(),
            }
            self.scaler_ = scaler_map[self.scaler]
            self.scaler_.fit(X[self.num_cols_])

        logger.success("ScalerTransformer fitted!")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply transformasi dan scaling.

        Example
        -------
        >>> X_train_scaled = scaler.transform(X_train)
        >>> X_val_scaled   = scaler.transform(X_val)    # gunakan parameter dari train
        >>> X_test_scaled  = scaler.transform(X_test)
        """
        X = X.copy()

        # ── 1. Apply Transformation ───────────────────────────────────────────
        if self.transform_cols_ and hasattr(self, "transformer_"):
            # Filter kolom yang ada di X
            cols_present = [c for c in self.transform_cols_ if c in X.columns]
            if cols_present:
                X[cols_present] = self.transformer_.transform(X[cols_present])

        # ── 2. Apply Scaling ──────────────────────────────────────────────────
        if self.scaler != "none" and hasattr(self, "scaler_"):
            cols_present = [c for c in self.num_cols_ if c in X.columns]
            if cols_present:
                X[cols_present] = self.scaler_.transform(X[cols_present])

        return X

    def get_skewness_report(self, X: pd.DataFrame) -> pd.DataFrame:
        """Laporan skewness per kolom sebelum transformasi.

        Example
        -------
        >>> scaler.fit(X_train)
        >>> report = scaler.get_skewness_report(X_train)
        >>> print(report)   # kolom diurutkan dari skewness tertinggi
        """
        report = []
        for col in self.num_cols_:
            if col not in X.columns:
                continue
            skew           = X[col].skew()
            will_transform = col in self.transform_cols_
            report.append({
                "column"        : col,
                "skewness"      : round(skew, 4),
                "abs_skewness"  : round(abs(skew), 4),
                "will_transform": will_transform,
                "severity"      : (
                    "HIGH ⚠" if abs(skew) > 2
                    else "MED ~" if abs(skew) > 1
                    else "OK ✅"
                ),
            })
        return pd.DataFrame(report).sort_values("abs_skewness", ascending=False)


# ── Visualisasi Before/After Transformation ───────────────────────────────────

def plot_transformation_effect(
    df_before : pd.DataFrame,
    df_after  : pd.DataFrame,
    columns   : List[str],
    max_cols  : int = 6,
):
    """Bandingkan distribusi sebelum dan sesudah transformasi.

    Example
    -------
    >>> scaler.fit(X_train)
    >>> X_train_scaled = scaler.transform(X_train)
    >>> plot_transformation_effect(X_train, X_train_scaled,
    ...                            columns=scaler.transform_cols_[:6])
    """
    import matplotlib.pyplot as plt
    from scipy import stats as scipy_stats

    cols = columns[:max_cols]
    fig, axes = plt.subplots(len(cols), 3, figsize=(18, len(cols) * 3))
    if len(cols) == 1:
        axes = [axes]

    for i, col in enumerate(cols):
        before = df_before[col].dropna()
        after  = df_after[col].dropna()

        # Before histogram
        axes[i][0].hist(
            before, bins=50, color="salmon", edgecolor="black", alpha=0.7
        )
        axes[i][0].set_title(f"{col}\nBEFORE (skew: {before.skew():.2f})")

        # After histogram
        axes[i][1].hist(
            after, bins=50, color="steelblue", edgecolor="black", alpha=0.7
        )
        axes[i][1].set_title(f"AFTER (skew: {after.skew():.2f})")

        # Q-Q Plot (normalitas check)
        scipy_stats.probplot(after, dist="norm", plot=axes[i][2])
        axes[i][2].set_title("Q-Q Plot (After)")

    plt.suptitle("Feature Transformation: Before vs After", fontsize=14)
    plt.tight_layout()
    plt.savefig("reports/transformation_effect.png", dpi=150, bbox_inches="tight")
    plt.show()


# ── Contoh Penggunaan ─────────────────────────────────────────────────────────
"""
scaler = ExpertScalerTransformer(
    scaler           = "robust",       # robust scaler
    auto_transform   = True,           # auto-detect skewed features
    transform_method = "yeojohnson",   # paling aman untuk umum
    skew_threshold   = 1.0,
    exclude_cols     = ["binary_flag", "year", "month"],  # jangan scale ini
)

# Cek skewness dulu
report = scaler.get_skewness_report(X_train)
print(report)

# Fit pada train, transform semua
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# Visualisasi efek transformasi
plot_transformation_effect(X_train, X_train_scaled,
                           columns=scaler.transform_cols_[:6])
"""