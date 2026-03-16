"""
Feature engineering — modular, production-grade.
Aktifkan/nonaktifkan setiap komponen sesuai kebutuhan.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer, PolynomialFeatures


class ExpertFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering expert-grade.
    Modular: aktifkan setiap komponen sesuai kebutuhan.
    """

    def __init__(
        self,
        add_polynomial: bool = False,
        poly_degree: int = 2,
        poly_cols: Optional[List[str]] = None,
        add_ratios: bool = False,
        ratio_pairs: Optional[List[Tuple[str, str]]] = None,
        add_bins: bool = False,
        bin_cols: Optional[List[str]] = None,
        n_bins: int = 5,
        bin_strategy: str = "quantile",
        add_interaction: bool = False,
        interaction_pairs: Optional[List[Tuple[str, str]]] = None,
        add_group_agg: bool = False,
        group_agg_config: Optional[List[Dict]] = None,
        add_text_features: bool = False,
        text_cols: Optional[List[str]] = None,
    ):
        self.add_polynomial = add_polynomial
        self.poly_degree = poly_degree
        self.poly_cols = poly_cols
        self.add_ratios = add_ratios
        self.ratio_pairs = ratio_pairs
        self.add_bins = add_bins
        self.bin_cols = bin_cols
        self.n_bins = n_bins
        self.bin_strategy = bin_strategy
        self.add_interaction = add_interaction
        self.interaction_pairs = interaction_pairs
        self.add_group_agg = add_group_agg
        self.group_agg_config = group_agg_config or []
        self.add_text_features = add_text_features
        self.text_cols = text_cols

    def fit(self, X: pd.DataFrame, y=None) -> "ExpertFeatureEngineer":
        """Fit transformers yang diperlukan (polynomial, binning, group agg maps).

        Example
        -------
        >>> fe = ExpertFeatureEngineer(
        ...     add_polynomial=True, poly_cols=["age", "income"],
        ...     add_bins=True, n_bins=5,
        ...     add_ratios=True, ratio_pairs=[("income", "debt")],
        ... )
        >>> fe.fit(X_train)
        """
        if self.add_polynomial:
            cols = self.poly_cols or X.select_dtypes(include=np.number).columns.tolist()[:10]
            self.poly_cols_ = cols
            self.poly_ = PolynomialFeatures(
                degree=self.poly_degree,
                include_bias=False,
                interaction_only=True,
            )
            self.poly_.fit(X[self.poly_cols_])
            self.poly_feature_names_ = self.poly_.get_feature_names_out(self.poly_cols_)

        if self.add_bins:
            cols = self.bin_cols or X.select_dtypes(include=np.number).columns.tolist()
            self.bin_cols_ = cols
            self.binner_ = KBinsDiscretizer(
                n_bins=self.n_bins,
                encode="ordinal",
                strategy=self.bin_strategy,
                subsample=None,
            )
            self.binner_.fit(X[self.bin_cols_])

        if self.add_group_agg and self.group_agg_config:
            self.group_agg_maps_ = {}
            for config in self.group_agg_config:
                group_col = config["group_col"]
                agg_col = config["agg_col"]
                funcs = config.get("funcs", ["mean", "std"])
                for func in funcs:
                    feat_name = f"{agg_col}_by_{group_col}_{func}"
                    agg_map = X.groupby(group_col)[agg_col].agg(func)
                    self.group_agg_maps_[feat_name] = {
                        "group_col": group_col,
                        "agg_map": agg_map,
                    }
                    logger.debug(f"Group agg feature: {feat_name}")

        logger.info("FeatureEngineer fitted")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering ke DataFrame.

        Example
        -------
        >>> X_train_fe = fe.transform(X_train)
        >>> X_val_fe   = fe.transform(X_val)    # gunakan parameter dari train
        >>> X_test_fe  = fe.transform(X_test)
        """
        X = X.copy()

        if self.add_polynomial and hasattr(self, "poly_"):
            cols_present = [c for c in self.poly_cols_ if c in X.columns]
            if cols_present:
                poly_array = self.poly_.transform(X[cols_present])
                poly_df = pd.DataFrame(
                    poly_array,
                    columns=self.poly_.get_feature_names_out(cols_present),
                    index=X.index,
                )
                new_cols = [c for c in poly_df.columns if c not in X.columns]
                X = pd.concat([X, poly_df[new_cols]], axis=1)
                logger.debug(f"Added {len(new_cols)} polynomial features")

        if self.add_ratios and self.ratio_pairs:
            for col_a, col_b in self.ratio_pairs:
                if col_a in X.columns and col_b in X.columns:
                    X[f"{col_a}_div_{col_b}"] = X[col_a] / (X[col_b] + 1e-8)
                    X[f"{col_a}_pct_of_sum"] = X[col_a] / (X[col_a] + X[col_b] + 1e-8)
            logger.debug("Added ratio features")

        if self.add_bins and hasattr(self, "binner_"):
            cols_present = [c for c in self.bin_cols_ if c in X.columns]
            if cols_present:
                binned = self.binner_.transform(X[cols_present])
                for i, col in enumerate(cols_present):
                    X[f"{col}_bin"] = binned[:, i].astype(int)
                logger.debug(f"Added {len(cols_present)} bin features")

        if self.add_interaction and self.interaction_pairs:
            for col_a, col_b in self.interaction_pairs:
                if col_a in X.columns and col_b in X.columns:
                    if X[col_a].dtype in [np.float64, np.int64] and X[col_b].dtype in [np.float64, np.int64]:
                        X[f"{col_a}_x_{col_b}"] = X[col_a] * X[col_b]
                    else:
                        X[f"{col_a}__{col_b}"] = (
                            X[col_a].astype(str) + "_" + X[col_b].astype(str)
                        )
            logger.debug("Added interaction features")

        if self.add_group_agg and hasattr(self, "group_agg_maps_"):
            for feat_name, config in self.group_agg_maps_.items():
                group_col = config["group_col"]
                agg_map = config["agg_map"]
                if group_col in X.columns:
                    X[feat_name] = X[group_col].map(agg_map)

        if self.add_text_features and self.text_cols:
            for col in self.text_cols:
                if col in X.columns:
                    s = X[col].astype(str)
                    X[f"{col}_length"] = s.str.len()
                    X[f"{col}_word_count"] = s.str.split().str.len().fillna(0).astype(int)
                    X[f"{col}_n_digits"] = s.str.count(r"\d").fillna(0).astype(int)
                    X[f"{col}_n_special"] = s.str.count(r"[^a-zA-Z0-9\s]").fillna(0).astype(int)
                    X[f"{col}_n_upper"] = s.str.count(r"[A-Z]").fillna(0).astype(int)
            logger.debug("Added text-based features")

        logger.info(f"Feature engineering selesai. Shape: {X.shape}")
        return X


class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Ekstrak fitur dari kolom datetime — production-grade.
    Termasuk cyclical encoding (sin/cos) untuk fitur circular (jam, bulan, weekday).
    """

    def __init__(
        self,
        date_cols: List[str],
        extract_components: bool = True,
        add_cyclical: bool = True,
        add_time_since: Optional[Dict[str, str]] = None,
        add_is_weekend: bool = True,
        drop_original: bool = True,
    ):
        self.date_cols = date_cols
        self.extract_components = extract_components
        self.add_cyclical = add_cyclical
        self.add_time_since = add_time_since or {}
        self.add_is_weekend = add_is_weekend
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y=None) -> "DatetimeFeatureExtractor":
        """Stateless — tidak perlu belajar dari data (sklearn Pipeline compat).

        Example
        -------
        >>> dte = DatetimeFeatureExtractor(date_cols=["created_at"])
        >>> dte.fit(X_train)   # no-op, tapi wajib dipanggil dalam Pipeline
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ekstrak fitur datetime dari semua date_cols yang dikonfigurasi.

        Example
        -------
        >>> dte = DatetimeFeatureExtractor(
        ...     date_cols=["created_at"],
        ...     add_cyclical=True,
        ...     add_time_since={"created_at": "2020-01-01"},
        ... )
        >>> X_fe = dte.fit(X_train).transform(X_train)
        """
        X = X.copy()

        for col in self.date_cols:
            if col not in X.columns:
                continue

            dt = pd.to_datetime(X[col], errors="coerce")
            weekday = dt.dt.weekday

            if self.extract_components:
                X[f"{col}_year"] = dt.dt.year
                X[f"{col}_month"] = dt.dt.month
                X[f"{col}_day"] = dt.dt.day
                X[f"{col}_hour"] = dt.dt.hour
                X[f"{col}_minute"] = dt.dt.minute
                X[f"{col}_weekday"] = weekday
                X[f"{col}_quarter"] = dt.dt.quarter
                X[f"{col}_dayofyear"] = dt.dt.dayofyear
                iso = dt.dt.isocalendar()
                X[f"{col}_weekofyear"] = iso.week.astype(int)

            if self.add_is_weekend:
                X[f"{col}_is_weekend"] = (weekday >= 5).astype(int)
                X[f"{col}_is_monthstart"] = dt.dt.is_month_start.astype(int)
                X[f"{col}_is_monthend"] = dt.dt.is_month_end.astype(int)

            if self.add_cyclical:
                X[f"{col}_hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
                X[f"{col}_hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
                X[f"{col}_month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
                X[f"{col}_month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
                X[f"{col}_weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
                X[f"{col}_weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
                doy = dt.dt.dayofyear
                X[f"{col}_doy_sin"] = np.sin(2 * np.pi * doy / 365)
                X[f"{col}_doy_cos"] = np.cos(2 * np.pi * doy / 365)

            if col in self.add_time_since:
                ref_date = pd.to_datetime(self.add_time_since[col])
                X[f"{col}_days_since_ref"] = (dt - ref_date).dt.days

            if self.drop_original:
                X = X.drop(columns=[col])

        logger.info(f"Datetime features extracted. Shape: {X.shape}")
        return X


def create_rfm_features(
    df: pd.DataFrame,
    customer_col: str,
    date_col: str,
    amount_col: str,
    reference_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Buat fitur RFM (Recency, Frequency, Monetary).
    Returns DataFrame dengan kolom: customer_id, recency, frequency, monetary, dll.

    Example
    -------
    >>> rfm = create_rfm_features(
    ...     df,
    ...     customer_col="customer_id",
    ...     date_col="transaction_date",
    ...     amount_col="amount",
    ...     reference_date="2024-01-01",
    ... )
    >>> print(rfm.head())
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    ref = pd.to_datetime(reference_date) if reference_date else df[date_col].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby(customer_col)
        .agg(
            recency=(date_col, lambda x: (ref - x.max()).days),
            frequency=(date_col, "count"),
            monetary=(amount_col, "sum"),
        )
        .reset_index()
    )

    rfm["avg_order_value"] = rfm["monetary"] / rfm["frequency"].replace(0, np.nan)
    rfm["monetary_log"] = np.log1p(rfm["monetary"])
    rfm["recency_log"] = np.log1p(rfm["recency"])
    logger.info(f"RFM features created: {rfm.shape}")
    return rfm


__all__ = [
    "ExpertFeatureEngineer",
    "DatetimeFeatureExtractor",
    "create_rfm_features",
]
