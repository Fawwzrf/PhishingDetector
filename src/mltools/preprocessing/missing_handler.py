# src/data/missing_handler.py

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa — MUST be before IterativeImputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.base import BaseEstimator, TransformerMixin
from loguru import logger
from mltools.shared.exceptions import MissingHandlerError
from typing import Optional, List, Union


class ExpertMissingHandler(BaseEstimator, TransformerMixin):
    """
    Handler missing values expert-grade.

    Fitur:
    - Auto-detect kolom yang perlu di-drop
    - Strategi berbeda untuk numerik dan kategorikal
    - Tambah missing indicator otomatis
    - Sklearn-compatible (bisa masuk Pipeline)
    - fit() pada train, transform() pada test (no leakage!)

    Parameters
    ----------
    drop_col_threshold : Drop kolom jika missing > threshold (default: 0.6 = 60%)
    drop_row_threshold : Drop baris jika missing > threshold (default: None)
    num_strategy : Strategi numerik: 'mean' | 'median' | 'knn' | 'iterative'
    cat_strategy : Strategi kategorikal: 'most_frequent' | 'constant'
    cat_fill_value : Nilai untuk constant fill categorical
    add_missing_indicator : Tambah kolom binary _was_missing
    knn_neighbors : Jumlah neighbors untuk KNN imputer
    """

    def __init__(
        self,
        drop_col_threshold    : float = 0.6,
        drop_row_threshold    : Optional[float] = None,
        num_strategy          : str = "median",
        cat_strategy          : str = "most_frequent",
        cat_fill_value        : str = "MISSING",
        add_missing_indicator : bool = True,
        knn_neighbors         : int = 5,
    ):
        self.drop_col_threshold    = drop_col_threshold
        self.drop_row_threshold    = drop_row_threshold
        self.num_strategy          = num_strategy
        self.cat_strategy          = cat_strategy
        self.cat_fill_value        = cat_fill_value
        self.add_missing_indicator = add_missing_indicator
        self.knn_neighbors         = knn_neighbors

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit: Belajar dari training data SAJA.
        JANGAN pernah fit pada test data!

        Example
        -------
        >>> handler = ExpertMissingHandler(num_strategy="median", add_missing_indicator=True)
        >>> handler.fit(X_train)   # fit HANYA pada train
        """
        self._fitted_cols = X.columns.tolist()
        missing_pct = X.isnull().mean()

        # ── 1. Identifikasi kolom yang perlu di-drop ──────────────────────────
        self.cols_to_drop_ = missing_pct[
            missing_pct > self.drop_col_threshold
        ].index.tolist()
        logger.info(
            f"Kolom akan di-drop ({self.drop_col_threshold * 100:.0f}%+ missing): "
            f"{self.cols_to_drop_}"
        )

        # ── 2. Identifikasi tipe kolom ────────────────────────────────────────
        X_filtered = X.drop(columns=self.cols_to_drop_)
        self.num_cols_ = X_filtered.select_dtypes(include=np.number).columns.tolist()
        self.cat_cols_ = X_filtered.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        # ── 3. Identifikasi kolom dengan missing values ───────────────────────
        self.num_missing_cols_ = [
            c for c in self.num_cols_ if X_filtered[c].isnull().any()
        ]
        self.cat_missing_cols_ = [
            c for c in self.cat_cols_ if X_filtered[c].isnull().any()
        ]
        logger.info(f"Numerical cols with missing  : {self.num_missing_cols_}")
        logger.info(f"Categorical cols with missing: {self.cat_missing_cols_}")

        # ── 4. Fit imputer untuk numerik ──────────────────────────────────────
        if self.num_missing_cols_:
            if self.num_strategy in ["mean", "median"]:
                self.num_imputer_ = SimpleImputer(strategy=self.num_strategy)
                self.num_imputer_.fit(X_filtered[self.num_missing_cols_])
            elif self.num_strategy == "knn":
                self.num_imputer_ = KNNImputer(n_neighbors=self.knn_neighbors)
                self.num_imputer_.fit(X_filtered[self.num_missing_cols_])
            elif self.num_strategy == "iterative":
                # MICE-like imputation — paling akurat tapi lambat
                self.num_imputer_ = IterativeImputer(
                    max_iter=10,
                    random_state=42,
                    initial_strategy="median",
                )
                self.num_imputer_.fit(X_filtered[self.num_missing_cols_])

        # ── 5. Fit imputer untuk kategorikal ──────────────────────────────────
        if self.cat_missing_cols_:
            if self.cat_strategy == "most_frequent":
                self.cat_imputer_ = SimpleImputer(strategy="most_frequent")
            else:
                self.cat_imputer_ = SimpleImputer(
                    strategy="constant",
                    fill_value=self.cat_fill_value,
                )
            self.cat_imputer_.fit(X_filtered[self.cat_missing_cols_])

        # ── 6. Catat kolom yang perlu missing indicator ───────────────────────
        if self.add_missing_indicator:
            self.indicator_cols_ = self.num_missing_cols_ + self.cat_missing_cols_
        else:
            self.indicator_cols_ = []

        logger.success("MissingHandler fitted!")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply imputation pada data baru.

        Example
        -------
        >>> X_train_clean = handler.transform(X_train)
        >>> X_val_clean   = handler.transform(X_val)    # pakai imputer dari train
        >>> X_test_clean  = handler.transform(X_test)   # pakai imputer dari train
        """
        X = X.copy()

        # ── 1. Drop baris jika perlu ──────────────────────────────────────────
        if self.drop_row_threshold:
            row_missing_pct = X.isnull().mean(axis=1)
            rows_to_drop = X[row_missing_pct > self.drop_row_threshold].index
            X = X.drop(index=rows_to_drop)
            logger.info(
                f"Dropped {len(rows_to_drop)} rows dengan > "
                f"{self.drop_row_threshold * 100:.0f}% missing"
            )

        # ── 2. Drop kolom ─────────────────────────────────────────────────────
        X = X.drop(columns=self.cols_to_drop_, errors="ignore")

        # ── 3. Tambah missing indicators SEBELUM impute ───────────────────────
        if self.add_missing_indicator:
            for col in self.indicator_cols_:
                if col in X.columns:
                    X[f"{col}_was_missing"] = X[col].isnull().astype(int)

        # ── 4. Impute numerik ─────────────────────────────────────────────────
        if self.num_missing_cols_:
            cols_present = [c for c in self.num_missing_cols_ if c in X.columns]
            if cols_present:
                imputed = self.num_imputer_.transform(X[cols_present])
                X[cols_present] = imputed

        # ── 5. Impute kategorikal ─────────────────────────────────────────────
        if self.cat_missing_cols_:
            cols_present = [c for c in self.cat_missing_cols_ if c in X.columns]
            if cols_present:
                imputed = self.cat_imputer_.transform(X[cols_present])
                X[cols_present] = imputed

        logger.debug(f"Transform selesai. Shape: {X.shape}")
        return X


# ── Helper: Analisis Pola Missing ─────────────────────────────────────────────

def analyze_missing_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analisis mendalam pola missing — membantu decision making.

    Example
    -------
    >>> pattern = analyze_missing_pattern(df_train)
    >>> print(pattern)   # DataFrame: column, n_missing, pct_missing, recommendation
    """
    missing_info = []
    for col in df.columns:
        n_missing = df[col].isnull().sum()
        if n_missing == 0:
            continue
        pct_missing = n_missing / len(df)
        dtype = df[col].dtype
        # Sarankan strategi
        if pct_missing > 0.6:
            recommendation = "🗑  DROP kolom"
        elif str(dtype) in ["object", "category"]:
            if pct_missing > 0.3:
                recommendation = "❓ Tambah kategori 'MISSING'"
            else:
                recommendation = "✏  MODE imputation"
        else:
            skew = abs(df[col].skew()) if pct_missing < 1 else 0
            if pct_missing < 0.05:
                recommendation = "✏  MEDIAN imputation"
            elif skew > 1:
                recommendation = "✏  MEDIAN imputation (skewed)"
            elif pct_missing > 0.3:
                recommendation = "🔍 KNN / MICE imputation"
            else:
                recommendation = "✏  MEDIAN imputation"
        missing_info.append({
            "column"        : col,
            "n_missing"     : n_missing,
            "pct_missing"   : f"{pct_missing * 100:.1f}%",
            "dtype"         : str(dtype),
            "recommendation": recommendation,
        })
    return pd.DataFrame(missing_info).sort_values("n_missing", ascending=False)


# ── KASUS KHUSUS 1: Time Series dengan Forward Fill ──────────────────────────

def handle_timeseries_missing(
    df: pd.DataFrame,
    date_col: str,
    method: str = "ffill",
) -> pd.DataFrame:
    """
    Untuk data time series: forward fill lebih logis dari mean/median.

    Example
    -------
    >>> df = handle_timeseries_missing(df, date_col="event_date", method="ffill")
    >>> df = handle_timeseries_missing(df, date_col="event_date", method="interpolate")
    """
    df = df.sort_values(date_col).copy()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if date_col in num_cols:
        num_cols.remove(date_col)
    if method == "ffill":
        df[num_cols] = df[num_cols].ffill().bfill()
    elif method == "interpolate":
        df[num_cols] = df[num_cols].interpolate(method="time")
    return df


# ── KASUS KHUSUS 2: Group-Based Imputation ───────────────────────────────────

def group_imputation(
    df: pd.DataFrame,
    col: str,
    group_col: str,
    strategy: str = "median",
) -> pd.DataFrame:
    """
    Impute berdasarkan grup — lebih akurat dari global imputation.
    Contoh: Impute umur berdasarkan group pekerjaan.

    Example
    -------
    >>> df = group_imputation(df, col="age",    group_col="job",    strategy="median")
    >>> df = group_imputation(df, col="income", group_col="region", strategy="mean")
    """
    df = df.copy()
    if strategy == "median":
        fill_values = df.groupby(group_col)[col].transform("median")
    elif strategy == "mean":
        fill_values = df.groupby(group_col)[col].transform("mean")
    elif strategy == "mode":
        fill_values = df.groupby(group_col)[col].transform(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan
        )
    df[col] = df[col].fillna(fill_values)
    # Fallback: global imputation untuk yang masih missing
    if df[col].isnull().any():
        if strategy == "median":
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


# ── KASUS KHUSUS 3: Validasi Setelah Imputation ──────────────────────────────

def validate_no_missing(df: pd.DataFrame, raise_error: bool = True) -> bool:
    """
    Validasi tidak ada missing values tersisa setelah imputation.

    Example
    -------
    >>> validate_no_missing(X_train_clean)               # raise ValueError jika ada missing
    >>> ok = validate_no_missing(X_train_clean, raise_error=False)  # return bool
    """
    remaining_missing = df.isnull().sum().sum()
    if remaining_missing > 0:
        problematic = df.columns[df.isnull().any()].tolist()
        msg = f"MASIH ADA {remaining_missing} missing values di kolom: {problematic}"
        if raise_error:
            raise MissingHandlerError(msg)
        else:
            logger.warning(msg)
        return False
    logger.success("✅ Tidak ada missing values tersisa!")
    return True


# ── Contoh Penggunaan ─────────────────────────────────────────────────────────
"""
# 1. Analisis dulu
pattern = analyze_missing_pattern(df_train)
print(pattern)

# 2. Buat handler
handler = ExpertMissingHandler(
    drop_col_threshold   = 0.6,
    num_strategy         = "iterative",   # MICE untuk accuracy tinggi
    cat_strategy         = "most_frequent",
    add_missing_indicator= True,          # Capture MNAR pattern
)

# 3. Fit HANYA pada train
handler.fit(X_train)

# 4. Transform semua split
X_train_clean = handler.transform(X_train)
X_val_clean   = handler.transform(X_val)   # Gunakan imputer dari train!
X_test_clean  = handler.transform(X_test)  # Gunakan imputer dari train!

print(f"Missing setelah handling: {X_train_clean.isnull().sum().sum()}")
"""
