# src/features/encoder.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
import category_encoders as ce
from feature_engine.encoding import RareLabelEncoder, MeanEncoder, WoEEncoder
from loguru import logger
from mltools.shared.exceptions import EncodingError
from typing import Dict, List, Optional


class ExpertCategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Encoder kategorikal production-grade.
    Strategy otomatis berdasarkan kardinalitas dan model type,
    atau bisa di-override manual per kolom.

    Parameters
    ----------
    model_type : 'tree' | 'linear' — pengaruhi strategi auto
    cardinality_threshold : Batas high/low cardinality (default: 10)
    high_card_method : 'target' | 'woe' | 'hash' | 'count'
    low_card_method : 'onehot' | 'ordinal'
    target_smoothing : Smoothing factor untuk target encoding (hindari overfit)
    handle_rare : Gabungkan kategori langka ke "Rare"
    rare_tol : Threshold frekuensi untuk "rare" (default: 0.05 = 5%)
    max_onehot_categories : Batas kategori untuk one-hot (cegah explosion)
    column_override : Dict manual override: {"col": "method"}
    """

    def __init__(
        self,
        model_type            : str = "tree",
        cardinality_threshold : int = 10,
        high_card_method      : str = "target",
        low_card_method       : str = "onehot",
        target_smoothing      : float = 5.0,
        handle_rare           : bool = True,
        rare_tol              : float = 0.05,
        max_onehot_categories : int = 15,
        column_override       : Optional[Dict[str, str]] = None,
    ):
        self.model_type            = model_type
        self.cardinality_threshold = cardinality_threshold
        self.high_card_method      = high_card_method
        self.low_card_method       = low_card_method
        self.target_smoothing      = target_smoothing
        self.handle_rare           = handle_rare
        self.rare_tol              = rare_tol
        self.max_onehot_categories = max_onehot_categories
        self.column_override       = column_override or {}

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit encoder pada training data.

        Example
        -------
        >>> encoder = ExpertCategoricalEncoder(model_type="tree",
        ...                                    high_card_method="target",
        ...                                    handle_rare=True)
        >>> encoder.fit(X_train, y_train)
        """
        self.cat_cols_ = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        if not self.cat_cols_:
            logger.info("Tidak ada kolom kategorikal ditemukan")
            return self

        logger.info(f"Fitting encoder untuk {len(self.cat_cols_)} kolom kategorikal")

        # ── 1. Rare Label Encoding ────────────────────────────────────────────
        if self.handle_rare:
            self.rare_encoder_ = RareLabelEncoder(
                tol          = self.rare_tol,
                n_categories = 5,
                replace_with = "Rare",
                variables    = self.cat_cols_,
            )
            self.rare_encoder_.fit(X)
            X = self.rare_encoder_.transform(X)

        # ── 2. Klasifikasi kolom ──────────────────────────────────────────────
        self.binary_cols_   = []
        self.low_card_cols_ = []
        self.high_card_cols_= []
        self.ordinal_map_   = {}

        for col in self.cat_cols_:
            n_unique = X[col].nunique()
            # Override manual
            if col in self.column_override:
                continue
            if n_unique <= 2:
                self.binary_cols_.append(col)
            elif n_unique <= self.cardinality_threshold:
                self.low_card_cols_.append(col)
            else:
                self.high_card_cols_.append(col)

        logger.info(f"  Binary cols    : {self.binary_cols_}")
        logger.info(f"  Low card cols  : {self.low_card_cols_}")
        logger.info(f"  High card cols : {self.high_card_cols_}")

        # ── 3. Fit Binary Encoder ─────────────────────────────────────────────
        if self.binary_cols_:
            self.binary_encoders_ = {}
            for col in self.binary_cols_:
                le = LabelEncoder()
                le.fit(X[col].astype(str))
                self.binary_encoders_[col] = le

        # ── 4. Fit Low Cardinality ────────────────────────────────────────────
        if self.low_card_cols_:
            if self.low_card_method == "onehot":
                # Batasi kategori untuk mencegah explosion
                self.onehot_encoder_ = OneHotEncoder(
                    sparse_output = False,
                    handle_unknown= "ignore",
                    max_categories= self.max_onehot_categories,
                    drop          = "if_binary",   # Hindari multikolinearitas
                )
                self.onehot_encoder_.fit(X[self.low_card_cols_])
                # Simpan nama kolom output
                self.onehot_feature_names_ = (
                    self.onehot_encoder_.get_feature_names_out(self.low_card_cols_)
                )
            else:   # ordinal
                self.ordinal_encoder_ = OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                )
                self.ordinal_encoder_.fit(X[self.low_card_cols_])

        # ── 5. Fit High Cardinality ───────────────────────────────────────────
        if self.high_card_cols_ and y is not None:
            if self.high_card_method == "target":
                # Target Encoding dengan Smoothing (mencegah overfit)
                self.target_encoder_ = ce.TargetEncoder(
                    cols             = self.high_card_cols_,
                    smoothing        = self.target_smoothing,
                    min_samples_leaf = 10,
                )
                self.target_encoder_.fit(X[self.high_card_cols_], y)
            elif self.high_card_method == "woe":
                # Weight of Evidence — bagus untuk credit scoring
                self.woe_encoder_ = ce.WOEEncoder(
                    cols       = self.high_card_cols_,
                    randomized = True,
                    sigma      = 0.05,   # Regularization
                )
                self.woe_encoder_.fit(X[self.high_card_cols_], y)
            elif self.high_card_method == "count":
                # Count Encoding — tidak butuh target, less overfit
                self.count_encoder_ = ce.CountEncoder(
                    cols           = self.high_card_cols_,
                    normalize      = True,
                    handle_unknown = 0,
                    handle_missing = "return_nan",
                )
                self.count_encoder_.fit(X[self.high_card_cols_])
            elif self.high_card_method == "hash":
                # Hash Encoding — cepat, untuk high cardinality ekstrem
                n_components = min(32, len(self.high_card_cols_) * 8)
                self.hash_encoder_ = ce.HashingEncoder(
                    cols        = self.high_card_cols_,
                    n_components= n_components,
                )
                self.hash_encoder_.fit(X[self.high_card_cols_])

        elif self.high_card_cols_ and y is None:
            # Fallback jika tidak ada target: gunakan count encoding
            logger.warning("y=None, menggunakan count encoding untuk high card cols")
            self.target_encoder_ = None
            self.high_card_method = "count"
            self.count_encoder_ = ce.CountEncoder(
                cols=self.high_card_cols_, normalize=True
            )
            self.count_encoder_.fit(X[self.high_card_cols_])

        logger.success("CategoricalEncoder fitted!")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply encoding pada data.

        Example
        -------
        >>> X_train_enc = encoder.transform(X_train)
        >>> X_val_enc   = encoder.transform(X_val)    # gunakan pemetaan dari train
        >>> X_test_enc  = encoder.transform(X_test)
        """
        X = X.copy()
        if not self.cat_cols_:
            return X

        # ── 1. Rare Label Transform ───────────────────────────────────────────
        if self.handle_rare and hasattr(self, "rare_encoder_"):
            X = self.rare_encoder_.transform(X)

        # ── 2. Binary Transform ───────────────────────────────────────────────
        if hasattr(self, "binary_encoders_"):
            for col, le in self.binary_encoders_.items():
                if col in X.columns:
                    X[col] = le.transform(X[col].astype(str))

        # ── 3. Low Card Transform ─────────────────────────────────────────────
        if self.low_card_cols_:
            if hasattr(self, "onehot_encoder_"):
                encoded = self.onehot_encoder_.transform(X[self.low_card_cols_])
                encoded_df = pd.DataFrame(
                    encoded,
                    columns=self.onehot_feature_names_,
                    index=X.index,
                )
                X = X.drop(columns=self.low_card_cols_)
                X = pd.concat([X, encoded_df], axis=1)
            elif hasattr(self, "ordinal_encoder_"):
                X[self.low_card_cols_] = self.ordinal_encoder_.transform(
                    X[self.low_card_cols_]
                )

        # ── 4. High Card Transform ────────────────────────────────────────────
        if self.high_card_cols_:
            if self.high_card_method == "target" and hasattr(self, "target_encoder_"):
                encoded = self.target_encoder_.transform(X[self.high_card_cols_])
                X[self.high_card_cols_] = encoded[self.high_card_cols_].values
            elif self.high_card_method == "woe" and hasattr(self, "woe_encoder_"):
                encoded = self.woe_encoder_.transform(X[self.high_card_cols_])
                X[self.high_card_cols_] = encoded[self.high_card_cols_].values
            elif self.high_card_method == "count" and hasattr(self, "count_encoder_"):
                encoded = self.count_encoder_.transform(X[self.high_card_cols_])
                X[self.high_card_cols_] = encoded[self.high_card_cols_].values
            elif self.high_card_method == "hash" and hasattr(self, "hash_encoder_"):
                encoded = self.hash_encoder_.transform(X[self.high_card_cols_])
                # Drop kolom asli dan tambah hasil hash
                X = X.drop(columns=self.high_card_cols_)
                X = pd.concat([X, encoded], axis=1)

        return X


# ── BONUS: Ordinal Encoding dengan Custom Order ───────────────────────────────

def encode_ordinal_custom(
    df   : pd.DataFrame,
    col  : str,
    order: List[str],
) -> pd.DataFrame:
    """
    Encode ordinal features dengan urutan yang ditentukan manual.

    Example
    -------
    >>> df = encode_ordinal_custom(df, "education",
    ...                            ["SD", "SMP", "SMA", "D3", "S1", "S2", "S3"])
    >>> df = encode_ordinal_custom(df, "risk_level", ["Low", "Medium", "High", "Critical"])
    """
    df        = df.copy()
    order_map = {val: i for i, val in enumerate(order)}
    df[col]   = df[col].map(order_map)
    # Validasi: ada nilai yang tidak di-map?
    unmapped = df[col].isnull()
    if unmapped.any():
        n_unmapped = unmapped.sum()
        logger.warning(f"⚠  {n_unmapped} nilai di kolom '{col}' tidak ada di order list!")
    return df


# ── Contoh Penggunaan ─────────────────────────────────────────────────────────
"""
encoder = ExpertCategoricalEncoder(
    model_type       = "tree",    # tree atau linear
    high_card_method = "target",  # target, woe, hash, count
    target_smoothing = 10.0,
    handle_rare      = True,
    rare_tol         = 0.03,
)

# WAJIB: fit hanya pada train, y diperlukan untuk target encoding
encoder.fit(X_train, y_train)
X_train_encoded = encoder.transform(X_train)
X_val_encoded   = encoder.transform(X_val)
X_test_encoded  = encoder.transform(X_test)

# Custom ordinal
df = encode_ordinal_custom(df, "risk_level", ["Low", "Medium", "High", "Critical"])
"""
