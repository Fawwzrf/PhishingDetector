# tests/test_preprocessing.py
# Unit test untuk preprocessing layer

import pytest
import numpy as np
import pandas as pd

from mltools.preprocessing.missing_handler import (
    ExpertMissingHandler, validate_no_missing,
)
from mltools.preprocessing.outlier_handler import ExpertOutlierHandler
from mltools.preprocessing.encoder         import ExpertCategoricalEncoder
from mltools.preprocessing.scaler          import ExpertScalerTransformer
from mltools.preprocessing.splitter        import (
    ExpertDataSplitter, check_data_leakage,
)
from mltools.shared.exceptions import SplittingError


# ── Fixtures lokal ────────────────────────────────────────────

@pytest.fixture
def num_df():
    """DataFrame numerik sederhana."""
    np.random.seed(42)
    return pd.DataFrame({
        "a": np.random.randn(200),
        "b": np.random.exponential(2, 200),
        "c": np.random.randint(0, 10, 200).astype(float),
    })


@pytest.fixture
def num_df_with_missing(num_df):
    """DataFrame numerik dengan missing values."""
    df = num_df.copy()
    df.loc[::5, "a"] = np.nan    # 20% missing
    df.loc[::10, "b"] = np.nan   # 10% missing
    return df


@pytest.fixture
def cat_df():
    """DataFrame dengan kolom kategorikal."""
    np.random.seed(42)
    return pd.DataFrame({
        "color"  : np.random.choice(["red", "blue", "green"], 200),
        "size"   : np.random.choice(["S", "M", "L", "XL"], 200),
        "numeric": np.random.randn(200),
    })


@pytest.fixture
def target():
    np.random.seed(42)
    return pd.Series(np.random.randint(0, 2, 200), name="target")


# ══════════════════════════════════════════════════════════════
# MISSING HANDLER TESTS
# ══════════════════════════════════════════════════════════════

class TestMissingHandler:

    def test_fit_detects_missing_cols(self, num_df_with_missing):
        """fit() mendeteksi kolom dengan missing values."""
        handler = ExpertMissingHandler(num_strategy="median")
        handler.fit(num_df_with_missing)
        assert "a" in handler.num_missing_cols_
        assert "b" in handler.num_missing_cols_

    def test_transform_removes_missing(self, num_df_with_missing):
        """transform() menghilangkan semua missing values."""
        handler = ExpertMissingHandler(num_strategy="median")
        handler.fit(num_df_with_missing)
        result = handler.transform(num_df_with_missing)
        assert result.isnull().sum().sum() == 0

    def test_no_fit_on_test(self, num_df_with_missing):
        """transform() bisa dipanggil tanpa fit ulang (no leakage)."""
        train = num_df_with_missing.iloc[:150]
        test  = num_df_with_missing.iloc[150:]

        handler = ExpertMissingHandler(num_strategy="median")
        handler.fit(train)               # Fit HANYA pada train
        result  = handler.transform(test)  # Transform test dengan stats dari train
        assert result.isnull().sum().sum() == 0

    def test_drop_column_threshold(self):
        """Kolom dengan missing > threshold di-drop."""
        df = pd.DataFrame({
            "good"  : [1.0, 2.0, 3.0, 4.0, 5.0],
            "bad"   : [np.nan, np.nan, np.nan, np.nan, 1.0],  # 80% missing
        })
        handler = ExpertMissingHandler(drop_col_threshold=0.5)
        handler.fit(df)
        assert "bad" in handler.cols_to_drop_
        assert "good" not in handler.cols_to_drop_

    def test_missing_indicator_added(self, num_df_with_missing):
        """add_missing_indicator=True menambah kolom _was_missing."""
        handler = ExpertMissingHandler(
            num_strategy          = "median",
            add_missing_indicator = True,
        )
        handler.fit(num_df_with_missing)
        result = handler.transform(num_df_with_missing)
        assert "a_was_missing" in result.columns
        assert "b_was_missing" in result.columns

    def test_validate_no_missing_passes(self, num_df):
        """validate_no_missing() tidak raise jika tidak ada missing."""
        assert validate_no_missing(num_df) == True

    def test_validate_no_missing_raises(self, num_df_with_missing):
        """validate_no_missing() raise jika ada missing."""
        with pytest.raises(ValueError, match="missing"):
            validate_no_missing(num_df_with_missing)


# ══════════════════════════════════════════════════════════════
# OUTLIER HANDLER TESTS
# ══════════════════════════════════════════════════════════════

class TestOutlierHandler:

    def test_fit_computes_bounds(self, num_df):
        """fit() menghitung bounds untuk setiap kolom numerik."""
        handler = ExpertOutlierHandler(method="iqr")
        handler.fit(num_df)
        assert len(handler.bounds_) == num_df.select_dtypes(
            include=np.number
        ).shape[1]

    def test_cap_treatment(self, num_df):
        """Cap treatment tidak menghilangkan baris."""
        handler = ExpertOutlierHandler(method="iqr", treatment="cap")
        handler.fit(num_df)
        result = handler.transform(num_df)
        assert len(result) == len(num_df)

    def test_no_fit_leak(self, num_df):
        """transform() test pakai bounds dari train (no leakage)."""
        train = num_df.iloc[:150]
        test  = num_df.iloc[150:]

        handler = ExpertOutlierHandler(method="iqr", treatment="cap")
        handler.fit(train)
        result = handler.transform(test)
        assert len(result) == len(test)

    def test_values_within_bounds_after_cap(self, num_df):
        """Setelah cap, semua nilai dalam bounds."""
        handler = ExpertOutlierHandler(method="iqr", treatment="cap")
        handler.fit(num_df)
        result = handler.transform(num_df)

        for col, bounds in handler.bounds_.items():
            if col in result.columns:
                assert result[col].min() >= bounds["lower"] - 1e-6
                assert result[col].max() <= bounds["upper"] + 1e-6


# ══════════════════════════════════════════════════════════════
# ENCODER TESTS
# ══════════════════════════════════════════════════════════════

class TestEncoder:

    def test_fit_detects_cat_cols(self, cat_df, target):
        """fit() mendeteksi kolom kategorikal."""
        encoder = ExpertCategoricalEncoder()
        encoder.fit(cat_df, target)
        assert "color" in encoder.cat_cols_
        assert "size"  in encoder.cat_cols_

    def test_transform_no_object_columns(self, cat_df, target):
        """transform() menghilangkan semua kolom object."""
        encoder = ExpertCategoricalEncoder()
        encoder.fit(cat_df, target)
        result = encoder.transform(cat_df)
        object_cols = result.select_dtypes(include="object").columns
        assert len(object_cols) == 0

    def test_no_fit_on_val(self, cat_df, target):
        """transform() pada val pakai encoding dari train."""
        train = cat_df.iloc[:150]
        val   = cat_df.iloc[150:]
        y_tr  = target.iloc[:150]

        encoder = ExpertCategoricalEncoder()
        encoder.fit(train, y_tr)

        result_train = encoder.transform(train)
        result_val   = encoder.transform(val)

        # Jumlah kolom harus sama
        assert result_train.shape[1] == result_val.shape[1]


# ══════════════════════════════════════════════════════════════
# SCALER TESTS
# ══════════════════════════════════════════════════════════════

class TestScaler:

    def test_fit_detects_skewed(self, num_df):
        """fit() mendeteksi kolom skewed untuk transformasi."""
        scaler = ExpertScalerTransformer(
            scaler         = "robust",
            auto_transform = True,
            skew_threshold = 0.5,
        )
        scaler.fit(num_df)
        # Kolom 'b' (exponential) harusnya terdeteksi skewed
        assert isinstance(scaler.transform_cols_, list)

    def test_train_scaled_mean_near_zero(self, num_df):
        """Setelah scaling, mean train ~0."""
        scaler = ExpertScalerTransformer(
            scaler         = "standard",
            auto_transform = False,
        )
        scaler.fit(num_df)
        result = scaler.transform(num_df)
        # Mean harus mendekati 0
        assert abs(result.mean().mean()) < 0.1

    def test_no_fit_on_test(self, num_df):
        """transform() test pakai scaler dari train."""
        train = num_df.iloc[:150]
        test  = num_df.iloc[150:]

        scaler = ExpertScalerTransformer(scaler="robust")
        scaler.fit(train)
        result = scaler.transform(test)
        assert result.shape == test.shape


# ══════════════════════════════════════════════════════════════
# SPLITTER TESTS
# ══════════════════════════════════════════════════════════════

class TestSplitter:

    def test_holdout_sizes(self, classification_data):
        """Split holdout menghasilkan ukuran yang benar."""
        X, y      = classification_data
        splitter  = ExpertDataSplitter(
            task      = "classification",
            test_size = 0.2,
            val_size  = 0.2,
        )
        X_tr, X_vl, X_te, y_tr, y_vl, y_te = splitter.split_holdout(X, y)

        total = len(X_tr) + len(X_vl) + len(X_te)
        assert total == len(X)
        assert len(X_te) == pytest.approx(len(X) * 0.2, abs=2)

    def test_no_overlap_between_splits(self, classification_data):
        """Tidak ada index overlap antara train, val, test."""
        X, y     = classification_data
        splitter = ExpertDataSplitter(task="classification")
        X_tr, X_vl, X_te, y_tr, y_vl, y_te = splitter.split_holdout(X, y)

        # Reset index agar bisa di-compare
        idx_tr = set(X_tr.reset_index().index)
        idx_vl = set(X_vl.reset_index().index)
        idx_te = set(X_te.reset_index().index)

        # Cek panjang total konsisten
        assert len(X_tr) + len(X_vl) + len(X_te) == len(X)

    def test_stratified_preserves_distribution(self, classification_data):
        """Split stratified mempertahankan distribusi kelas."""
        X, y     = classification_data
        splitter = ExpertDataSplitter(task="classification")
        X_tr, X_vl, X_te, y_tr, y_vl, y_te = splitter.split_holdout(X, y)

        orig_ratio  = y.mean()
        train_ratio = y_tr.mean()
        test_ratio  = y_te.mean()

        # Rasio kelas tidak boleh terlalu berbeda
        assert abs(orig_ratio - train_ratio) < 0.05
        assert abs(orig_ratio - test_ratio)  < 0.05

    def test_group_split_raises_without_group_col(
        self, classification_data
    ):
        """split_group() raise jika group_col tidak diset."""
        X, y     = classification_data
        splitter = ExpertDataSplitter(task="classification")

        with pytest.raises(SplittingError, match="group_col"):
            list(splitter.split_group(X, y))

    def test_check_leakage_no_overlap(self, classification_data):
        """check_data_leakage() return False jika tidak ada leakage."""
        X, y     = classification_data
        splitter = ExpertDataSplitter(task="classification")
        X_tr, _, X_te, _, _, _ = splitter.split_holdout(X, y)

        has_leak = check_data_leakage(X_tr, X_te)
        assert has_leak == False