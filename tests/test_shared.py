# tests/test_shared.py
# Unit test untuk shared/ layer: config, exceptions, schemas

import pytest
import pandas as pd
import numpy as np

from mltools.shared.config import (
    MLConfig, ScalingConfig, EncodingConfig,
    MissingValuesConfig, ModelingConfig,
)
from mltools.shared.exceptions import (
    ConfigError, DataError, ModelNotFittedError,
    PreprocessingError, ModelingError,
)
from mltools.shared.schemas import DataSplit, TrainingResult


# ══════════════════════════════════════════════════════════════
# CONFIG TESTS
# ══════════════════════════════════════════════════════════════

class TestMLConfig:

    def test_load_from_yaml(self, ml_config):
        """Config berhasil di-load dari YAML."""
        assert ml_config.project.name      == "test_project"
        assert ml_config.data.target_column == "target"
        assert ml_config.modeling.task      == "classification"

    def test_nested_access(self, ml_config):
        """Nested config bisa diakses dengan dot notation."""
        assert ml_config.preprocessing.scaling.strategy     == "robust"
        assert ml_config.preprocessing.encoding.default_strategy == "target"
        assert ml_config.modeling.tuning.n_trials           == 0

    def test_invalid_scaling_strategy(self):
        """ScalingConfig raise ConfigError jika strategy tidak valid."""
        with pytest.raises(ConfigError, match="scaling strategy"):
            ScalingConfig(strategy="invalid")

    def test_invalid_encoding_strategy(self):
        """EncodingConfig raise ConfigError jika strategy tidak valid."""
        with pytest.raises(ConfigError, match="encoding strategy"):
            EncodingConfig(default_strategy="unknown")

    def test_invalid_task(self):
        """ModelingConfig raise ConfigError jika task tidak valid."""
        with pytest.raises(ConfigError, match="task"):
            ModelingConfig(task="clustering")

    def test_invalid_model_name(self):
        """ModelingConfig raise ConfigError jika nama model tidak dikenal."""
        with pytest.raises(ConfigError, match="Model tidak dikenal"):
            ModelingConfig(models_to_try=["unknown_model"])

    def test_file_not_found(self):
        """from_yaml raise ConfigError jika file tidak ada."""
        with pytest.raises(ConfigError, match="tidak ditemukan"):
            MLConfig.from_yaml("tidak_ada.yaml")

    def test_summary_returns_string(self, ml_config):
        """summary() return string non-empty."""
        result = ml_config.summary()
        assert isinstance(result, str)
        assert len(result) > 0


# ══════════════════════════════════════════════════════════════
# EXCEPTION TESTS
# ══════════════════════════════════════════════════════════════

class TestExceptions:

    def test_exception_hierarchy(self):
        """Semua custom exception inherit dari MLToolsError."""
        from mltools.shared.exceptions import MLToolsError
        assert issubclass(ConfigError,          MLToolsError)
        assert issubclass(DataError,            MLToolsError)
        assert issubclass(PreprocessingError,   MLToolsError)
        assert issubclass(ModelingError,        MLToolsError)
        assert issubclass(ModelNotFittedError,  ModelingError)

    def test_exception_has_details(self):
        """Exception menyimpan details dict."""
        err = ModelingError("test", details={"key": "value"})
        assert err.details["key"] == "value"

    def test_exception_str_includes_details(self):
        """str(exception) menampilkan details."""
        err = ModelNotFittedError("my_model")
        assert "my_model" in str(err)

    def test_model_not_fitted_details(self):
        """ModelNotFittedError menyimpan nama model."""
        err = ModelNotFittedError("lightgbm")
        assert err.details["model_name"] == "lightgbm"


# ══════════════════════════════════════════════════════════════
# SCHEMA TESTS
# ══════════════════════════════════════════════════════════════

class TestDataSplit:

    def test_valid_split(self, data_split):
        """DataSplit valid berhasil dibuat."""
        assert data_split.task         == "classification"
        assert data_split.target_name  == "target"
        assert data_split.n_features   == 15
        assert data_split.n_classes    == 2
        assert data_split.is_binary    == True

    def test_shapes_property(self, data_split):
        """shapes property return dict yang benar."""
        shapes = data_split.shapes
        assert "train" in shapes
        assert "val"   in shapes
        assert "test"  in shapes
        assert shapes["train"][1] == data_split.n_features

    def test_class_balance_property(self, data_split):
        """class_balance return dict dengan semua kelas."""
        balance = data_split.class_balance
        assert isinstance(balance, dict)
        assert len(balance) == 2

    def test_mismatch_shapes_raise_error(self):
        """DataSplit raise AssertionError jika X dan y berbeda panjang."""
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([0, 1])   # Panjang berbeda!

        with pytest.raises(AssertionError):
            DataSplit(
                X_train=X, X_val=X, X_test=X,
                y_train=y, y_val=y, y_test=y,
                feature_names=["a"],
                target_name="target",
                task="classification",
            )

    def test_summary_returns_string(self, data_split):
        """summary() return string non-empty."""
        result = data_split.summary()
        assert isinstance(result, str)
        assert "classification" in result