# tests/conftest.py
# Shared fixtures untuk semua test

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from mltools.shared.config  import MLConfig
from mltools.shared.schemas import DataSplit


# ── DATA FIXTURES ─────────────────────────────────────────────

@pytest.fixture(scope="session")
def classification_data():
    """Dataset klasifikasi binary — dibuat sekali per session."""
    X, y = make_classification(
        n_samples     = 500,
        n_features    = 15,
        n_informative = 8,
        n_redundant   = 3,
        random_state  = 42,
    )
    X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(15)])
    y_s  = pd.Series(y, name="target")
    return X_df, y_s


@pytest.fixture(scope="session")
def regression_data():
    """Dataset regresi — dibuat sekali per session."""
    X, y = make_regression(
        n_samples    = 500,
        n_features   = 10,
        noise        = 0.1,
        random_state = 42,
    )
    X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)])
    y_s  = pd.Series(y, name="target")
    return X_df, y_s


@pytest.fixture(scope="session")
def raw_dataframe(classification_data):
    """DataFrame mentah dengan kolom target — untuk test pipeline."""
    X, y = classification_data
    df   = X.copy()
    df["target"] = y.values
    # Tambah kolom kategorikal
    df["category"] = np.random.choice(["A", "B", "C"], len(df))
    return df


# ── SPLIT FIXTURES ────────────────────────────────────────────

@pytest.fixture(scope="session")
def data_split(classification_data):
    """DataSplit siap pakai — untuk test modeling."""
    X, y = classification_data

    n       = len(X)
    tr_end  = int(n * 0.6)
    vl_end  = int(n * 0.8)

    split = DataSplit(
        X_train      = X.iloc[:tr_end].reset_index(drop=True),
        X_val        = X.iloc[tr_end:vl_end].reset_index(drop=True),
        X_test       = X.iloc[vl_end:].reset_index(drop=True),
        y_train      = y.iloc[:tr_end].reset_index(drop=True),
        y_val        = y.iloc[tr_end:vl_end].reset_index(drop=True),
        y_test       = y.iloc[vl_end:].reset_index(drop=True),
        feature_names= list(X.columns),
        target_name  = "target",
        task         = "classification",
    )
    return split


# ── CONFIG FIXTURE ────────────────────────────────────────────

@pytest.fixture(scope="session")
def ml_config(tmp_path_factory):
    """MLConfig dengan dataset phishing — dari YAML."""
    tmp = tmp_path_factory.mktemp("config")
    yaml_content = """
project:
  name        : "test_project"
  version     : "0.1.0"
  random_state: 42
  log_level   : "WARNING"

data:
  target_column: "target"
  id_columns   : []
  date_columns : []

preprocessing:
  missing_values:
    threshold_drop_column  : 0.6
    threshold_drop_row     : 0.5
    strategy_numerical     : "median"
    strategy_categorical   : "most_frequent"
  outliers:
    method   : "iqr"
    threshold: 1.5
    treatment: "cap"
  encoding:
    high_cardinality_threshold: 10
    default_strategy          : "target"
  scaling:
    strategy: "robust"
  feature_selection:
    variance_threshold   : 0.01
    correlation_threshold: 0.95
    n_features_to_select : "auto"

modeling:
  task        : "classification"
  metric      : "roc_auc"
  n_cv_folds  : 3
  random_state: 42
  baseline:
    strategy: "most_frequent"
  models_to_try:
    - logistic_regression
    - lightgbm
  tuning:
    n_trials: 0
    timeout : null
    sampler : "tpe"

mlflow:
  experiment_name: "test_experiment"
  tracking_uri   : "http://localhost:5000"
"""
    config_path = tmp / "ml_config.yaml"
    config_path.write_text(yaml_content)
    return MLConfig.from_yaml(str(config_path))