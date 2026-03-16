# src/mltools/__init__.py

"""
mltools — Reusable ML Pipeline Library
==============================

Quick start (full pipeline):
    from mltools import FullMLPipeline, MLConfig
    result = FullMLPipeline(MLConfig.from_yaml("configs/ml_config.yaml")).run(df)

Quick start (EDA):
    from mltools import DataLoader, EDAVisualizer, generate_eda_report
    df      = DataLoader.load("data/raw/dataset.csv")
    eda     = EDAVisualizer(df, target="phishing")
    eda.run_full_eda()
"""

__version__ = "1.0.0"
__author__  = "mltools"

# ── Top-level pipeline ────────────────────────────────────────
from mltools.pipeline import FullMLPipeline

# ── Config & Schemas ──────────────────────────────────────────
from mltools.shared.config  import MLConfig
from mltools.shared.schemas import DataSplit, TrainingResult
from mltools.shared.logging import setup_logging

# ── Exceptions ────────────────────────────────────────────────
from mltools.shared.exceptions import (
    MLToolsError,
    ConfigError,
    DataError,
    PreprocessingError,
    ModelingError,
    ModelNotFittedError,
    PipelineError,
)

# ── Data (EDA & Loading) ─────────────────────────────────────
from mltools.data.loader  import DataLoader
from mltools.data.eda     import EDAVisualizer
from mltools.data.autoeda import generate_eda_report, quick_eda

# ── Preprocessing ─────────────────────────────────────────────
from mltools.preprocessing.pipeline        import PreprocessingPipeline
from mltools.preprocessing.inspector       import DataInspector
from mltools.preprocessing.missing_handler import ExpertMissingHandler
from mltools.preprocessing.outlier_handler import ExpertOutlierHandler
from mltools.preprocessing.encoder         import ExpertCategoricalEncoder
from mltools.preprocessing.scaler          import ExpertScalerTransformer
from mltools.preprocessing.engineer        import ExpertFeatureEngineer
from mltools.preprocessing.selector        import ExpertFeatureSelector
from mltools.preprocessing.splitter        import ExpertDataSplitter

# ── Modeling ──────────────────────────────────────────────────
from mltools.modeling.pipeline        import ModelingPipeline
from mltools.modeling.baseline        import BaselineModel
from mltools.modeling.boosting_models import (
    ExpertLightGBM,
    ExpertXGBoost,
    ExpertCatBoost,
)
from mltools.modeling.tree_models     import ExpertRandomForest
from mltools.modeling.evaluator       import ModelEvaluator
from mltools.modeling.tuner           import OptunaTuner

# ── Interpretation ────────────────────────────────────────────
from mltools.interpretation.shap_analysis import SHAPAnalyzer

# ── Registry ──────────────────────────────────────────────────
from mltools.registry.model_registry import ModelRegistry

__all__ = [
    # Top-level
    "FullMLPipeline",

    # Config & Schemas
    "MLConfig",
    "DataSplit",
    "TrainingResult",
    "setup_logging",

    # Exceptions
    "MLToolsError",
    "ConfigError",
    "DataError",
    "PreprocessingError",
    "ModelingError",
    "ModelNotFittedError",
    "PipelineError",

    # Data / EDA
    "DataLoader",
    "EDAVisualizer",
    "generate_eda_report",
    "quick_eda",

    # Preprocessing
    "PreprocessingPipeline",
    "DataInspector",
    "ExpertMissingHandler",
    "ExpertOutlierHandler",
    "ExpertCategoricalEncoder",
    "ExpertScalerTransformer",
    "ExpertFeatureEngineer",
    "ExpertFeatureSelector",
    "ExpertDataSplitter",

    # Modeling
    "ModelingPipeline",
    "BaselineModel",
    "ExpertLightGBM",
    "ExpertXGBoost",
    "ExpertCatBoost",
    "ExpertRandomForest",
    "ModelEvaluator",
    "OptunaTuner",

    # Interpretation
    "SHAPAnalyzer",

    # Registry
    "ModelRegistry",
]