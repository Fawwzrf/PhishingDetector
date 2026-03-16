# src/mltools/shared/config.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Literal
import yaml

from mltools.shared.exceptions import ConfigError


# ══════════════════════════════════════════════════════════════
# PROJECT
# ══════════════════════════════════════════════════════════════

@dataclass
class ProjectConfig:
    name        : str  = "my_ml_project"
    version     : str  = "1.0.0"
    random_state: int  = 42
    log_level   : str  = "INFO"


# ══════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════

@dataclass
class DataConfig:
    target_column: str        = "target"
    id_columns   : List[str]  = field(default_factory=list)
    date_columns : List[str]  = field(default_factory=list)


# ══════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════

@dataclass
class MissingValuesConfig:
    threshold_drop_column  : float = 0.6
    threshold_drop_row     : float = 0.5
    strategy_numerical     : str   = "median"
    strategy_categorical   : str   = "most_frequent"

    def __post_init__(self):
        valid_num = {"mean", "median", "knn", "iterative"}
        valid_cat = {"most_frequent", "constant", "knn"}

        if self.strategy_numerical not in valid_num:
            raise ConfigError(
                f"strategy_numerical tidak valid: {self.strategy_numerical}",
                details={"valid": str(valid_num)}
            )
        if self.strategy_categorical not in valid_cat:
            raise ConfigError(
                f"strategy_categorical tidak valid: {self.strategy_categorical}",
                details={"valid": str(valid_cat)}
            )


@dataclass
class OutliersConfig:
    method   : str   = "iqr"
    threshold: float = 1.5
    treatment: str   = "cap"

    def __post_init__(self):
        valid_methods    = {"iqr", "zscore", "isolation_forest", "lof"}
        valid_treatments = {"cap", "remove", "transform"}

        if self.method not in valid_methods:
            raise ConfigError(
                f"outlier method tidak valid: {self.method}",
                details={"valid": str(valid_methods)}
            )
        if self.treatment not in valid_treatments:
            raise ConfigError(
                f"outlier treatment tidak valid: {self.treatment}",
                details={"valid": str(valid_treatments)}
            )


@dataclass
class EncodingConfig:
    high_cardinality_threshold: int = 20
    default_strategy          : str = "target"

    def __post_init__(self):
        valid = {"onehot", "ordinal", "target", "woe"}
        if self.default_strategy not in valid:
            raise ConfigError(
                f"encoding strategy tidak valid: {self.default_strategy}",
                details={"valid": str(valid)}
            )


@dataclass
class ScalingConfig:
    strategy: str = "robust"

    def __post_init__(self):
        valid = {"standard", "minmax", "robust", "power"}
        if self.strategy not in valid:
            raise ConfigError(
                f"scaling strategy tidak valid: {self.strategy}",
                details={"valid": str(valid)}
            )


@dataclass
class FeatureSelectionConfig:
    variance_threshold   : float      = 0.01
    correlation_threshold: float      = 0.95
    n_features_to_select : str | int  = "auto"


@dataclass
class PreprocessingConfig:
    missing_values   : MissingValuesConfig    = field(default_factory=MissingValuesConfig)
    outliers         : OutliersConfig         = field(default_factory=OutliersConfig)
    encoding         : EncodingConfig         = field(default_factory=EncodingConfig)
    scaling          : ScalingConfig          = field(default_factory=ScalingConfig)
    feature_selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)


# ══════════════════════════════════════════════════════════════
# MODELING
# ══════════════════════════════════════════════════════════════

@dataclass
class BaselineConfig:
    strategy: str = "most_frequent"


@dataclass
class TuningConfig:
    n_trials: int          = 100
    timeout : Optional[int] = 3600
    sampler : str           = "tpe"


@dataclass
class ModelingConfig:
    task          : str          = "classification"
    metric        : str          = "roc_auc"
    n_cv_folds    : int          = 5
    random_state  : int          = 42
    baseline      : BaselineConfig = field(default_factory=BaselineConfig)
    models_to_try : List[str]    = field(
        default_factory=lambda: ["lightgbm"]
    )
    tuning        : TuningConfig = field(default_factory=TuningConfig)

    def __post_init__(self):
        valid_tasks = {"classification", "regression"}
        if self.task not in valid_tasks:
            raise ConfigError(
                f"task tidak valid: {self.task}",
                details={"valid": str(valid_tasks)}
            )

        valid_models = {
            "logistic_regression", "random_forest",
            "xgboost", "lightgbm", "catboost"
        }
        invalid = set(self.models_to_try) - valid_models
        if invalid:
            raise ConfigError(
                f"Model tidak dikenal: {invalid}",
                details={"valid": str(valid_models)}
            )


# ══════════════════════════════════════════════════════════════
# MLFLOW
# ══════════════════════════════════════════════════════════════

@dataclass
class MLflowConfig:
    experiment_name: str = "default_experiment"
    tracking_uri   : str = "http://localhost:5000"


# ══════════════════════════════════════════════════════════════
# MASTER CONFIG
# ══════════════════════════════════════════════════════════════

@dataclass
class MLConfig:
    """
    Single source of truth untuk seluruh pipeline.

    Cara pakai:
        config = MLConfig.from_yaml("configs/ml_config.yaml")
        print(config.data.target_column)       # "phishing"
        print(config.modeling.task)            # "classification"
        print(config.preprocessing.scaling.strategy)  # "robust"
    """

    project       : ProjectConfig       = field(default_factory=ProjectConfig)
    data          : DataConfig          = field(default_factory=DataConfig)
    preprocessing : PreprocessingConfig = field(default_factory=PreprocessingConfig)
    modeling      : ModelingConfig      = field(default_factory=ModelingConfig)
    mlflow        : MLflowConfig        = field(default_factory=MLflowConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MLConfig":
        """
        Load config dari YAML file.
        Raises ConfigError jika file tidak ada atau format salah.
        """
        path = Path(path)

        if not path.exists():
            raise ConfigError(
                f"Config file tidak ditemukan: {path}"
            )

        try:
            with open(path, encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"YAML tidak valid: {e}")

        try:
            prep_raw = raw.get("preprocessing", {})
            mod_raw  = raw.get("modeling", {})
            mlf_raw  = raw.get("mlflow", {})
            proj_raw = raw.get("project", {})
            data_raw = raw.get("data", {})

            return cls(
                project = ProjectConfig(**proj_raw),
                data    = DataConfig(**data_raw),

                preprocessing = PreprocessingConfig(
                    missing_values    = MissingValuesConfig(
                        **prep_raw.get("missing_values", {})
                    ),
                    outliers          = OutliersConfig(
                        **prep_raw.get("outliers", {})
                    ),
                    encoding          = EncodingConfig(
                        **prep_raw.get("encoding", {})
                    ),
                    scaling           = ScalingConfig(
                        **prep_raw.get("scaling", {})
                    ),
                    feature_selection = FeatureSelectionConfig(
                        **prep_raw.get("feature_selection", {})
                    ),
                ),

                modeling = ModelingConfig(
                    task         = mod_raw.get("task", "classification"),
                    metric       = mod_raw.get("metric", "roc_auc"),
                    n_cv_folds   = mod_raw.get("n_cv_folds", 5),
                    random_state = mod_raw.get("random_state", 42),
                    baseline     = BaselineConfig(
                        **mod_raw.get("baseline", {})
                    ),
                    models_to_try= mod_raw.get("models_to_try", ["lightgbm"]),
                    tuning       = TuningConfig(
                        **mod_raw.get("tuning", {})
                    ),
                ),

                mlflow = MLflowConfig(**mlf_raw),
            )

        except (TypeError, KeyError) as e:
            raise ConfigError(
                f"Struktur YAML tidak sesuai: {e}. "
                "Cek apakah semua key di YAML sudah benar."
            )

    def validate(self) -> "MLConfig":
        """
        Validasi silang antar section config.
        Panggil setelah from_yaml() untuk verifikasi menyeluruh.
        """
        # random_state harus konsisten
        if self.modeling.random_state != self.project.random_state:
            import warnings
            warnings.warn(
                f"random_state di project ({self.project.random_state}) "
                f"dan modeling ({self.modeling.random_state}) berbeda. "
                "Disarankan pakai satu nilai yang sama."
            )

        return self

    def summary(self) -> str:
        lines = [
            f"MLConfig: {self.project.name} v{self.project.version}",
            f"  Target      : {self.data.target_column}",
            f"  Task        : {self.modeling.task}",
            f"  Metric      : {self.modeling.metric}",
            f"  Models      : {self.modeling.models_to_try}",
            f"  CV Folds    : {self.modeling.n_cv_folds}",
            f"  Scaling     : {self.preprocessing.scaling.strategy}",
            f"  Encoding    : {self.preprocessing.encoding.default_strategy}",
            f"  Tuning      : {self.modeling.tuning.n_trials} trials",
        ]
        return "\n".join(lines)