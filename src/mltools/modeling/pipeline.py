# src/mltools/modeling/pipeline.py
# BARU: pakai DataSplit sebagai input, return TrainingResult
# Menggantikan versi lama yang terima X/y terpisah

from __future__ import annotations

import joblib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score, r2_score

from mltools.shared.config     import MLConfig
from mltools.shared.exceptions import ModelingError, PipelineError
from mltools.shared.schemas    import DataSplit, TrainingResult

from mltools.modeling.baseline        import BaselineModel
from mltools.modeling.linear_models   import ExpertLogisticRegression
from mltools.modeling.tree_models     import ExpertRandomForest
from mltools.modeling.boosting_models import (
    ExpertXGBoost, ExpertLightGBM, ExpertCatBoost,
)
from mltools.modeling.evaluator       import ModelEvaluator
from mltools.modeling.cross_validator import CrossValidator
from mltools.modeling.tuner           import OptunaTuner


# ── Map nama model (dari YAML) ke class ──────────────────────
_MODEL_REGISTRY = {
    "logistic_regression": ExpertLogisticRegression,
    "random_forest"      : ExpertRandomForest,
    "xgboost"            : ExpertXGBoost,
    "lightgbm"           : ExpertLightGBM,
    "catboost"           : ExpertCatBoost,
}


class ModelingPipeline:
    """
    Full modeling pipeline yang membaca MLConfig
    dan menerima DataSplit dari PreprocessingPipeline.

    Cara pakai:
        config   = MLConfig.from_yaml("configs/ml_config.yaml")
        pipeline = ModelingPipeline(config)
        result   = pipeline.run(data_split)

    Flow internal:
        1. Baseline evaluation
        2. Train semua model di models_to_try
        3. Pilih champion berdasarkan val score
        4. Opsional: hyperparameter tuning
        5. Final evaluation di test set
        6. Save model + return TrainingResult
    """

    def __init__(self, config: MLConfig):
        self.config    = config
        self.evaluator = ModelEvaluator(
            task = config.modeling.task,
        )
        self.cv = CrossValidator.from_config(config)

    def run(self, split: DataSplit) -> TrainingResult:
        """
        Jalankan full modeling pipeline.

        Args:
            split : DataSplit dari PreprocessingPipeline

        Returns:
            TrainingResult berisi champion model + semua metrics
        """
        cfg = self.config

        logger.info("=" * 55)
        logger.info(f"MODELING PIPELINE: {cfg.project.name}")
        logger.info(
            f"Task: {cfg.modeling.task} | "
            f"Metric: {cfg.modeling.metric}"
        )
        logger.info(f"Models: {cfg.modeling.models_to_try}")
        logger.info("=" * 55)

        # ── Step 1: Baseline ──────────────────────────────────
        logger.info("[1/5] BASELINE EVALUATION")
        baseline = BaselineModel(
            task        = cfg.modeling.task,
            strategy    = cfg.modeling.baseline.strategy,
            random_state= cfg.project.random_state,
        )
        baseline_scores = baseline.evaluate_from_split(split)

        # ── Step 2: Train semua model ─────────────────────────
        logger.info("[2/5] TRAINING ALL MODELS")
        all_models : dict = {}
        all_scores : dict = {}

        for model_name in cfg.modeling.models_to_try:
            if model_name not in _MODEL_REGISTRY:
                raise ModelingError(
                    f"Model '{model_name}' tidak ada di registry",
                    details={"available": list(_MODEL_REGISTRY.keys())},
                )

            logger.info(f"  → Training {model_name}...")
            model, score = self._train_model(model_name, split)
            all_models[model_name] = model
            all_scores[model_name] = score
            logger.info(
                f"    {model_name}: "
                f"{cfg.modeling.metric}={score:.4f}"
            )

        # ── Step 3: Pilih champion ────────────────────────────
        logger.info("[3/5] SELECTING CHAMPION")

        direction = "max" if cfg.modeling.metric in {
            "roc_auc", "f1", "accuracy", "r2"
        } else "min"

        if direction == "max":
            champion_name = max(all_scores, key=all_scores.get)
        else:
            champion_name = min(all_scores, key=all_scores.get)

        champion_model = all_models[champion_name]

        logger.info("  Model comparison (val score):")
        for name, score in sorted(
            all_scores.items(),
            key     = lambda x: x[1],
            reverse = (direction == "max"),
        ):
            flag = " <- CHAMPION" if name == champion_name else ""
            logger.info(f"    {name:<25}: {score:.4f}{flag}")

        # Cek vs baseline
        baseline_metric = baseline_scores.get(cfg.modeling.metric, 0.0)
        if all_scores[champion_name] <= baseline_metric:
            logger.warning(
                f"PERINGATAN: Champion ({all_scores[champion_name]:.4f}) "
                f"tidak mengalahkan baseline ({baseline_metric:.4f})! "
                "Periksa preprocessing dan feature engineering."
            )

        # ── Step 4: Tuning (opsional) ─────────────────────────
        best_params : dict = {}

        if cfg.modeling.tuning.n_trials > 0:
            logger.info(
                f"[4/5] TUNING CHAMPION: {champion_name} "
                f"({cfg.modeling.tuning.n_trials} trials)"
            )
            champion_model, best_params, tuned_score = self._tune_model(
                champion_name, champion_model, split
            )
            # Update score setelah tuning
            if tuned_score > all_scores[champion_name]:
                all_scores[champion_name] = tuned_score
                logger.success(
                    f"Tuning improved score: "
                    f"{all_scores[champion_name]:.4f} → {tuned_score:.4f}"
                )
        else:
            logger.info("[4/5] Tuning dilewati (n_trials=0)")

        # ── Step 5: Final evaluation di TEST SET ──────────────
        logger.warning("[5/5] FINAL TEST SET EVALUATION")
        logger.warning("Test set hanya boleh dievaluasi SEKALI!")

        test_metrics = self._evaluate_test(champion_model, split)

        logger.info("Test metrics:")
        for k, v in test_metrics.items():
            logger.info(f"  {k:<15}: {v:.4f}")

        # ── Save champion ─────────────────────────────────────
        model_path = self._save_model(
            champion_model, champion_name, test_metrics
        )

        # ── Build TrainingResult ──────────────────────────────
        result = TrainingResult(
            champion_name  = champion_name,
            champion_model = self._get_raw_model(champion_model),
            all_scores     = all_scores,
            test_metrics   = test_metrics,
            model_path     = model_path,
            feature_names  = split.feature_names,
            best_params    = best_params,
        )

        logger.success("=" * 55)
        logger.success("MODELING PIPELINE SELESAI!")
        logger.success(result.summary())
        logger.success("=" * 55)

        return result

    # ── Internal helpers ──────────────────────────────────────

    def _train_model(
        self,
        model_name: str,
        split     : DataSplit,
    ) -> tuple:
        """Train satu model, return (model, val_score)."""
        rs   = self.config.project.random_state
        task = self.config.modeling.task

        if model_name == "logistic_regression":
            model = ExpertLogisticRegression()
            model.fit(split.X_train, split.y_train)
            score = self._score_model(model, split)

        elif model_name == "random_forest":
            model = ExpertRandomForest(
                task         = task,
                n_estimators = 200,
                random_state = rs,
            )
            model.fit(split.X_train, split.y_train,
                      split.X_val, split.y_val)
            score = model.val_score_ or self._score_model(model, split)

        elif model_name == "xgboost":
            model = ExpertXGBoost(random_state=rs)
            model.fit(split.X_train, split.y_train,
                      split.X_val, split.y_val)
            score = model.val_score_

        elif model_name == "lightgbm":
            model = ExpertLightGBM(random_state=rs)
            model.fit(split.X_train, split.y_train,
                      split.X_val, split.y_val)
            score = model.val_score_

        elif model_name == "catboost":
            model = ExpertCatBoost(random_seed=rs)
            model.fit(split.X_train, split.y_train,
                      split.X_val, split.y_val)
            score = model.val_score_

        else:
            raise ModelingError(f"Model tidak dikenal: {model_name}")

        return model, score

    def _score_model(self, model, split: DataSplit) -> float:
        """Hitung val score untuk model yang tidak return val_score_."""
        cfg  = self.config
        task = cfg.modeling.task

        y_pred = model.predict(split.X_val)

        if task == "classification":
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(split.X_val)
                if proba.shape[1] == 2:
                    return roc_auc_score(split.y_val, proba[:, 1])
                return roc_auc_score(
                    split.y_val, proba, multi_class="ovr"
                )
        return r2_score(split.y_val, y_pred)

    def _tune_model(
        self,
        model_name  : str,
        model,
        split       : DataSplit,
    ) -> tuple:
        """
        Tune champion model dengan Optuna.
        Return (tuned_model, best_params, best_score).
        """
        cfg = self.config

        # ── Definisi search space per model ───────────────────
        search_spaces = {
            "lightgbm": {
                "n_estimators"    : lambda t: t.suggest_int("n_estimators", 200, 2000, step=100),
                "learning_rate"   : lambda t: t.suggest_float("learning_rate", 1e-4, 0.3, log=True),
                "num_leaves"      : lambda t: t.suggest_int("num_leaves", 15, 255),
                "min_data_in_leaf": lambda t: t.suggest_int("min_data_in_leaf", 5, 100),
                "lambda_l1"       : lambda t: t.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                "lambda_l2"       : lambda t: t.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                "feature_fraction": lambda t: t.suggest_float("feature_fraction", 0.4, 1.0),
                "bagging_fraction": lambda t: t.suggest_float("bagging_fraction", 0.4, 1.0),
                "verbose"         : lambda t: -1,
                "random_state"    : lambda t: cfg.project.random_state,
            },
            "xgboost": {
                "n_estimators"    : lambda t: t.suggest_int("n_estimators", 100, 2000, step=50),
                "learning_rate"   : lambda t: t.suggest_float("learning_rate", 1e-4, 0.3, log=True),
                "max_depth"       : lambda t: t.suggest_int("max_depth", 3, 12),
                "min_child_weight": lambda t: t.suggest_int("min_child_weight", 1, 50),
                "subsample"       : lambda t: t.suggest_float("subsample", 0.4, 1.0),
                "colsample_bytree": lambda t: t.suggest_float("colsample_bytree", 0.3, 1.0),
                "gamma"           : lambda t: t.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha"       : lambda t: t.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda"      : lambda t: t.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "random_state"    : lambda t: cfg.project.random_state,
            },
            "random_forest": {
                "n_estimators"    : lambda t: t.suggest_int("n_estimators", 100, 800, step=50),
                "max_depth"       : lambda t: t.suggest_int("max_depth", 3, 30),
                "min_samples_leaf": lambda t: t.suggest_int("min_samples_leaf", 1, 20),
                "max_features"    : lambda t: t.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
                "random_state"    : lambda t: cfg.project.random_state,
            },
        }

        if model_name not in search_spaces:
            logger.info(
                f"Tidak ada search space untuk {model_name}, skip tuning"
            )
            return model, {}, self._score_model(model, split)

        # Gunakan model class yang sesuai
        model_class_map = {
            "lightgbm"    : self._get_lgbm_class(),
            "xgboost"     : self._get_xgb_class(),
            "random_forest": ExpertRandomForest,
        }
        sklearn_class = model_class_map.get(model_name)
        if sklearn_class is None:
            return model, {}, self._score_model(model, split)

        tuner = OptunaTuner.from_config(
            model_class = sklearn_class,
            param_space = search_spaces[model_name],
            config      = cfg,
        )

        best_params, best_score = tuner.tune(
            split.X_train, split.y_train
        )

        # Refit dengan best params
        tuned_model = self._refit_with_params(
            model_name, best_params, split
        )

        return tuned_model, best_params, best_score

    def _get_lgbm_class(self):
        """Return sklearn-compatible LightGBM class untuk Optuna."""
        import lightgbm as lgb
        return (lgb.LGBMClassifier
                if self.config.modeling.task == "classification"
                else lgb.LGBMRegressor)

    def _get_xgb_class(self):
        """Return sklearn-compatible XGBoost class untuk Optuna."""
        import xgboost as xgb
        return (xgb.XGBClassifier
                if self.config.modeling.task == "classification"
                else xgb.XGBRegressor)

    def _refit_with_params(
        self,
        model_name : str,
        params     : dict,
        split      : DataSplit,
    ):
        """Refit model dengan best params dari Optuna."""
        import lightgbm as lgb
        import xgboost as xgb

        if model_name == "lightgbm":
            ModelClass = (lgb.LGBMClassifier
                         if self.config.modeling.task == "classification"
                         else lgb.LGBMRegressor)
            m = ModelClass(**params)
            m.fit(
                split.X_train, split.y_train,
                eval_set  = [(split.X_val, split.y_val)],
                callbacks = [
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )

        elif model_name == "xgboost":
            ModelClass = (xgb.XGBClassifier
                         if self.config.modeling.task == "classification"
                         else xgb.XGBRegressor)
            m = ModelClass(**params, verbosity=0)
            m.fit(
                split.X_train, split.y_train,
                eval_set = [(split.X_val, split.y_val)],
                verbose  = False,
            )

        elif model_name == "random_forest":
            from sklearn.ensemble import (
                RandomForestClassifier, RandomForestRegressor,
            )
            ModelClass = (
                RandomForestClassifier
                if self.config.modeling.task == "classification"
                else RandomForestRegressor
            )
            m = ModelClass(**params, n_jobs=-1)
            m.fit(split.X_train, split.y_train)

        else:
            raise ModelingError(f"Refit tidak support: {model_name}")

        # Wrap ke object sederhana dengan .predict dan .predict_proba
        return _SklearnModelWrapper(m, model_name)

    def _evaluate_test(
        self,
        model,
        split: DataSplit,
    ) -> dict:
        """Evaluasi di test set."""
        raw_model = self._get_raw_model(model)

        y_pred = raw_model.predict(split.X_test)
        task   = self.config.modeling.task

        if task == "classification":
            from sklearn.metrics import (
                accuracy_score, f1_score,
                roc_auc_score, matthews_corrcoef,
            )
            metrics = {
                "accuracy": accuracy_score(split.y_test, y_pred),
                "f1"      : f1_score(
                    split.y_test, y_pred,
                    average="weighted", zero_division=0,
                ),
            }
            if hasattr(raw_model, "predict_proba"):
                proba = raw_model.predict_proba(split.X_test)
                if proba.shape[1] == 2:
                    metrics["roc_auc"] = roc_auc_score(
                        split.y_test, proba[:, 1]
                    )
            metrics["mcc"] = matthews_corrcoef(split.y_test, y_pred)
        else:
            from sklearn.metrics import (
                mean_absolute_error, mean_squared_error, r2_score,
            )
            metrics = {
                "mae" : mean_absolute_error(split.y_test, y_pred),
                "rmse": float(np.sqrt(
                    mean_squared_error(split.y_test, y_pred)
                )),
                "r2"  : r2_score(split.y_test, y_pred),
            }

        return metrics

    def _save_model(
        self,
        model,
        name   : str,
        metrics: dict,
    ) -> str:
        """Save model ke disk, return path."""
        Path("models").mkdir(parents=True, exist_ok=True)

        raw_model  = self._get_raw_model(model)
        model_path = f"models/{name}_champion.joblib"
        joblib.dump(raw_model, model_path, compress=3)

        # Juga save feature names untuk validasi saat inference
        import json
        meta = {
            "model_name"   : name,
            "metrics"      : {
                k: round(float(v), 4) for k, v in metrics.items()
            },
            "feature_names": self.config.data.target_column,
        }
        with open(f"models/{name}_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.success(f"Model saved: {model_path}")
        return model_path

    def _get_raw_model(self, model):
        """Ambil sklearn model asli dari wrapper."""
        if isinstance(model, _SklearnModelWrapper):
            return model.model
        # Untuk model kita sendiri (ExpertLightGBM, dll)
        if hasattr(model, "model"):
            return model.model
        if hasattr(model, "model_"):
            return model.model_
        return model

    # ── Backward compatible methods ───────────────────────────

    def fit(
        self,
        X_train,
        y_train,
        X_val  = None,
        y_val  = None,
        **kwargs,
    ):
        """
        Backward compatible fit().
        Direkomendasikan pakai .run(DataSplit) untuk penggunaan baru.
        """
        logger.warning(
            "ModelingPipeline.fit() adalah backward-compat mode. "
            "Gunakan .run(DataSplit) untuk full pipeline."
        )
        # Langsung train model pertama dari daftar
        if not self.config.modeling.models_to_try:
            raise PipelineError("models_to_try kosong di config")

        model_name = self.config.modeling.models_to_try[0]
        logger.info(f"Fitting {model_name}...")

        if model_name == "lightgbm":
            self._active_model = ExpertLightGBM(
                random_state=self.config.project.random_state
            )
            if X_val is not None:
                self._active_model.fit(X_train, y_train, X_val, y_val)
            else:
                self._active_model.model.fit(X_train, y_train)
        else:
            raise PipelineError(
                "fit() hanya support lightgbm. "
                "Pakai .run(DataSplit) untuk semua model."
            )
        self.fitted = True
        return self

    def predict(self, X):
        if not hasattr(self, "_active_model"):
            raise PipelineError("Jalankan fit() atau run() dulu")
        return self._active_model.predict(X)

    def predict_proba(self, X):
        if not hasattr(self, "_active_model"):
            raise PipelineError("Jalankan fit() atau run() dulu")
        return self._active_model.predict_proba(X)


class _SklearnModelWrapper:
    """
    Wrapper tipis untuk sklearn model hasil refit.
    Menyeragamkan interface .predict() dan .predict_proba().
    """

    def __init__(self, model, name: str):
        self.model = model
        self.name  = name

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise ModelingError(
            f"{self.name} tidak mendukung predict_proba"
        )