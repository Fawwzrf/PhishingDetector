# src/mltools/modeling/boosting_models.py
# Perubahan dari versi asli:
# 1. Tambah import dari mltools.shared.exceptions
# 2. Ganti RuntimeError/ValueError dengan custom exceptions
# 3. Tidak ada perubahan logika

from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from mltools.shared.exceptions import ModelNotFittedError, ModelingError


class ExpertXGBoost:
    """
    XGBoost expert wrapper dengan early stopping dan auto scale_pos_weight.
    """

    def __init__(
        self,
        n_estimators         : int   = 2000,
        learning_rate        : float = 0.05,
        max_depth            : int   = 6,
        subsample            : float = 0.8,
        colsample_bytree     : float = 0.8,
        min_child_weight     : int   = 5,
        gamma                : float = 0.1,
        reg_alpha            : float = 0.1,
        reg_lambda           : float = 1.0,
        early_stopping_rounds: int   = 50,
        eval_metric          : str   = "auc",
        tree_method          : str   = "hist",
        device               : str   = "cpu",
        scale_pos_weight     : float = 1.0,
        n_jobs               : int   = -1,
        random_state         : int   = 42,
    ):
        import xgboost as xgb
        self.xgb = xgb
        self.params = dict(
            n_estimators          = n_estimators,
            learning_rate         = learning_rate,
            max_depth             = max_depth,
            subsample             = subsample,
            colsample_bytree      = colsample_bytree,
            min_child_weight      = min_child_weight,
            gamma                 = gamma,
            reg_alpha             = reg_alpha,
            reg_lambda            = reg_lambda,
            tree_method           = tree_method,
            device                = device,
            scale_pos_weight      = scale_pos_weight,
            n_jobs                = n_jobs,
            random_state          = random_state,
            eval_metric           = eval_metric,
            early_stopping_rounds = early_stopping_rounds,
        )
        self.feature_names = None
        self.model         = None

    def auto_set_imbalance(self, y: pd.Series):
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        if n_pos == 0:
            raise ModelingError("Tidak ada kelas positif di y_train")
        ratio = round(n_neg / n_pos, 2)
        self.params["scale_pos_weight"] = ratio
        logger.info(
            f"scale_pos_weight auto-set = {ratio:.2f} "
            f"(neg={n_neg:,}, pos={n_pos:,})"
        )

    def fit(
        self,
        X_train        : pd.DataFrame,
        y_train        : pd.Series,
        X_val          : pd.DataFrame,
        y_val          : pd.Series,
        auto_imbalance : bool = True,
    ) -> "ExpertXGBoost":
        from sklearn.metrics import roc_auc_score

        self.feature_names = X_train.columns.tolist()
        if auto_imbalance:
            self.auto_set_imbalance(y_train)

        self.model = self.xgb.XGBClassifier(**self.params, verbosity=1)
        logger.info("Training XGBoost dengan early stopping...")

        self.model.fit(
            X_train, y_train,
            eval_set = [(X_train, y_train), (X_val, y_val)],
            verbose  = 100,
        )

        y_prob   = self.model.predict_proba(X_val)[:, 1]
        val_auc  = roc_auc_score(y_val, y_prob)
        self.val_score_ = val_auc

        logger.success("XGBoost selesai!")
        logger.info(f"  Best iteration : {self.model.best_iteration}")
        logger.info(f"  Val ROC-AUC    : {val_auc:.4f}")
        return self

    def plot_training_curve(self):
        if self.model is None:
            raise ModelNotFittedError("xgboost")
        results = self.model.evals_result()
        metric  = self.params["eval_metric"]
        epochs  = len(results["validation_0"][metric])
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(range(epochs), results["validation_0"][metric], label="Train")
        ax.plot(range(epochs), results["validation_1"][metric], label="Validation")
        ax.set_xlabel("Epochs")
        ax.set_ylabel(metric)
        ax.set_title("XGBoost Training Curve")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def predict(self, X):
        if self.model is None:
            raise ModelNotFittedError("xgboost")
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.model is None:
            raise ModelNotFittedError("xgboost")
        return self.model.predict_proba(X)


class ExpertLightGBM:
    """
    LightGBM expert wrapper dengan native categorical support.
    """

    def __init__(
        self,
        n_estimators         : int            = 2000,
        learning_rate        : float          = 0.05,
        num_leaves           : int            = 63,
        max_depth            : int            = -1,
        min_data_in_leaf     : int            = 20,
        feature_fraction     : float          = 0.8,
        bagging_fraction     : float          = 0.8,
        bagging_freq         : int            = 5,
        lambda_l1            : float          = 0.1,
        lambda_l2            : float          = 0.1,
        early_stopping_rounds: int            = 50,
        metric               : str            = "auc",
        cat_features         : Optional[List[str]] = None,
        is_unbalance         : bool           = True,
        n_jobs               : int            = -1,
        device_type          : str            = "cpu",
        verbose              : int            = -1,
        random_state         : int            = 42,
    ):
        import lightgbm as lgb
        self.lgb = lgb
        self.params = dict(
            n_estimators     = n_estimators,
            learning_rate    = learning_rate,
            num_leaves       = num_leaves,
            max_depth        = max_depth,
            min_data_in_leaf = min_data_in_leaf,
            feature_fraction = feature_fraction,
            bagging_fraction = bagging_fraction,
            bagging_freq     = bagging_freq,
            lambda_l1        = lambda_l1,
            lambda_l2        = lambda_l2,
            is_unbalance     = is_unbalance,
            n_jobs           = n_jobs,
            device_type      = device_type,
            verbose          = verbose,
            random_state     = random_state,
            metric           = metric,
        )
        self.early_stopping_rounds = early_stopping_rounds
        self.cat_features  = cat_features or []
        self.feature_names = None
        self.model         = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val  : pd.DataFrame,
        y_val  : pd.Series,
    ) -> "ExpertLightGBM":
        self.feature_names = X_train.columns.tolist()
        cat_cols = [c for c in self.cat_features if c in X_train.columns]

        logger.info("Training LightGBM...")
        logger.info(
            f"  Categorical features: "
            f"{cat_cols if cat_cols else 'none'}"
        )

        # ── FIX: jangan pass categorical_feature ke constructor ──
        # Pass hanya ke fit() via callbacks, bukan ke LGBMClassifier()
        self.model = self.lgb.LGBMClassifier(**self.params)

        # Tentukan categorical_feature untuk fit()
        # Jika ada cat_cols → pakai nama kolom dengan prefix "name:"
        # Jika tidak ada    → jangan pass sama sekali
        fit_kwargs = {
            "eval_set" : [(X_val, y_val)],
            "callbacks": [
                self.lgb.early_stopping(
                    stopping_rounds = self.early_stopping_rounds,
                    verbose         = True,
                ),
                self.lgb.log_evaluation(period=100),
            ],
        }

        if cat_cols:
            # LightGBM butuh format "name:column_name"
            fit_kwargs["categorical_feature"] = [
                f"name:{c}" for c in cat_cols
            ]

        self.model.fit(X_train, y_train, **fit_kwargs)

        best_score      = self.model.best_score_["valid_0"][self.params["metric"]]
        self.val_score_ = best_score

        logger.success("LightGBM selesai!")
        logger.info(f"  Best iteration : {self.model.best_iteration_}")
        logger.info(f"  Best score     : {best_score:.4f}")
        return self

    def predict(self, X):
        if self.model is None:
            raise ModelNotFittedError("lightgbm")
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.model is None:
            raise ModelNotFittedError("lightgbm")
        return self.model.predict_proba(X)


class ExpertCatBoost:
    """
    CatBoost expert wrapper — tidak butuh encoding manual.
    """

    def __init__(
        self,
        iterations           : int   = 2000,
        learning_rate        : float = 0.05,
        depth                : int   = 6,
        l2_leaf_reg          : float = 3.0,
        early_stopping_rounds: int   = 50,
        eval_metric          : str   = "AUC",
        auto_class_weights   : str   = "Balanced",
        task_type            : str   = "CPU",
        thread_count         : int   = -1,
        random_seed          : int   = 42,
        verbose              : int   = 100,
    ):
        from catboost import CatBoostClassifier
        self.model = CatBoostClassifier(
            iterations            = iterations,
            learning_rate         = learning_rate,
            depth                 = depth,
            l2_leaf_reg           = l2_leaf_reg,
            early_stopping_rounds = early_stopping_rounds,
            eval_metric           = eval_metric,
            auto_class_weights    = auto_class_weights,
            task_type             = task_type,
            thread_count          = thread_count,
            random_seed           = random_seed,
            verbose               = verbose,
        )
        self.cat_features  = None
        self.feature_names = None
        self._fitted       = False

    def fit(
        self,
        X_train     : pd.DataFrame,
        y_train     : pd.Series,
        X_val       : pd.DataFrame,
        y_val       : pd.Series,
        cat_features: Optional[List[str]] = None,
    ) -> "ExpertCatBoost":
        from catboost import Pool

        self.feature_names = X_train.columns.tolist()
        if cat_features is None:
            cat_features = X_train.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
        self.cat_features = cat_features

        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        val_pool   = Pool(X_val,   y_val,   cat_features=cat_features)

        logger.info(
            f"Training CatBoost... "
            f"Categorical features: {cat_features}"
        )
        self.model.fit(train_pool, eval_set=val_pool)
        self._fitted = True

        best_score      = self.model.get_best_score()["validation"]["AUC"]
        self.val_score_ = best_score
        logger.success(f"CatBoost selesai! Best AUC: {best_score:.4f}")
        return self

    def predict(self, X):
        if not self._fitted:
            raise ModelNotFittedError("catboost")
        return self.model.predict(X)

    def predict_proba(self, X):
        if not self._fitted:
            raise ModelNotFittedError("catboost")
        return self.model.predict_proba(X)