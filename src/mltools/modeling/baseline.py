# src/mltools/modeling/baseline.py

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    accuracy_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score,
)

from mltools.shared.exceptions import ModelingError
from mltools.shared.schemas    import DataSplit


class BaselineModel:
    """
    Baseline model menggunakan DummyClassifier / DummyRegressor.

    WAJIB dijalankan sebelum model apapun.
    Jika model asli tidak bisa mengalahkan baseline → ada bug serius.

    Parameters
    ----------
    task     : 'classification' | 'regression'
    strategy : strategi Dummy (default 'most_frequent')
    """

    def __init__(
        self,
        task        : str = "classification",
        strategy    : str = "most_frequent",
        random_state: int = 42,
    ):
        self.task         = task
        self.strategy     = strategy
        self.random_state = random_state
        self.model        = None
        self.scores_      : dict = {}

    # ── Fit dengan X, y biasa (backward compatible) ───────────
    def fit(self, X, y) -> "BaselineModel":
        """Fit baseline. Bisa terima X/y langsung atau DataSplit."""
        if self.task == "classification":
            self.model = DummyClassifier(
                strategy     = self.strategy,
                random_state = self.random_state,
            )
        else:
            self.model = DummyRegressor(strategy=self.strategy)

        logger.info(
            f"Fitting baseline model "
            f"(task={self.task}, strategy={self.strategy})..."
        )
        self.model.fit(X, y)
        return self

    # ── Fit + evaluasi sekaligus dari DataSplit ────────────────
    def evaluate_from_split(self, split: DataSplit) -> dict:
        """
        Fit pada train, evaluasi pada val — dari DataSplit langsung.

        Returns dict berisi semua metrik baseline.
        """
        logger.info("─" * 50)
        logger.info("BASELINE EVALUATION")
        logger.info("─" * 50)

        self.fit(split.X_train, split.y_train)

        y_pred = self.model.predict(split.X_val)

        if self.task == "classification":
            self.scores_ = {
                "accuracy": accuracy_score(split.y_val, y_pred),
                "f1"      : f1_score(
                    split.y_val, y_pred,
                    average="weighted", zero_division=0
                ),
            }
            # AUC untuk binary
            if split.is_binary and hasattr(self.model, "predict_proba"):
                y_proba = self.model.predict_proba(split.X_val)[:, 1]
                try:
                    self.scores_["roc_auc"] = roc_auc_score(
                        split.y_val, y_proba
                    )
                except ValueError:
                    self.scores_["roc_auc"] = 0.5
        else:
            self.scores_ = {
                "mae" : mean_absolute_error(split.y_val, y_pred),
                "rmse": float(np.sqrt(
                    mean_squared_error(split.y_val, y_pred)
                )),
                "r2"  : r2_score(split.y_val, y_pred),
            }

        logger.info(f"Strategy : {self.strategy}")
        for k, v in self.scores_.items():
            logger.info(f"  {k:<12}: {v:.4f}")
        logger.info("─" * 50)

        return self.scores_

    def is_better(self, score: float, metric: str = "roc_auc") -> bool:
        """
        Cek apakah model asli lebih baik dari baseline.
        Raise ModelingError jika tidak lebih baik.
        """
        if not self.scores_:
            raise ModelingError(
                "Jalankan evaluate_from_split() dulu sebelum is_better()"
            )

        baseline_score = self.scores_.get(metric, 0.0)

        if score <= baseline_score:
            raise ModelingError(
                f"Model ({score:.4f}) tidak mengalahkan baseline "
                f"({baseline_score:.4f}) pada metrik '{metric}'. "
                "Investigasi preprocessing dan feature engineering!",
                details={
                    "model_score"   : score,
                    "baseline_score": baseline_score,
                    "metric"        : metric,
                },
            )

        improvement = score - baseline_score
        logger.success(
            f"Model ({score:.4f}) mengalahkan baseline "
            f"({baseline_score:.4f}) → +{improvement:.4f}"
        )
        return True

    # ── Backward compatible predict ───────────────────────────
    def predict(self, X):
        if self.model is None:
            raise ModelingError("BaselineModel belum di-fit")
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.task != "classification":
            raise ModelingError(
                "predict_proba hanya untuk classification"
            )
        if self.model is None:
            raise ModelingError("BaselineModel belum di-fit")
        return self.model.predict_proba(X)