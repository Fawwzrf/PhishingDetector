# src/mltools/modeling/evaluator.py
# Perubahan: tambah import exceptions + tambah support DataSplit

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, mean_squared_error,
    mean_absolute_error, r2_score,
)
from loguru import logger

from mltools.shared.exceptions import EvaluationError
from mltools.shared.schemas    import DataSplit


class ModelEvaluator:
    """
    Evaluator untuk classification dan regression.
    Mendukung evaluasi langsung dari DataSplit.

    Parameters
    ----------
    average : 'binary' | 'weighted' | 'macro'
    task    : 'classification' | 'regression'
    """

    def __init__(
        self,
        average: str = "binary",
        task   : str = "classification",
    ):
        self.average = average
        self.task    = task

    def evaluate(
        self,
        y_true,
        y_pred,
        y_prob = None,
    ) -> dict:
        """Evaluasi dengan y_true, y_pred, y_prob langsung."""
        if self.task == "classification":
            results = {
                "accuracy": accuracy_score(y_true, y_pred),
                "f1"      : f1_score(
                    y_true, y_pred,
                    average    = self.average,
                    zero_division = 0,
                ),
            }
            if y_prob is not None and len(np.unique(y_true)) == 2:
                results["roc_auc"] = roc_auc_score(y_true, y_prob)
        else:
            results = {
                "mae" : mean_absolute_error(y_true, y_pred),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "r2"  : r2_score(y_true, y_pred),
            }

        logger.info("Evaluation results:")
        for k, v in results.items():
            logger.info(f"  {k:<12}: {v:.4f}")
        return results

    def evaluate_from_split(
        self,
        model,
        split     : DataSplit,
        split_name: str = "val",
    ) -> dict:
        """
        Evaluasi model langsung dari DataSplit.
        split_name: 'val' atau 'test'
        """
        if split_name == "val":
            X, y = split.X_val, split.y_val
        elif split_name == "test":
            X, y = split.X_test, split.y_test
        else:
            raise EvaluationError(
                f"split_name tidak valid: {split_name}",
                details={"valid": ["val", "test"]},
            )

        y_pred = model.predict(X)
        y_prob = None

        if self.task == "classification" and hasattr(model, "predict_proba"):
            proba  = model.predict_proba(X)
            y_prob = proba[:, 1] if proba.shape[1] == 2 else None

        return self.evaluate(y, y_pred, y_prob)

    def plot_confusion(
        self,
        y_true,
        y_pred,
        labels = None,
        title  : str = "Confusion Matrix",
    ):
        cm   = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(
            confusion_matrix = cm,
            display_labels   = labels,
        )
        disp.plot(cmap=plt.cm.Blues)
        plt.title(title)
        plt.tight_layout()
        plt.savefig("reports/confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.show()

    def print_report(self, y_true, y_pred) -> str:
        report = classification_report(y_true, y_pred, zero_division=0)
        print(report)
        return report