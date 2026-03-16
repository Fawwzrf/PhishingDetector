# src/mltools/modeling/cross_validator.py
# Perubahan: tambah MLConfig support + absolute imports

from __future__ import annotations

import numpy as np
from loguru import logger
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, KFold,
)

from mltools.shared.config     import MLConfig
from mltools.shared.exceptions import ModelingError


class CrossValidator:
    """
    Cross-validation utility untuk classification dan regression.

    Parameters
    ----------
    task         : 'classification' | 'regression'
    n_splits     : jumlah fold
    random_state : seed
    shuffle      : acak data sebelum split
    """

    def __init__(
        self,
        task        : str  = "classification",
        n_splits    : int  = 5,
        random_state: int  = 42,
        shuffle     : bool = True,
    ):
        self.task         = task
        self.n_splits     = n_splits
        self.random_state = random_state
        self.shuffle      = shuffle

    @classmethod
    def from_config(cls, config: MLConfig) -> "CrossValidator":
        """Buat CrossValidator dari MLConfig."""
        return cls(
            task         = config.modeling.task,
            n_splits     = config.modeling.n_cv_folds,
            random_state = config.project.random_state,
        )

    def get_cv(self, y):
        """Return CV splitter yang tepat berdasarkan task."""
        if self.task == "classification" and len(np.unique(y)) <= 20:
            return StratifiedKFold(
                n_splits     = self.n_splits,
                shuffle      = self.shuffle,
                random_state = self.random_state,
            )
        return KFold(
            n_splits     = self.n_splits,
            shuffle      = self.shuffle,
            random_state = self.random_state,
        )

    def score(
        self,
        model,
        X,
        y,
        scoring: str = "roc_auc",
    ) -> np.ndarray:
        """
        Jalankan cross-validation dan return array scores per fold.
        """
        cv = self.get_cv(y)
        logger.info(
            f"Running {self.n_splits}-fold CV "
            f"(scoring={scoring})..."
        )

        scores = cross_val_score(
            model, X, y,
            cv      = cv,
            scoring = scoring,
            n_jobs  = -1,
        )

        logger.info(f"CV scores per fold : {np.round(scores, 4)}")
        logger.info(
            f"Mean: {np.mean(scores):.4f} | "
            f"Std: {np.std(scores):.4f}"
        )
        return scores