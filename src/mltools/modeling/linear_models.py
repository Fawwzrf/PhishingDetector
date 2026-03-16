# src/mltools/modeling/linear_models.py
# Perubahan: tambah import exceptions

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression, LinearRegression

from mltools.shared.exceptions import ModelNotFittedError


class ExpertLogisticRegression:
    """
    Logistic Regression dengan default yang baik dan logging.
    """

    def __init__(
        self,
        penalty     : str   = "l2",
        C           : float = 1.0,
        solver      : str   = "lbfgs",
        max_iter    : int   = 1000,
        random_state: int   = 42,
    ):
        self.penalty      = penalty
        self.C            = C
        self.solver       = solver
        self.max_iter     = max_iter
        self.random_state = random_state
        self.model        = None

    def fit(self, X, y) -> "ExpertLogisticRegression":
        self.model = LogisticRegression(
            penalty      = self.penalty,
            C            = self.C,
            solver       = self.solver,
            max_iter     = self.max_iter,
            random_state = self.random_state,
            n_jobs       = -1,
        )
        logger.info("Training Logistic Regression...")
        self.model.fit(X, y)
        logger.success("Logistic Regression selesai!")
        return self

    def predict(self, X):
        if self.model is None:
            raise ModelNotFittedError("logistic_regression")
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.model is None:
            raise ModelNotFittedError("logistic_regression")
        return self.model.predict_proba(X)

    @property
    def coef_(self):
        if self.model is None:
            raise ModelNotFittedError("logistic_regression")
        return self.model.coef_


class ExpertLinearRegression:
    """Linear Regression dengan logging."""

    def __init__(self):
        self.model = None

    def fit(self, X, y) -> "ExpertLinearRegression":
        logger.info("Training Linear Regression...")
        self.model = LinearRegression(n_jobs=-1)
        self.model.fit(X, y)
        logger.success("Linear Regression selesai!")
        return self

    def predict(self, X):
        if self.model is None:
            raise ModelNotFittedError("linear_regression")
        return self.model.predict(X)