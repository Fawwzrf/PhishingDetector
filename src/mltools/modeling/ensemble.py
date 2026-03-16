# src/mltools/modeling/ensemble.py
# Perubahan: relative → absolute imports + tambah exceptions

from __future__ import annotations

from loguru import logger
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingClassifier, StackingClassifier

from mltools.shared.exceptions import ModelNotFittedError


class VotingEnsembler(BaseEstimator, ClassifierMixin):
    """
    Voting ensemble untuk classification.

    Parameters
    ----------
    estimators : list of (name, estimator) tuples
    voting     : 'soft' | 'hard'
    """

    def __init__(
        self,
        estimators,
        voting : str = "soft",
        n_jobs : int = -1,
    ):
        self.estimators = estimators
        self.voting     = voting
        self.n_jobs     = n_jobs
        self.model      = None

    def fit(self, X, y) -> "VotingEnsembler":
        logger.info(
            f"Training VotingEnsembler "
            f"({len(self.estimators)} models, voting={self.voting})..."
        )
        self.model = VotingClassifier(
            estimators = self.estimators,
            voting     = self.voting,
            n_jobs     = self.n_jobs,
        )
        self.model.fit(X, y)
        logger.success("VotingEnsembler selesai!")
        return self

    def predict(self, X):
        if self.model is None:
            raise ModelNotFittedError("voting_ensembler")
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.model is None:
            raise ModelNotFittedError("voting_ensembler")
        return self.model.predict_proba(X)


class StackingEnsembler(BaseEstimator, ClassifierMixin):
    """
    Stacking ensemble untuk classification.

    Parameters
    ----------
    estimators       : list of (name, estimator) tuples — Level 0
    final_estimator  : model meta-learner — Level 1
    passthrough      : pass fitur asli ke meta-learner juga
    """

    def __init__(
        self,
        estimators,
        final_estimator,
        n_jobs     : int  = -1,
        passthrough: bool = False,
    ):
        self.estimators      = estimators
        self.final_estimator = final_estimator
        self.n_jobs          = n_jobs
        self.passthrough     = passthrough
        self.model           = None

    def fit(self, X, y) -> "StackingEnsembler":
        logger.info(
            f"Training StackingEnsembler "
            f"({len(self.estimators)} base models)..."
        )
        self.model = StackingClassifier(
            estimators       = self.estimators,
            final_estimator  = self.final_estimator,
            n_jobs           = self.n_jobs,
            passthrough      = self.passthrough,
        )
        self.model.fit(X, y)
        logger.success("StackingEnsembler selesai!")
        return self

    def predict(self, X):
        if self.model is None:
            raise ModelNotFittedError("stacking_ensembler")
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.model is None:
            raise ModelNotFittedError("stacking_ensembler")
        return self.model.predict_proba(X)