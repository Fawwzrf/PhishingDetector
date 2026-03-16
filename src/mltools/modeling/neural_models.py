# src/mltools/modeling/neural_models.py
# Perubahan: tambah import exceptions

from __future__ import annotations

from loguru import logger
from sklearn.neural_network import MLPClassifier

from mltools.shared.exceptions import ModelNotFittedError


class ExpertMLPClassifier:
    """
    MLPClassifier wrapper untuk tabular data.
    Untuk deep learning yang lebih serius, gunakan PyTorch TabNet.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple = (100,),
        activation        : str   = "relu",
        solver            : str   = "adam",
        max_iter          : int   = 200,
        random_state      : int   = 42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation         = activation
        self.solver             = solver
        self.max_iter           = max_iter
        self.random_state       = random_state
        self.model              = None

    def fit(self, X, y) -> "ExpertMLPClassifier":
        logger.info(
            f"Training MLPClassifier "
            f"(layers={self.hidden_layer_sizes})..."
        )
        self.model = MLPClassifier(
            hidden_layer_sizes = self.hidden_layer_sizes,
            activation         = self.activation,
            solver             = self.solver,
            max_iter           = self.max_iter,
            random_state       = self.random_state,
        )
        self.model.fit(X, y)
        logger.success("MLPClassifier selesai!")
        return self

    def predict(self, X):
        if self.model is None:
            raise ModelNotFittedError("mlp_classifier")
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.model is None:
            raise ModelNotFittedError("mlp_classifier")
        return self.model.predict_proba(X)