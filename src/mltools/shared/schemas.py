# src/mltools/shared/schemas.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np


@dataclass
class DataSplit:
    """
    Output dari PreprocessingPipeline.
    Input untuk ModelingPipeline.

    Ini adalah 'kontrak' resmi antara dua dunia.
    Pastikan PreprocessingPipeline.run() selalu return DataSplit,
    dan ModelingPipeline.run() selalu terima DataSplit.
    """

    X_train       : pd.DataFrame
    X_val         : pd.DataFrame
    X_test        : pd.DataFrame
    y_train       : pd.Series
    y_val         : pd.Series
    y_test        : pd.Series
    feature_names : List[str]
    target_name   : str           # Dari config: "phishing"
    task          : str           # "classification" / "regression"
    n_classes     : Optional[int] = None   # 2 untuk binary, None untuk regression

    def __post_init__(self):
        # Validasi konsistensi shape
        assert len(self.X_train) == len(self.y_train), \
            f"X_train ({len(self.X_train)}) dan y_train ({len(self.y_train)}) tidak sama"
        assert len(self.X_val) == len(self.y_val), \
            f"X_val dan y_val tidak sama"
        assert len(self.X_test) == len(self.y_test), \
            f"X_test dan y_test tidak sama"
        assert list(self.X_train.columns) == self.feature_names, \
            "feature_names tidak cocok dengan kolom X_train"

        # Auto-detect n_classes jika belum diset
        if self.n_classes is None and self.task == "classification":
            self.n_classes = int(self.y_train.nunique())

    @property
    def shapes(self) -> Dict[str, tuple]:
        return {
            "train": self.X_train.shape,
            "val"  : self.X_val.shape,
            "test" : self.X_test.shape,
        }

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def is_binary(self) -> bool:
        return self.task == "classification" and self.n_classes == 2

    @property
    def class_balance(self) -> Optional[Dict]:
        """Distribusi kelas di training set."""
        if self.task != "classification":
            return None
        counts = self.y_train.value_counts(normalize=True)
        return counts.to_dict()

    def summary(self) -> str:
        lines = [
            f"DataSplit Summary",
            f"  Target    : {self.target_name} ({self.task})",
            f"  Train     : {self.X_train.shape[0]:,} samples",
            f"  Val       : {self.X_val.shape[0]:,} samples",
            f"  Test      : {self.X_test.shape[0]:,} samples",
            f"  Features  : {self.n_features}",
        ]
        if self.task == "classification":
            balance = self.class_balance
            lines.append(
                f"  Class dist: "
                + " | ".join(f"{k}={v:.1%}" for k, v in balance.items())
            )
        return "\n".join(lines)


@dataclass
class TrainingResult:
    """
    Output dari ModelingPipeline.
    Berisi semua yang perlu diketahui setelah training selesai.
    """

    champion_name  : str
    champion_model : Any                    # Model object asli
    all_scores     : Dict[str, float]       # {"lightgbm": 0.92, "xgboost": 0.91}
    test_metrics   : Dict[str, float]       # {"roc_auc": 0.91, "f1": 0.87}
    model_path     : str
    feature_names  : List[str]
    best_params    : Dict[str, Any] = field(default_factory=dict)
    shap_importance: Optional[pd.DataFrame] = None

    def summary(self) -> str:
        lines = [
            f"TrainingResult Summary",
            f"  Champion  : {self.champion_name}",
            f"  Model path: {self.model_path}",
            f"  Test metrics:",
        ]
        for k, v in self.test_metrics.items():
            lines.append(f"    {k:<20}: {v:.4f}")
        lines.append(f"  All scores:")
        for name, score in sorted(
            self.all_scores.items(),
            key=lambda x: x[1], reverse=True
        ):
            flag = " ← CHAMPION" if name == self.champion_name else ""
            lines.append(f"    {name:<20}: {score:.4f}{flag}")
        return "\n".join(lines)