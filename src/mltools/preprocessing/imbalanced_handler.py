"""
Handler imbalanced data — production-grade.
HANYA apply pada training data! Jangan pernah pada test/validation.
"""
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE, SMOTENC
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from loguru import logger
from sklearn.metrics import precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight

# Suppress warnings from imblearn internals
import warnings
warnings.filterwarnings("ignore", module="imblearn")


class ExpertImbalancedHandler:
    """
    Handler imbalanced data production-grade.
    HANYA apply pada training data!

    Parameters
    ----------
    strategy : str
        'smote' | 'smotenc' | 'adasyn' | 'smotetomek' | 'smoteenn' |
        'undersample' | 'oversample' | 'class_weight'
    sampling_strategy : str | float | dict
        "auto" (balance ke minority) atau float, atau dict {class: count}
    cat_indices : list of int, optional
        Index kolom kategorik — wajib untuk SMOTENC
    k_neighbors : int
        Neighbors untuk SMOTE (default 5)
    random_state : int
        Seed (default 42)
    """

    def __init__(
        self,
        strategy: str = "smotetomek",
        sampling_strategy="auto",
        cat_indices: Optional[List[int]] = None,
        k_neighbors: int = 5,
        random_state: int = 42,
    ):
        self.strategy = strategy
        self.sampling_strategy = sampling_strategy
        self.cat_indices = cat_indices
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def get_class_weights(self, y: pd.Series) -> dict:
        """Hitung class weights untuk parameter model (mis. class_weight=balanced).

        Example
        -------
        >>> handler = ExpertImbalancedHandler()
        >>> cw = handler.get_class_weights(y_train)
        >>> # Gunakan: model = RandomForestClassifier(class_weight=cw)
        """
        classes = np.unique(y)
        weights = compute_class_weight("balanced", classes=classes, y=y)
        cw = dict(zip(classes, weights))
        logger.info(f"Class weights: {cw}")
        return cw

    def fit_resample(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply resampling. HANYA pada training data!

        Example
        -------
        >>> handler = ExpertImbalancedHandler(strategy="smotetomek")
        >>> X_res, y_res = handler.fit_resample(X_train, y_train)
        >>> # Pakai X_res, y_res untuk training — JANGAN untuk val/test
        """
        self._log_dist("SEBELUM", y)
        feat_names = X.columns.tolist()

        def _smote():
            return SMOTE(
                sampling_strategy=self.sampling_strategy,
                k_neighbors=self.k_neighbors,
                random_state=self.random_state,
                n_jobs=-1,
            )

        resampler_map = {
            "smote": _smote,
            "smotenc": lambda: SMOTENC(
                categorical_features=self.cat_indices or [],
                sampling_strategy=self.sampling_strategy,
                k_neighbors=self.k_neighbors,
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "adasyn": lambda: ADASYN(
                sampling_strategy=self.sampling_strategy,
                n_neighbors=self.k_neighbors,
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "smotetomek": lambda: SMOTETomek(
                smote=_smote(),
                tomek=TomekLinks(n_jobs=-1),
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "smoteenn": lambda: SMOTEENN(
                smote=_smote(),
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "undersample": lambda: RandomUnderSampler(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
            ),
            "oversample": lambda: RandomOverSampler(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
            ),
        }

        if self.strategy == "class_weight":
            logger.info(
                "class_weight mode: data tidak diubah. "
                "Pass class_weights ke parameter model!"
            )
            self.class_weights_ = self.get_class_weights(y)
            return X.copy(), y.copy()

        if self.strategy not in resampler_map:
            raise ValueError(f"Strategy tidak dikenali: {self.strategy}")

        resampler = resampler_map[self.strategy]()
        X_res, y_res = resampler.fit_resample(X, y)

        X_res = pd.DataFrame(X_res, columns=feat_names)
        y_res = pd.Series(y_res, name=y.name)

        self._log_dist("SETELAH", y_res)
        return X_res, y_res

    def plot_distribution(self, y_before: pd.Series, y_after: pd.Series) -> None:
        """Bar chart distribusi kelas sebelum vs sesudah resampling.

        Example
        -------
        >>> handler = ExpertImbalancedHandler(strategy="smote")
        >>> X_res, y_res = handler.fit_resample(X_train, y_train)
        >>> handler.plot_distribution(y_train, y_res)
        """
        from pathlib import Path
        Path("reports").mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, y, title in zip(axes, [y_before, y_after], ["SEBELUM", "SETELAH"]):
            vc = y.value_counts().sort_index()
            clr = plt.cm.Set2(np.linspace(0, 1, len(vc)))
            bars = ax.bar(vc.index.astype(str), vc.values, color=clr, edgecolor="black")
            for bar, val in zip(bars, vc.values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vc) * 0.01,
                    f"{val:,}",
                    ha="center",
                    fontsize=9,
                )
            ax.set_title(f"{title} (n={len(y):,})")
            ax.set_xlabel("Class")

        plt.suptitle("Class Distribution: Before vs After Resampling", fontsize=14)
        plt.tight_layout()
        plt.savefig("reports/class_distribution.png", dpi=150, bbox_inches="tight")
        plt.show()

    def _log_dist(self, label: str, y: pd.Series) -> None:
        vc = y.value_counts().sort_index()
        logger.info(f"{label}:")
        for cls, cnt in vc.items():
            logger.info(f"  Class {cls}: {cnt:,} ({cnt / len(y) * 100:.1f}%)")
        if vc.min() > 0:
            logger.info(f"  Imbalance ratio: {vc.max() / vc.min():.1f}x")

    @staticmethod
    def find_optimal_threshold(
        y_true,
        y_prob,
        metric: str = "f1",
    ) -> Tuple[float, float]:
        """
        Cari threshold optimal via PR curve.
        Default 0.5 sering tidak optimal untuk imbalanced data.

        Parameters
        ----------
        metric : str
            'f1' | 'precision' | 'recall' | 'gmean'

        Example
        -------
        >>> y_prob = model.predict_proba(X_val)[:, 1]
        >>> thr, score = ExpertImbalancedHandler.find_optimal_threshold(y_val, y_prob, metric="f1")
        >>> y_pred = (y_prob >= thr).astype(int)
        """
        prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
        best_thr, best_score = 0.5, 0.0

        for thr, p, r in zip(thresholds, prec[:-1], rec[:-1]):
            score = {
                "f1": 2 * p * r / (p + r + 1e-8),
                "precision": p,
                "recall": r,
                "gmean": (p * r) ** 0.5,
            }.get(metric, 0.0)
            if score > best_score:
                best_score, best_thr = score, thr

        logger.info(f"Optimal threshold ({metric}): {best_thr:.4f} | score: {best_score:.4f}")
        return best_thr, best_score


__all__ = ["ExpertImbalancedHandler"]
