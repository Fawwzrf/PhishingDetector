"""
Data splitting — production-grade dengan leakage prevention.
"""
from pathlib import Path
from typing import Generator, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from mltools.shared.exceptions import SplittingError, DataLeakageError
from scipy.stats import ks_2samp
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)


class ExpertDataSplitter:
    """
    Data splitting production-grade dengan leakage prevention.

    Parameters
    ----------
    task : str
        'classification' | 'regression'
    test_size : float
        Fraksi test set (default 0.15)
    val_size : float
        Fraksi val dari total data (default 0.15)
    strategy : str
        'holdout' | 'kfold' | 'timeseries' | 'group'
    n_splits : int
        Jumlah fold untuk CV
    group_col : str, optional
        Kolom grup untuk GroupKFold / StratifiedGroupKFold
    time_col : str, optional
        Kolom waktu untuk sorting (TimeSeriesSplit)
    random_state : int
        Seed
    """

    def __init__(
        self,
        task: str = "classification",
        test_size: float = 0.15,
        val_size: float = 0.15,
        strategy: str = "holdout",
        n_splits: int = 5,
        group_col: Optional[str] = None,
        time_col: Optional[str] = None,
        random_state: int = 42,
    ):
        self.task = task
        self.test_size = test_size
        self.val_size = val_size
        self.strategy = strategy
        self.n_splits = n_splits
        self.group_col = group_col
        self.time_col = time_col
        self.random_state = random_state

    def split_holdout(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Train / Val / Test — stratified untuk classification.

        Example
        -------
        >>> splitter = ExpertDataSplitter(task="classification", test_size=0.15, val_size=0.15)
        >>> X_train, X_val, X_test, y_train, y_val, y_test = splitter.split_holdout(X, y)
        """
        strat = y if self.task == "classification" else None

        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=strat,
            random_state=self.random_state,
        )

        val_rel = self.val_size / (1 - self.test_size)
        strat2 = y_tmp if self.task == "classification" else None

        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp,
            test_size=val_rel,
            stratify=strat2,
            random_state=self.random_state,
        )

        self._log([("Train", y_train), ("Val", y_val), ("Test", y_test)])
        return X_train, X_val, X_test, y_train, y_val, y_test

    def split_kfold(self, X: pd.DataFrame, y: pd.Series) -> Generator:
        """Stratified K-Fold — ideal untuk dataset kecil–medium.

        Example
        -------
        >>> splitter = ExpertDataSplitter(task="classification", n_splits=5)
        >>> for fold, X_tr, X_va, y_tr, y_va in splitter.split_kfold(X, y):
        ...     model.fit(X_tr, y_tr)
        ...     score = model.score(X_va, y_va)
        """
        CV = StratifiedKFold if self.task == "classification" else KFold
        kf = CV(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        logger.info(f"K-Fold: {self.n_splits} folds")
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
            yield fold, X.iloc[tr_idx], X.iloc[va_idx], y.iloc[tr_idx], y.iloc[va_idx]

    def split_timeseries(self, X: pd.DataFrame, y: pd.Series) -> Generator:
        """
        TimeSeriesSplit — WAJIB untuk data temporal.
        Menjaga urutan; future tidak bocor ke past.

        Example
        -------
        >>> splitter = ExpertDataSplitter(task="regression", n_splits=5, time_col="date")
        >>> for fold, X_tr, X_va, y_tr, y_va in splitter.split_timeseries(X, y):
        ...     model.fit(X_tr, y_tr)
        """
        if self.time_col and self.time_col in X.columns:
            order = X[self.time_col].argsort()
            X = X.iloc[order].reset_index(drop=True)
            y = y.iloc[order].reset_index(drop=True)

        tscv = TimeSeriesSplit(n_splits=self.n_splits, gap=0)
        logger.info(f"TimeSeriesSplit: {self.n_splits} folds (expanding window)")
        for fold, (tr_idx, va_idx) in enumerate(tscv.split(X)):
            yield fold, X.iloc[tr_idx], X.iloc[va_idx], y.iloc[tr_idx], y.iloc[va_idx]

    def split_group(self, X: pd.DataFrame, y: pd.Series) -> Generator:
        """
        StratifiedGroupKFold — untuk data dengan entity (user/customer).
        Satu entity hanya ada di satu split (no leakage).

        Example
        -------
        >>> splitter = ExpertDataSplitter(task="classification", n_splits=5,
        ...                              group_col="customer_id")
        >>> for fold, X_tr, X_va, y_tr, y_va in splitter.split_group(X, y):
        ...     model.fit(X_tr, y_tr)
        """
        if not self.group_col:
            raise SplittingError("group_col wajib untuk group split!")
        groups = X[self.group_col].values

        CV = StratifiedGroupKFold if self.task == "classification" else GroupKFold
        kf = CV(n_splits=self.n_splits)
        logger.info(f"GroupKFold: {self.n_splits} folds | {len(np.unique(groups))} groups")

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y, groups)):
            tr_groups = set(groups[tr_idx])
            va_groups = set(groups[va_idx])
            overlap = tr_groups & va_groups
            if overlap:
                raise DataLeakageError(
    f"Group overlap di fold {fold + 1}",
    details={"overlap_count": len(overlap)}
)
            yield fold, X.iloc[tr_idx], X.iloc[va_idx], y.iloc[tr_idx], y.iloc[va_idx]

    def visualize_splits(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Visualisasi distribusi sampel per fold.

        Example
        -------
        >>> splitter = ExpertDataSplitter(n_splits=5)
        >>> splitter.visualize_splits(X, y)   # disimpan ke reports/cv_splits.png
        """
        Path("reports").mkdir(parents=True, exist_ok=True)

        if self.strategy == "timeseries":
            splits = list(TimeSeriesSplit(n_splits=self.n_splits).split(X))
        elif self.task == "classification":
            splits = list(
                StratifiedKFold(
                    n_splits=self.n_splits,
                    shuffle=True,
                    random_state=self.random_state,
                ).split(X, y)
            )
        else:
            splits = list(
                KFold(
                    n_splits=self.n_splits,
                    shuffle=True,
                    random_state=self.random_state,
                ).split(X)
            )

        fig, ax = plt.subplots(figsize=(12, max(4, self.n_splits * 0.8)))
        for fold, (tr_idx, va_idx) in enumerate(splits):
            ax.scatter(tr_idx, [fold] * len(tr_idx), c="steelblue", s=1, label="Train" if fold == 0 else "")
            ax.scatter(va_idx, [fold] * len(va_idx), c="salmon", s=1, label="Validation" if fold == 0 else "")

        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Fold")
        ax.set_title(f"CV Split Visualization ({self.strategy})")
        ax.legend(markerscale=5)
        ax.set_yticks(range(self.n_splits))
        ax.set_yticklabels([f"Fold {i + 1}" for i in range(self.n_splits)])
        plt.tight_layout()
        plt.savefig("reports/cv_splits.png", dpi=150, bbox_inches="tight")
        plt.show()

    def _log(self, splits: list) -> None:
        total = sum(len(y) for _, y in splits)
        logger.info("Data Split Summary:")
        for name, y in splits:
            pct = len(y) / total * 100
            logger.info(f"  {name:8s}: {len(y):7,} ({pct:.1f}%)")


def check_data_leakage(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> bool:
    """
    Deteksi potensi leakage antara train dan test.
    Cek: index overlap, duplicate rows, KS distribusi test.

    Example
    -------
    >>> has_leak = check_data_leakage(X_train, X_test)
    >>> if has_leak:
    ...     raise RuntimeError("Leakage terdeteksi!")
    """
    logger.info("Checking for data leakage...")
    leakage = False

    overlap = set(X_train.index) & set(X_test.index)
    if overlap:
        logger.error(f"LEAKAGE! Index overlap: {len(overlap)} rows!")
        leakage = True
    else:
        logger.info("  OK: Tidak ada index overlap")

    merged = pd.merge(
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        how="inner",
        on=X_train.columns.tolist(),
    )
    if len(merged) > 0:
        logger.error(f"LEAKAGE! {len(merged)} duplicate rows antara train-test!")
        leakage = True
    else:
        logger.info("  OK: Tidak ada duplikat rows")

    num_cols = X_train.select_dtypes(include=np.number).columns
    suspicious = []
    for c in num_cols:
        if c not in X_test.columns:
            continue
        tr_vals = X_train[c].dropna()
        te_vals = X_test[c].dropna()
        if len(tr_vals) > 0 and len(te_vals) > 0:
            _, pval = ks_2samp(tr_vals, te_vals)
            if pval > 0.999:
                suspicious.append(c)

    if suspicious:
        logger.warning(f"  WARN: Distribusi identik (cek manual): {suspicious}")
    else:
        logger.info("  OK: Distribusi train-test terlihat wajar")

    if not leakage:
        logger.success("CLEAR: Tidak ada indikasi leakage")
    return leakage


__all__ = ["ExpertDataSplitter", "check_data_leakage"]
