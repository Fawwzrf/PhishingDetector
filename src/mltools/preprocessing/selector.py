"""
Feature selection — 4-layer filter, production-grade.
Layer 1: Quasi-constant | Layer 2: Correlation | Layer 3: Model-based | Layer 4: RFECV
"""
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from loguru import logger
from mltools.shared.exceptions import FeatureSelectionError
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    RFECV,
    VarianceThreshold,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.model_selection import KFold, StratifiedKFold

import warnings
warnings.filterwarnings("ignore", module="shap")


class ExpertFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Feature selection production-grade — 4 layer filter.

    Layer 1: Quasi-constant (VarianceThreshold)
    Layer 2: High correlation (hapus redundansi)
    Layer 3: Model-based importance (SHAP / Tree / Lasso / MutualInfo)
    Layer 4: RFECV optional (jumlah fitur optimal via CV)

    Parameters
    ----------
    task : str
        'classification' | 'regression'
    variance_thr : float
        Threshold variansi (default 0.01)
    corr_threshold : float
        Threshold korelasi untuk drop (default 0.95)
    importance_method : str
        'shap' | 'tree' | 'lasso' | 'mutual_info'
    n_features : int, optional
        Jumlah fitur final. None = gunakan top_n_pct
    top_n_pct : float
        Fraksi top features jika n_features=None
    use_rfecv : bool
        RFECV fine-tune (lambat tapi akurat)
    cv_folds : int
        Jumlah folds untuk RFECV
    random_state : int
        Seed
    """

    def __init__(
        self,
        task: str = "classification",
        variance_thr: float = 0.01,
        corr_threshold: float = 0.95,
        importance_method: str = "shap",
        n_features: Optional[int] = None,
        top_n_pct: float = 0.5,
        use_rfecv: bool = False,
        cv_folds: int = 5,
        random_state: int = 42,
    ):
        self.task = task
        self.variance_thr = variance_thr
        self.corr_threshold = corr_threshold
        self.importance_method = importance_method
        self.n_features = n_features
        self.top_n_pct = top_n_pct
        self.use_rfecv = use_rfecv
        self.cv_folds = cv_folds
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ExpertFeatureSelector":
        """Jalankan 4-layer feature selection pipeline.

        Example
        -------
        >>> selector = ExpertFeatureSelector(
        ...     task="classification",
        ...     importance_method="shap",
        ...     n_features=30,
        ... )
        >>> selector.fit(X_train, y_train)
        """
        self.feature_names_in_ = X.columns.tolist()
        self.selection_log_ = {}
        current_cols = X.columns.tolist()

        logger.info(f"Feature selection mulai | Fitur awal: {len(current_cols)}")

        # Layer 1: Quasi-constant
        var_sel = VarianceThreshold(threshold=self.variance_thr)
        var_sel.fit(X[current_cols])
        mask = var_sel.get_support()
        dropped = [c for c, m in zip(current_cols, mask) if not m]
        current_cols = [c for c in current_cols if c not in dropped]
        self.selection_log_["layer1_dropped"] = dropped
        logger.info(f"[L1] Quasi-constant: {len(dropped)} dropped | Sisa: {len(current_cols)}")

        # Layer 2: Correlation filter
        num_cols = X[current_cols].select_dtypes(include=np.number).columns.tolist()
        if not num_cols:
            logger.warning("Tidak ada kolom numerik untuk korelasi — skip L2")
        else:
            corr = X[num_cols].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            corr_target = X[num_cols].corrwith(y).abs()

            dropped_corr = []
            for col in upper.columns:
                hi_feats = upper.index[upper[col] > self.corr_threshold].tolist()
                for feat in hi_feats:
                    if feat in current_cols and col in current_cols:
                        to_drop = feat if corr_target.get(feat, 0) < corr_target.get(col, 0) else col
                        dropped_corr.append(to_drop)
            dropped_corr = list(set(dropped_corr))
            current_cols = [c for c in current_cols if c not in dropped_corr]
            self.selection_log_["layer2_dropped"] = dropped_corr
            logger.info(f"[L2] High-corr: {len(dropped_corr)} dropped | Sisa: {len(current_cols)}")

        # Layer 3: Model-based importance
        X_curr = X[current_cols]
        dispatch = {
            "shap": self._fit_shap,
            "tree": self._fit_tree,
            "lasso": self._fit_lasso,
            "mutual_info": self._fit_mutual_info,
        }
        if self.importance_method not in dispatch:
            raise FeatureSelectionError(
    f"importance_method tidak dikenali: {self.importance_method}",
    details={"valid": ["shap", "tree", "lasso", "mutual_info"]}
)

        if isinstance(y, pd.DataFrame):
            y_array = y.iloc[:, 0]
        else:
            y_array = y

        scores = dispatch[self.importance_method](X_curr, y_array)
        imp_df = pd.DataFrame({"feature": current_cols, "importance": scores}).sort_values(
            "importance", ascending=False
        )
        self.importance_ranking_ = imp_df

        n_select = self.n_features if self.n_features else max(1, int(len(current_cols) * self.top_n_pct))
        n_select = min(n_select, len(current_cols))
        selected = imp_df.head(n_select)["feature"].tolist()
        dropped_imp = [c for c in current_cols if c not in selected]
        current_cols = selected
        self.selection_log_["layer3_dropped"] = dropped_imp
        logger.info(f"[L3] Kept top {n_select} | Dropped: {len(dropped_imp)}")

        # Layer 4: RFECV
        if self.use_rfecv and len(current_cols) > 5:
            current_cols = self._fit_rfecv(X[current_cols], y, current_cols)
            logger.info(f"[L4] RFECV optimal: {len(current_cols)} features")

        self.selected_features_ = current_cols
        logger.success(f"SELESAI: {len(self.feature_names_in_)} => {len(current_cols)} fitur")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Kembalikan DataFrame hanya dengan fitur yang dipilih.

        Example
        -------
        >>> X_train_sel = selector.transform(X_train)
        >>> X_val_sel   = selector.transform(X_val)
        >>> X_test_sel  = selector.transform(X_test)
        """
        return X[[c for c in self.selected_features_ if c in X.columns]].copy()

    def _fit_shap(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        M = RandomForestClassifier if self.task == "classification" else RandomForestRegressor
        m = M(n_estimators=200, max_depth=6, random_state=self.random_state, n_jobs=-1)
        m.fit(X, y)
        sv = shap.TreeExplainer(m).shap_values(X)
        if isinstance(sv, list):
            sv_arr = np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)
        else:
            sv_arr = np.abs(sv).mean(axis=0)
            if sv_arr.ndim > 1:
                sv_arr = sv_arr.mean(axis=1)
        
        return np.ravel(sv_arr)

    def _fit_tree(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        M = RandomForestClassifier if self.task == "classification" else RandomForestRegressor
        m = M(n_estimators=200, random_state=self.random_state, n_jobs=-1)
        m.fit(X, y)
        importances = m.feature_importances_
        return np.ravel(importances)

    def _fit_lasso(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        if self.task == "classification":
            m = LogisticRegressionCV(
                cv=5,
                penalty="l1",
                solver="saga",
                random_state=self.random_state,
                max_iter=1000,
                n_jobs=-1,
            )
        else:
            m = LassoCV(cv=5, random_state=self.random_state, n_jobs=-1)
        m.fit(X, y)
        coef = m.coef_.ravel() if hasattr(m.coef_, "ravel") else m.coef_
        return np.abs(coef)

    def _fit_mutual_info(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        fn = mutual_info_classif if self.task == "classification" else mutual_info_regression
        return fn(X, y, random_state=self.random_state)

    def _fit_rfecv(self, X: pd.DataFrame, y: pd.Series, cols: List[str]) -> List[str]:
        M = RandomForestClassifier if self.task == "classification" else RandomForestRegressor
        est = M(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        cv = (
            StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            if self.task == "classification"
            else KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        )
        scoring = "roc_auc" if self.task == "classification" else "r2"
        rfecv = RFECV(
            estimator=est,
            step=1,
            cv=cv,
            scoring=scoring,
            min_features_to_select=min(5, X.shape[1] - 1),
            n_jobs=-1,
        )
        rfecv.fit(X, y)
        self.rfecv_n_features_ = rfecv.n_features_
        return [c for c, s in zip(cols, rfecv.support_) if s]

    def plot_importance(self, top_n: int = 30) -> None:
        """Bar chart importance — biru=selected, merah=dropped.

        Example
        -------
        >>> selector.fit(X_train, y_train)
        >>> selector.plot_importance(top_n=20)   # simpan ke reports/feature_importance.png
        """
        Path("reports").mkdir(parents=True, exist_ok=True)

        df = self.importance_ranking_.head(top_n)
        clrs = ["steelblue" if f in self.selected_features_ else "lightcoral" for f in df["feature"]]

        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
        ax.barh(df["feature"], df["importance"], color=clrs)
        ax.invert_yaxis()
        ax.set_xlabel("Importance Score")
        ax.set_title(
            f"Feature Importance ({self.importance_method.upper()})\n"
            "Biru=Selected | Merah=Dropped"
        )
        plt.tight_layout()
        plt.savefig("reports/feature_importance.png", dpi=150, bbox_inches="tight")
        plt.show()

    def get_selection_report(self) -> pd.DataFrame:
        """Ringkasan log seleksi.

        Example
        -------
        >>> df_importance = selector.get_selection_report()
        >>> print(df_importance.head(10))
        """
        logger.info("=" * 55)
        for i, k in enumerate(["layer1_dropped", "layer2_dropped", "layer3_dropped"], 1):
            n = len(self.selection_log_.get(k, []))
            logger.info(f"  Layer {i} dropped: {n}")
        logger.info(f"  Final features: {len(self.selected_features_)}")
        logger.info("=" * 55)
        return self.importance_ranking_


def compute_permutation_importance(
    model,
    X_val: pd.DataFrame,
    y_val,
    task: str = "classification",
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Permutation importance pada val set — tidak bias ke train.

    Example
    -------
    >>> perm = compute_permutation_importance(model, X_val, y_val, task="classification")
    >>> print(perm.head(10))   # fitur diurutkan dari importance tertinggi
    """
    scoring = "roc_auc" if task == "classification" else "r2"
    result = permutation_importance(
        model, X_val, y_val, n_repeats=n_repeats, random_state=42, scoring=scoring, n_jobs=-1
    )
    return pd.DataFrame({
        "feature": X_val.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)


__all__ = ["ExpertFeatureSelector", "compute_permutation_importance"]
