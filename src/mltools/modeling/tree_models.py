# src/mltools/modeling/tree_models.py
# Perubahan: tambah import exceptions + absolute imports

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from mltools.shared.exceptions import ModelNotFittedError


class ExpertDecisionTree:
    """
    Decision Tree dengan automatic cost-complexity pruning via CV.
    """

    def __init__(
        self,
        min_samples_leaf: int  = 20,
        class_weight    : str  = "balanced",
        random_state    : int  = 42,
        auto_prune      : bool = True,
        cv              : int  = 5,
    ):
        self.min_samples_leaf = min_samples_leaf
        self.class_weight     = class_weight
        self.random_state     = random_state
        self.auto_prune       = auto_prune
        self.cv               = cv
        self.feature_names    = None
        self.model            = None
        self.best_alpha_      = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ExpertDecisionTree":
        self.feature_names = X.columns.tolist()

        if self.auto_prune:
            logger.info("Mencari alpha optimal untuk pruning...")
            base = DecisionTreeClassifier(
                random_state = self.random_state,
                class_weight = self.class_weight,
            )
            path   = base.cost_complexity_pruning_path(X, y)
            alphas = path.ccp_alphas[::max(1, len(path.ccp_alphas) // 20)]

            cv_scores = []
            for alpha in alphas:
                tree = DecisionTreeClassifier(
                    ccp_alpha    = alpha,
                    class_weight = self.class_weight,
                    random_state = self.random_state,
                )
                scores = cross_val_score(
                    tree, X, y, cv=self.cv, scoring="roc_auc"
                )
                cv_scores.append(scores.mean())

            best_alpha   = alphas[np.argmax(cv_scores)]
            self.best_alpha_ = best_alpha
            logger.success(f"Best ccp_alpha: {best_alpha:.6f}")

            self.model = DecisionTreeClassifier(
                ccp_alpha        = best_alpha,
                min_samples_leaf = self.min_samples_leaf,
                class_weight     = self.class_weight,
                random_state     = self.random_state,
            )
        else:
            self.model = DecisionTreeClassifier(
                min_samples_leaf = self.min_samples_leaf,
                class_weight     = self.class_weight,
                random_state     = self.random_state,
            )

        self.model.fit(X, y)
        logger.success(
            f"Decision Tree: "
            f"depth={self.model.get_depth()}, "
            f"leaves={self.model.get_n_leaves()}"
        )
        return self

    def visualize(self, class_names=None, max_depth_display: int = 4):
        if self.model is None:
            raise ModelNotFittedError("decision_tree")
        fig, ax = plt.subplots(figsize=(25, 12))
        plot_tree(
            self.model,
            feature_names    = self.feature_names,
            class_names      = class_names,
            filled           = True,
            rounded          = True,
            fontsize         = 9,
            max_depth        = max_depth_display,
            ax               = ax,
        )
        plt.title(f"Decision Tree (max depth shown: {max_depth_display})")
        plt.tight_layout()
        plt.savefig(
            "reports/decision_tree.png",
            dpi=150, bbox_inches="tight"
        )
        plt.show()

    def print_rules(self, max_depth: int = 4) -> str:
        if self.model is None:
            raise ModelNotFittedError("decision_tree")
        rules = export_text(
            self.model,
            feature_names = self.feature_names,
            max_depth     = max_depth,
        )
        print(rules)
        return rules

    def predict(self, X):
        if self.model is None:
            raise ModelNotFittedError("decision_tree")
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.model is None:
            raise ModelNotFittedError("decision_tree")
        return self.model.predict_proba(X)


class ExpertRandomForest:
    """
    Random Forest dengan optional hyperparameter tuning
    dan feature importance plotting.
    """

    def __init__(
        self,
        task        : str  = "classification",
        n_estimators: int  = 200,
        tune        : bool = False,
        cv          : int  = 5,
        n_iter      : int  = 20,
        random_state: int  = 42,
        n_jobs      : int  = -1,
    ):
        self.task         = task
        self.n_estimators = n_estimators
        self.tune         = tune
        self.cv           = cv
        self.n_iter       = n_iter
        self.random_state = random_state
        self.n_jobs       = n_jobs
        self.model_       = None
        self.best_params_ = None
        self.cv_score_    = None
        self.val_score_   = None
        self.feature_importance_ = None

    def _get_model_class(self):
        return (RandomForestClassifier if self.task == "classification"
                else RandomForestRegressor)

    def _get_scoring(self, y) -> str:
        if self.task == "regression":
            return "neg_root_mean_squared_error"
        return "roc_auc" if y.nunique() == 2 else "f1_macro"

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val  : pd.DataFrame = None,
        y_val  : pd.Series    = None,
    ) -> "ExpertRandomForest":
        ModelClass = self._get_model_class()
        scoring    = self._get_scoring(y_train)

        if self.tune:
            from sklearn.model_selection import RandomizedSearchCV
            logger.info("Tuning Random Forest hyperparameters...")
            param_dist = {
                "n_estimators"   : [100, 200, 300, 500, 700],
                "max_depth"      : [None, 5, 10, 15, 20, 30],
                "min_samples_leaf": [1, 2, 4, 8, 16],
                "min_samples_split": [2, 5, 10, 20],
                "max_features"   : ["sqrt", "log2", 0.3, 0.5, 0.7],
                "bootstrap"      : [True, False],
                "max_samples"    : [0.6, 0.7, 0.8, 0.9, None],
            }
            base   = ModelClass(
                oob_score    = True,
                random_state = self.random_state,
                n_jobs       = self.n_jobs,
            )
            search = RandomizedSearchCV(
                base,
                param_distributions = param_dist,
                n_iter              = self.n_iter,
                cv                  = self.cv,
                scoring             = scoring,
                n_jobs              = self.n_jobs,
                random_state        = self.random_state,
                verbose             = 1,
                refit               = True,
            )
            search.fit(X_train, y_train)
            self.model_       = search.best_estimator_
            self.best_params_ = search.best_params_
            self.cv_score_    = search.best_score_
            logger.success(f"Best params: {self.best_params_}")
            logger.success(f"CV Score ({scoring}): {self.cv_score_:.4f}")

        else:
            self.model_ = ModelClass(
                n_estimators     = self.n_estimators,
                max_depth        = None,
                min_samples_leaf = 2,
                max_features     = "sqrt",
                oob_score        = True,
                bootstrap        = True,
                n_jobs           = self.n_jobs,
                random_state     = self.random_state,
            )
            self.model_.fit(X_train, y_train)
            cv_res = cross_validate(
                self.model_, X_train, y_train,
                cv=self.cv, scoring=scoring, n_jobs=self.n_jobs,
            )
            self.cv_score_ = cv_res["test_score"].mean()

        if hasattr(self.model_, "oob_score_") and self.model_.oob_score_:
            logger.info(f"OOB Score: {self.model_.oob_score_:.4f}")

        if X_val is not None and y_val is not None:
            self._evaluate_val(X_val, y_val)

        self.feature_importance_ = pd.Series(
            self.model_.feature_importances_,
            index = X_train.columns,
        ).sort_values(ascending=False)

        logger.success(
            f"Random Forest selesai! CV {scoring}: {self.cv_score_:.4f}"
        )
        return self

    def _evaluate_val(self, X_val, y_val):
        from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error
        if self.task == "classification":
            proba = self.model_.predict_proba(X_val)
            if y_val.nunique() == 2:
                self.val_score_ = roc_auc_score(y_val, proba[:, 1])
                logger.info(f"Val ROC-AUC: {self.val_score_:.4f}")
            else:
                y_pred = self.model_.predict(X_val)
                self.val_score_ = f1_score(
                    y_val, y_pred, average="macro"
                )
                logger.info(f"Val F1-Macro: {self.val_score_:.4f}")
        else:
            y_pred = self.model_.predict(X_val)
            self.val_score_ = float(
                np.sqrt(mean_squared_error(y_val, y_pred))
            )
            logger.info(f"Val RMSE: {self.val_score_:.4f}")

    def plot_feature_importance(self, top_n: int = 25):
        if self.feature_importance_ is None:
            raise ModelNotFittedError("random_forest")
        imp = self.feature_importance_.head(top_n)
        fig, ax = plt.subplots(
            figsize=(10, max(6, top_n * 0.35))
        )
        colors = plt.cm.RdYlGn(
            np.linspace(0.3, 0.9, len(imp))[::-1]
        )
        ax.barh(
            imp.index[::-1], imp.values[::-1],
            color=colors[::-1], edgecolor="white",
        )
        ax.set_title(
            f"Random Forest Feature Importance (Top {top_n})\n"
            "MDI — Mean Decrease in Impurity",
            fontsize=12,
        )
        ax.set_xlabel("Importance Score")
        plt.tight_layout()
        plt.savefig(
            "reports/rf_importance.png",
            dpi=150, bbox_inches="tight",
        )
        plt.show()

    def predict(self, X):
        if self.model_ is None:
            raise ModelNotFittedError("random_forest")
        return self.model_.predict(X)

    def predict_proba(self, X):
        if self.model_ is None:
            raise ModelNotFittedError("random_forest")
        return self.model_.predict_proba(X)