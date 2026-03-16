# src/mltools/modeling/tuner.py
# Perubahan: tambah MLConfig support + absolute imports

from __future__ import annotations

import optuna
from loguru import logger
from sklearn.model_selection import cross_val_score

from mltools.shared.config     import MLConfig
from mltools.shared.exceptions import TuningError

optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaTuner:
    """
    Optuna-based hyperparameter tuner.

    Parameters
    ----------
    model_class  : class model sklearn-compatible
    param_space  : dict {param_name: lambda trial: trial.suggest_*(…)}
    n_trials     : jumlah trials
    scoring      : metrik sklearn
    cv           : jumlah fold CV
    random_state : seed
    """

    def __init__(
        self,
        model_class,
        param_space : dict,
        n_trials    : int = 50,
        scoring     : str = "roc_auc",
        cv          : int = 5,
        timeout     : int = None,
        random_state: int = 42,
    ):
        self.model_class  = model_class
        self.param_space  = param_space
        self.n_trials     = n_trials
        self.scoring      = scoring
        self.cv           = cv
        self.timeout      = timeout
        self.random_state = random_state
        self.study_       = None
        self.best_params_ = None
        self.best_score_  = None

    @classmethod
    def from_config(
        cls,
        model_class,
        param_space: dict,
        config     : MLConfig,
    ) -> "OptunaTuner":
        """Buat OptunaTuner dari MLConfig."""
        return cls(
            model_class  = model_class,
            param_space  = param_space,
            n_trials     = config.modeling.tuning.n_trials,
            scoring      = config.modeling.metric,
            cv           = config.modeling.n_cv_folds,
            timeout      = config.modeling.tuning.timeout,
            random_state = config.project.random_state,
        )

    def _objective(self, trial, X, y) -> float:
        params = {k: v(trial) for k, v in self.param_space.items()}
        try:
            model  = self.model_class(**params)
            scores = cross_val_score(
                model, X, y,
                cv      = self.cv,
                scoring = self.scoring,
                n_jobs  = 1,
            )
            return scores.mean()
        except Exception as e:
            raise optuna.TrialPruned(f"Trial failed: {e}")

    def tune(self, X, y) -> tuple:
        """
        Jalankan tuning. Return (best_params, best_score).
        """
        logger.info(
            f"Optuna tuning {self.model_class.__name__} "
            f"({self.n_trials} trials, scoring={self.scoring})..."
        )

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study   = optuna.create_study(
            direction = "maximize",
            sampler   = sampler,
        )

        try:
            study.optimize(
                lambda trial: self._objective(trial, X, y),
                n_trials          = self.n_trials,
                timeout           = self.timeout,
                show_progress_bar = True,
            )
        except Exception as e:
            raise TuningError(f"Optuna study gagal: {e}")

        self.study_       = study
        self.best_params_ = study.best_params
        self.best_score_  = study.best_value

        logger.success(f"Best CV score : {self.best_score_:.4f}")
        logger.info(f"Best params   : {self.best_params_}")

        return self.best_params_, self.best_score_