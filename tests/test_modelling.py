# tests/test_modeling.py
# Unit test untuk modeling layer

import pytest
import numpy as np
import pandas as pd

from mltools.modeling.baseline        import BaselineModel
from mltools.modeling.linear_models   import ExpertLogisticRegression
from mltools.modeling.tree_models     import ExpertRandomForest
from mltools.modeling.boosting_models import ExpertLightGBM
from mltools.modeling.evaluator       import ModelEvaluator
from mltools.modeling.cross_validator import CrossValidator
from mltools.modeling.pipeline        import ModelingPipeline
from mltools.shared.exceptions        import ModelNotFittedError, ModelingError


# ══════════════════════════════════════════════════════════════
# BASELINE TESTS
# ══════════════════════════════════════════════════════════════

class TestBaseline:

    def test_fit_and_predict(self, data_split):
        """Baseline berhasil fit dan predict."""
        baseline = BaselineModel(task="classification")
        baseline.fit(data_split.X_train, data_split.y_train)
        preds = baseline.predict(data_split.X_val)
        assert len(preds) == len(data_split.X_val)

    def test_evaluate_from_split(self, data_split):
        """evaluate_from_split() return dict dengan metrik yang benar."""
        baseline = BaselineModel(task="classification")
        scores   = baseline.evaluate_from_split(data_split)

        assert "accuracy" in scores
        assert "f1"       in scores
        assert "roc_auc"  in scores
        assert scores["roc_auc"] == pytest.approx(0.5, abs=0.05)

    def test_is_better_passes(self, data_split):
        """is_better() tidak raise jika model lebih baik."""
        baseline = BaselineModel(task="classification")
        baseline.evaluate_from_split(data_split)
        # Skor 0.9 harus lebih baik dari baseline ~0.5
        assert baseline.is_better(0.9, metric="roc_auc") == True

    def test_is_better_raises(self, data_split):
        """is_better() raise ModelingError jika model tidak lebih baik."""
        baseline = BaselineModel(task="classification")
        baseline.evaluate_from_split(data_split)
        with pytest.raises(ModelingError, match="tidak mengalahkan baseline"):
            baseline.is_better(0.3, metric="roc_auc")

    def test_predict_before_fit_raises(self, data_split):
        """predict() sebelum fit raise ModelingError."""
        baseline = BaselineModel()
        with pytest.raises(ModelingError):
            baseline.predict(data_split.X_val)


# ══════════════════════════════════════════════════════════════
# LINEAR MODEL TESTS
# ══════════════════════════════════════════════════════════════

class TestLogisticRegression:

    def test_fit_and_predict(self, data_split):
        """Logistic Regression fit dan predict berhasil."""
        model = ExpertLogisticRegression()
        model.fit(data_split.X_train, data_split.y_train)
        preds = model.predict(data_split.X_val)
        assert len(preds) == len(data_split.X_val)

    def test_predict_proba_shape(self, data_split):
        """predict_proba() return shape yang benar."""
        model = ExpertLogisticRegression()
        model.fit(data_split.X_train, data_split.y_train)
        proba = model.predict_proba(data_split.X_val)
        assert proba.shape == (len(data_split.X_val), 2)
        # Setiap baris harus sum ke 1
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_before_fit_raises(self, data_split):
        """predict() sebelum fit raise ModelNotFittedError."""
        model = ExpertLogisticRegression()
        with pytest.raises(ModelNotFittedError):
            model.predict(data_split.X_val)


# ══════════════════════════════════════════════════════════════
# RANDOM FOREST TESTS
# ══════════════════════════════════════════════════════════════

class TestRandomForest:

    def test_fit_sets_feature_importance(self, data_split):
        """fit() menghasilkan feature_importance_."""
        model = ExpertRandomForest(
            task         = "classification",
            n_estimators = 50,
        )
        model.fit(data_split.X_train, data_split.y_train)
        assert model.feature_importance_ is not None
        assert len(model.feature_importance_) == data_split.n_features

    def test_oob_score_computed(self, data_split):
        """OOB score dihitung saat fit."""
        model = ExpertRandomForest(
            task         = "classification",
            n_estimators = 50,
        )
        model.fit(data_split.X_train, data_split.y_train)
        assert hasattr(model.model_, "oob_score_")

    def test_val_score_with_split(self, data_split):
        """val_score_ di-set saat X_val diberikan."""
        model = ExpertRandomForest(
            task         = "classification",
            n_estimators = 50,
        )
        model.fit(
            data_split.X_train, data_split.y_train,
            data_split.X_val,   data_split.y_val,
        )
        assert model.val_score_ is not None
        assert 0.0 <= model.val_score_ <= 1.0


# ══════════════════════════════════════════════════════════════
# LIGHTGBM TESTS
# ══════════════════════════════════════════════════════════════

class TestLightGBM:

    def test_fit_and_predict(self, data_split):
        """LightGBM fit dan predict berhasil."""
        model = ExpertLightGBM(
            n_estimators          = 50,
            early_stopping_rounds = 10,
        )
        model.fit(
            data_split.X_train, data_split.y_train,
            data_split.X_val,   data_split.y_val,
        )
        preds = model.predict(data_split.X_val)
        assert len(preds) == len(data_split.X_val)

    def test_val_score_set_after_fit(self, data_split):
        """val_score_ di-set setelah fit."""
        model = ExpertLightGBM(
            n_estimators          = 50,
            early_stopping_rounds = 10,
        )
        model.fit(
            data_split.X_train, data_split.y_train,
            data_split.X_val,   data_split.y_val,
        )
        assert hasattr(model, "val_score_")
        assert 0.0 <= model.val_score_ <= 1.0

    def test_predict_before_fit_raises(self, data_split):
        """predict() sebelum fit raise ModelNotFittedError."""
        model = ExpertLightGBM()
        with pytest.raises(ModelNotFittedError):
            model.predict(data_split.X_val)

    def test_proba_sums_to_one(self, data_split):
        """predict_proba() setiap baris sum ke 1."""
        model = ExpertLightGBM(
            n_estimators          = 50,
            early_stopping_rounds = 10,
        )
        model.fit(
            data_split.X_train, data_split.y_train,
            data_split.X_val,   data_split.y_val,
        )
        proba = model.predict_proba(data_split.X_val)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# ══════════════════════════════════════════════════════════════
# EVALUATOR TESTS
# ══════════════════════════════════════════════════════════════

class TestEvaluator:

    def test_evaluate_returns_metrics(self, data_split):
        """evaluate() return dict dengan metrik yang benar."""
        model = ExpertLogisticRegression()
        model.fit(data_split.X_train, data_split.y_train)

        evaluator = ModelEvaluator(task="classification")
        y_pred    = model.predict(data_split.X_val)
        y_proba   = model.predict_proba(data_split.X_val)[:, 1]
        metrics   = evaluator.evaluate(
            data_split.y_val, y_pred, y_proba
        )

        assert "accuracy" in metrics
        assert "f1"       in metrics
        assert "roc_auc"  in metrics
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_evaluate_from_split(self, data_split):
        """evaluate_from_split() dari DataSplit langsung."""
        model = ExpertLogisticRegression()
        model.fit(data_split.X_train, data_split.y_train)

        evaluator = ModelEvaluator(task="classification")
        metrics   = evaluator.evaluate_from_split(
            model, data_split, split_name="val"
        )
        assert "accuracy" in metrics


# ══════════════════════════════════════════════════════════════
# CROSS VALIDATOR TESTS
# ══════════════════════════════════════════════════════════════

class TestCrossValidator:

    def test_score_returns_array(self, data_split):
        """score() return array dengan panjang = n_splits."""
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=100)
        cv    = CrossValidator(task="classification", n_splits=3)
        X     = pd.concat([data_split.X_train, data_split.X_val])
        y     = pd.concat([data_split.y_train, data_split.y_val])
        scores = cv.score(model, X, y, scoring="roc_auc")
        assert len(scores) == 3

    def test_from_config(self, ml_config):
        """from_config() membuat CrossValidator dengan benar."""
        cv = CrossValidator.from_config(ml_config)
        assert cv.task     == ml_config.modeling.task
        assert cv.n_splits == ml_config.modeling.n_cv_folds


# ══════════════════════════════════════════════════════════════
# MODELING PIPELINE TESTS
# ══════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestModelingPipeline:

    def test_run_returns_training_result(self, data_split, ml_config):
        """run() return TrainingResult yang valid."""
        from mltools.shared.schemas import TrainingResult

        pipeline = ModelingPipeline(ml_config)
        result   = pipeline.run(data_split)

        assert isinstance(result, TrainingResult)
        assert result.champion_name in ml_config.modeling.models_to_try
        assert len(result.test_metrics) > 0
        assert len(result.all_scores)   > 0

    def test_champion_beats_baseline(self, data_split, ml_config):
        """Champion model harus lebih baik dari baseline."""
        pipeline = ModelingPipeline(ml_config)
        result   = pipeline.run(data_split)

        champion_score = result.all_scores[result.champion_name]
        baseline       = BaselineModel(task=ml_config.modeling.task)
        baseline_scores= baseline.evaluate_from_split(data_split)

        assert champion_score >= baseline_scores.get("roc_auc", 0.0)

    def test_model_saved_to_disk(self, data_split, ml_config, tmp_path):
        """run() menyimpan model ke disk."""
        import os
        pipeline = ModelingPipeline(ml_config)
        result   = pipeline.run(data_split)
        assert os.path.exists(result.model_path)