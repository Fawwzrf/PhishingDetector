# tests/test_pipeline.py
# Integration test untuk FullMLPipeline end-to-end

import pytest
import numpy as np
import pandas as pd

from mltools import FullMLPipeline, MLConfig
from mltools.shared.exceptions import PipelineError, PipelineNotFittedError
from mltools.shared.schemas    import TrainingResult


# ══════════════════════════════════════════════════════════════
# VALIDATION TESTS
# ══════════════════════════════════════════════════════════════

class TestPipelineValidation:

    def test_missing_target_raises(self, raw_dataframe, ml_config):
        """Pipeline raise jika target kolom tidak ada."""
        df_no_target = raw_dataframe.drop(columns=["target"])

        pipeline = FullMLPipeline(ml_config)
        with pytest.raises(PipelineError, match="Target kolom"):
            pipeline.run(df_no_target)

    def test_null_target_raises(self, raw_dataframe, ml_config):
        """Pipeline raise jika target mengandung null."""
        df_null    = raw_dataframe.copy()
        df_null.loc[0, "target"] = np.nan

        pipeline = FullMLPipeline(ml_config)
        with pytest.raises(PipelineError, match="missing values"):
            pipeline.run(df_null)

    def test_not_fitted_predict_raises(self, raw_dataframe, ml_config):
        """predict() sebelum run() raise PipelineNotFittedError."""
        pipeline = FullMLPipeline(ml_config)
        X_new    = raw_dataframe.drop(columns=["target"]).head(5)

        with pytest.raises(PipelineNotFittedError):
            pipeline.predict(X_new)

    def test_not_fitted_transform_raises(self, raw_dataframe, ml_config):
        """transform_new() sebelum run() raise PipelineNotFittedError."""
        pipeline = FullMLPipeline(ml_config)
        X_new    = raw_dataframe.drop(columns=["target"]).head(5)

        with pytest.raises(PipelineNotFittedError):
            pipeline.transform_new(X_new)


# ══════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ══════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestFullPipelineIntegration:

    @pytest.fixture(scope="class")
    def fitted_pipeline(self, raw_dataframe, ml_config):
        """FullMLPipeline yang sudah di-run() — dibuat sekali per class."""
        pipeline = FullMLPipeline(ml_config)
        pipeline.run(raw_dataframe)
        return pipeline

    def test_run_returns_training_result(
        self, raw_dataframe, ml_config
    ):
        """run() return TrainingResult yang valid."""
        pipeline = FullMLPipeline(ml_config)
        result   = pipeline.run(raw_dataframe)

        assert isinstance(result, TrainingResult)
        assert result.champion_name is not None
        assert len(result.test_metrics) > 0
        assert len(result.feature_names) > 0

    def test_pipeline_is_fitted_after_run(
        self, raw_dataframe, ml_config
    ):
        """_is_fitted = True setelah run()."""
        pipeline = FullMLPipeline(ml_config)
        pipeline.run(raw_dataframe)
        assert pipeline._is_fitted == True

    def test_predict_returns_series(
        self, fitted_pipeline, raw_dataframe
    ):
        """predict() return Series dengan panjang yang benar."""
        X_new    = raw_dataframe.drop(columns=["target"]).head(10)
        preds    = fitted_pipeline.predict(X_new)

        assert isinstance(preds, pd.Series)
        assert len(preds) == 10

    def test_predict_proba_returns_dataframe(
        self, fitted_pipeline, raw_dataframe
    ):
        """predict_proba() return DataFrame dengan kolom prob."""
        X_new  = raw_dataframe.drop(columns=["target"]).head(10)
        proba  = fitted_pipeline.predict_proba(X_new)

        assert isinstance(proba, pd.DataFrame)
        assert len(proba) == 10
        # Setiap baris sum ke 1
        np.testing.assert_allclose(
            proba.sum(axis=1), 1.0, atol=1e-5
        )

    def test_transform_new_correct_features(
        self, fitted_pipeline, raw_dataframe
    ):
        """transform_new() menghasilkan fitur yang sama dengan training."""
        X_new        = raw_dataframe.drop(columns=["target"]).head(5)
        X_processed  = fitted_pipeline.transform_new(X_new)

        # Kolom harus sesuai dengan feature_names hasil training
        expected     = fitted_pipeline.result_.feature_names
        assert list(X_processed.columns) == expected

    def test_summary_returns_string(self, fitted_pipeline):
        """summary() return string non-empty."""
        result = fitted_pipeline.summary()
        assert isinstance(result, str)
        assert "champion" in result.lower()

    def test_data_split_stored(self, fitted_pipeline):
        """data_split_ tersimpan setelah run()."""
        assert fitted_pipeline.data_split_ is not None
        assert fitted_pipeline.data_split_.task == "classification"

    def test_result_stored(self, fitted_pipeline):
        """result_ tersimpan setelah run()."""
        assert fitted_pipeline.result_ is not None


# ══════════════════════════════════════════════════════════════
# PERSISTENCE TESTS
# ══════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestPipelinePersistence:

    def test_save_and_load(
        self, raw_dataframe, ml_config, tmp_path
    ):
        """Pipeline bisa di-save dan di-load kembali."""
        path     = str(tmp_path / "pipeline.joblib")

        pipeline = FullMLPipeline(ml_config)
        pipeline.run(raw_dataframe)
        pipeline.save(path)

        pipeline2 = FullMLPipeline.load(path)
        assert pipeline2._is_fitted == True
        assert pipeline2.result_.champion_name == \
               pipeline.result_.champion_name

    def test_loaded_pipeline_can_predict(
        self, raw_dataframe, ml_config, tmp_path
    ):
        """Pipeline yang di-load bisa melakukan prediksi."""
        path     = str(tmp_path / "pipeline.joblib")

        pipeline = FullMLPipeline(ml_config)
        pipeline.run(raw_dataframe)
        pipeline.save(path)

        pipeline2 = FullMLPipeline.load(path)
        X_new     = raw_dataframe.drop(columns=["target"]).head(5)
        preds     = pipeline2.predict(X_new)
        assert len(preds) == 5

    def test_predictions_consistent_after_load(
        self, raw_dataframe, ml_config, tmp_path
    ):
        """Prediksi sebelum dan sesudah load harus sama."""
        path      = str(tmp_path / "pipeline.joblib")
        X_new     = raw_dataframe.drop(columns=["target"]).head(10)

        pipeline  = FullMLPipeline(ml_config)
        pipeline.run(raw_dataframe)
        preds_before = pipeline.predict(X_new).values

        pipeline.save(path)
        pipeline2 = FullMLPipeline.load(path)
        preds_after  = pipeline2.predict(X_new).values

        np.testing.assert_array_equal(preds_before, preds_after)

    def test_load_nonexistent_file_raises(self, ml_config):
        """load() raise PipelineError jika file tidak ada."""
        with pytest.raises(PipelineError, match="tidak ditemukan"):
            FullMLPipeline.load("tidak_ada.joblib")