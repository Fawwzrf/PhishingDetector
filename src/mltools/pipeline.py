# src/mltools/pipeline.py
# Top-level pipeline: Raw DataFrame → TrainingResult
# Menyambungkan PreprocessingPipeline dan ModelingPipeline

from __future__ import annotations

import json
import joblib
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from mltools.shared.config     import MLConfig
from mltools.shared.exceptions import PipelineError, PipelineNotFittedError
from mltools.shared.logging    import setup_logging
from mltools.shared.schemas    import DataSplit, TrainingResult

from mltools.preprocessing.pipeline import PreprocessingPipeline
from mltools.modeling.pipeline       import ModelingPipeline


class FullMLPipeline:
    """
    End-to-end ML pipeline: raw DataFrame → trained model.

    Menyambungkan PreprocessingPipeline dan ModelingPipeline
    menjadi satu interface tunggal yang dikontrol oleh MLConfig.

    Cara pakai:
        config   = MLConfig.from_yaml("configs/ml_config.yaml")
        pipeline = FullMLPipeline(config)
        result   = pipeline.run(df_raw)

    Untuk inference data baru:
        X_processed = pipeline.transform_new(df_new)
        predictions = pipeline.predict(df_new)
    """

    def __init__(self, config: MLConfig):
        self.config  = config
        self._is_fitted = False

        # Setup logging dari config
        setup_logging(
            log_level  = config.project.log_level,
            experiment = config.project.name,
        )

        # Inisialisasi dua pipeline utama
        self.preprocessing = PreprocessingPipeline(config)
        self.modeling      = ModelingPipeline(config)

        # Disimpan setelah run() selesai
        self.data_split_  : Optional[DataSplit]     = None
        self.result_      : Optional[TrainingResult] = None

    # ── MAIN ENTRY POINT ──────────────────────────────────────

    def run(self, df: pd.DataFrame) -> TrainingResult:
        """
        Jalankan full pipeline dari raw DataFrame.

        Args:
            df : DataFrame mentah yang sudah include kolom target

        Returns:
            TrainingResult berisi champion model + semua metrics
        """
        self._validate_input(df)

        logger.info("=" * 60)
        logger.info(f"  FULL ML PIPELINE: {self.config.project.name}")
        logger.info(f"  v{self.config.project.version}")
        logger.info("=" * 60)
        logger.info(f"  Input shape  : {df.shape}")
        logger.info(
            f"  Target       : {self.config.data.target_column}"
        )
        logger.info(f"  Task         : {self.config.modeling.task}")
        logger.info("=" * 60)

        # ── Step 1: Preprocessing ─────────────────────────────
        logger.info("")
        logger.info("PHASE 1/2 — PREPROCESSING")
        logger.info("-" * 60)

        data_split: DataSplit = self.preprocessing.run(df)
        self.data_split_ = data_split

        logger.success("Preprocessing selesai:")
        logger.success(f"  {data_split.summary()}")

        # ── Step 2: Modeling ──────────────────────────────────
        logger.info("")
        logger.info("PHASE 2/2 — MODELING")
        logger.info("-" * 60)

        result: TrainingResult = self.modeling.run(data_split)
        self.result_    = result
        self._is_fitted = True

        # ── Save pipeline artifacts ───────────────────────────
        self._save_artifacts()

        logger.info("")
        logger.success("=" * 60)
        logger.success("  FULL PIPELINE SELESAI!")
        logger.success("=" * 60)
        logger.success(result.summary())
        logger.success("=" * 60)

        return result

    # ── INFERENCE ─────────────────────────────────────────────

    def transform_new(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data baru untuk inference.
        Pipeline harus sudah di-run() dulu.

        Args:
            df : DataFrame baru TANPA kolom target

        Returns:
            DataFrame yang sudah dipreprocess
        """
        self._check_fitted("transform_new")
        return self.preprocessing.transform_new(df)

    def predict(self, df: pd.DataFrame) -> "pd.Series":
        """
        Prediksi kelas untuk data baru.

        Args:
            df : DataFrame baru TANPA kolom target

        Returns:
            Series berisi prediksi kelas
        """
        self._check_fitted("predict")

        X_processed = self.transform_new(df)
        raw_model   = self.result_.champion_model

        predictions = raw_model.predict(X_processed)
        return pd.Series(
            predictions,
            name  = f"predicted_{self.config.data.target_column}",
            index = df.index,
        )

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prediksi probabilitas untuk data baru (classification only).

        Args:
            df : DataFrame baru TANPA kolom target

        Returns:
            DataFrame berisi probabilitas per kelas
        """
        self._check_fitted("predict_proba")

        if self.config.modeling.task != "classification":
            raise PipelineError(
                "predict_proba hanya tersedia untuk task=classification"
            )

        X_processed = self.transform_new(df)
        raw_model   = self.result_.champion_model

        if not hasattr(raw_model, "predict_proba"):
            raise PipelineError(
                f"Model {self.result_.champion_name} "
                "tidak mendukung predict_proba"
            )

        proba = raw_model.predict_proba(X_processed)

        if self.data_split_ and self.data_split_.n_classes == 2:
            return pd.DataFrame(
                {
                    "prob_negative": proba[:, 0],
                    "prob_positive": proba[:, 1],
                },
                index = df.index,
            )

        return pd.DataFrame(
            proba,
            columns = [f"prob_class_{i}" for i in range(proba.shape[1])],
            index   = df.index,
        )

    # ── PERSISTENCE ───────────────────────────────────────────

    def save(self, path: str = "models/full_pipeline.joblib"):
        """
        Simpan seluruh FullMLPipeline ke disk.
        Termasuk preprocessing transformers + model champion.

        Args:
            path : path file output (.joblib)
        """
        self._check_fitted("save")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        logger.success(f"FullMLPipeline saved: {path}")

    @classmethod
    def load(cls, path: str) -> "FullMLPipeline":
        """
        Load FullMLPipeline dari disk.

        Args:
            path : path file .joblib

        Returns:
            FullMLPipeline yang sudah di-fit
        """
        if not Path(path).exists():
            raise PipelineError(
                f"File tidak ditemukan: {path}"
            )

        pipeline = joblib.load(path)

        if not isinstance(pipeline, cls):
            raise PipelineError(
                f"File bukan FullMLPipeline: {type(pipeline)}"
            )

        logger.success(f"FullMLPipeline loaded: {path}")
        return pipeline

    # ── REPORTING ─────────────────────────────────────────────

    def summary(self) -> str:
        """Ringkasan lengkap hasil pipeline."""
        self._check_fitted("summary")

        lines = [
            "",
            "=" * 60,
            f"  PIPELINE SUMMARY: {self.config.project.name}",
            "=" * 60,
            "",
            "  CONFIG",
            f"    Target  : {self.config.data.target_column}",
            f"    Task    : {self.config.modeling.task}",
            f"    Metric  : {self.config.modeling.metric}",
            "",
            "  DATA",
        ]

        if self.data_split_:
            for split_name, shape in self.data_split_.shapes.items():
                lines.append(f"    {split_name:<8}: {shape}")
            lines.append(
                f"    Features : {self.data_split_.n_features}"
            )

        if self.result_:
            lines += [
                "",
                "  MODELING",
                f"    Champion : {self.result_.champion_name}",
                "",
                "  TEST METRICS",
            ]
            for k, v in self.result_.test_metrics.items():
                lines.append(f"    {k:<15}: {v:.4f}")

            lines += [
                "",
                "  ALL MODEL SCORES (val)",
            ]
            for name, score in sorted(
                self.result_.all_scores.items(),
                key=lambda x: x[1], reverse=True,
            ):
                flag = " <- champion" if name == self.result_.champion_name else ""
                lines.append(f"    {name:<25}: {score:.4f}{flag}")

        lines += ["", "=" * 60]
        return "\n".join(lines)

    # ── INTERNAL HELPERS ──────────────────────────────────────

    def _validate_input(self, df: pd.DataFrame):
        """Validasi DataFrame input sebelum pipeline dijalankan."""
        target = self.config.data.target_column

        if not isinstance(df, pd.DataFrame):
            raise PipelineError(
                f"Input harus DataFrame, bukan {type(df)}",
            )

        if target not in df.columns:
            raise PipelineError(
                f"Target kolom '{target}' tidak ada di DataFrame",
                details={"available_columns": list(df.columns)},
            )

        if len(df) < 100:
            logger.warning(
                f"Dataset sangat kecil ({len(df)} baris). "
                "Hasil model mungkin tidak reliable."
            )

        if df[target].isnull().any():
            n_null = df[target].isnull().sum()
            raise PipelineError(
                f"Target '{target}' mengandung {n_null} missing values. "
                "Hapus baris tersebut sebelum training.",
                details={"n_null_target": n_null},
            )

    def _check_fitted(self, method_name: str):
        """Pastikan pipeline sudah di-run() sebelum method dipanggil."""
        if not self._is_fitted:
            raise PipelineNotFittedError(
                f"FullMLPipeline.{method_name}() dipanggil "
                "tapi pipeline belum di-fit. "
                "Jalankan .run(df) dulu."
            )

    def _save_artifacts(self):
        """Save semua artifacts penting setelah pipeline selesai."""
        Path("models").mkdir(parents=True, exist_ok=True)

        # 1. Simpan feature names untuk validasi saat inference
        if self.result_:
            feature_path = "models/feature_names.json"
            with open(feature_path, "w") as f:
                json.dump(
                    {
                        "feature_names": self.result_.feature_names,
                        "target"       : self.config.data.target_column,
                        "task"         : self.config.modeling.task,
                        "champion"     : self.result_.champion_name,
                    },
                    f, indent=2,
                )
            logger.info(f"Feature names saved: {feature_path}")

        # 2. Simpan pipeline summary ke text file
        summary_path = f"models/{self.config.project.name}_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(self.summary())
        logger.info(f"Summary saved: {summary_path}")