# src/mltools/preprocessing/pipeline.py

from __future__ import annotations

import pandas as pd
from loguru import logger

from mltools.shared.config import MLConfig
from mltools.shared.exceptions import PipelineError, PipelineNotFittedError
from mltools.shared.schemas import DataSplit

from mltools.preprocessing.inspector          import DataInspector
from mltools.preprocessing.missing_handler    import ExpertMissingHandler, validate_no_missing
from mltools.preprocessing.outlier_handler    import ExpertOutlierHandler
from mltools.preprocessing.encoder            import ExpertCategoricalEncoder
from mltools.preprocessing.scaler             import ExpertScalerTransformer
from mltools.preprocessing.engineer           import ExpertFeatureEngineer, DatetimeFeatureExtractor
from mltools.preprocessing.selector           import ExpertFeatureSelector
from mltools.preprocessing.splitter           import ExpertDataSplitter, check_data_leakage
from mltools.preprocessing.imbalanced_handler import ExpertImbalancedHandler


class PreprocessingPipeline:
    """
    Full preprocessing pipeline yang membaca konfigurasi dari MLConfig
    dan menghasilkan DataSplit siap pakai untuk ModelingPipeline.

    Cara pakai:
        config   = MLConfig.from_yaml("configs/ml_config.yaml")
        pipeline = PreprocessingPipeline(config)
        split    = pipeline.run(df_raw)
    """

    def __init__(self, config: MLConfig):
        self.config   = config
        self._is_fitted = False

        # ── Inisialisasi semua komponen dari config ────────────────────
        cfg_miss = config.preprocessing.missing_values
        cfg_out  = config.preprocessing.outliers
        cfg_enc  = config.preprocessing.encoding
        cfg_scl  = config.preprocessing.scaling
        cfg_sel  = config.preprocessing.feature_selection
        cfg_mod  = config.modeling

        self.missing_handler = ExpertMissingHandler(
            drop_col_threshold = cfg_miss.threshold_drop_column,
            drop_row_threshold = cfg_miss.threshold_drop_row,
            num_strategy       = cfg_miss.strategy_numerical,
            cat_strategy       = cfg_miss.strategy_categorical,
            add_missing_indicator = True,
        )

        self.outlier_handler = ExpertOutlierHandler(
            method    = cfg_out.method,
            treatment = cfg_out.treatment,
            threshold = cfg_out.threshold,
        )

        self.encoder = ExpertCategoricalEncoder(
            model_type            = "tree",
            cardinality_threshold = cfg_enc.high_cardinality_threshold,
            high_card_method      = cfg_enc.default_strategy,
        )

        self.scaler = ExpertScalerTransformer(
            scaler         = cfg_scl.strategy,
            auto_transform = True,
            exclude_cols   = config.data.id_columns,
        )

        self.engineer = ExpertFeatureEngineer()

        self.selector = ExpertFeatureSelector(
            task              = cfg_mod.task,
            variance_thr      = cfg_sel.variance_threshold,
            corr_threshold    = cfg_sel.correlation_threshold,
            importance_method = "tree",
            n_features        = (
                None if cfg_sel.n_features_to_select == "auto"
                else int(cfg_sel.n_features_to_select)
            ),
            random_state      = config.project.random_state,
        )

        self.splitter = ExpertDataSplitter(
            task         = cfg_mod.task,
            random_state = config.project.random_state,
        )

        self.imbalanced_handler = ExpertImbalancedHandler(
            strategy     = "smotetomek",
            random_state = config.project.random_state,
        )

    def run(self, df: pd.DataFrame) -> DataSplit:
        """
        Jalankan full preprocessing pipeline.

        Args:
            df: DataFrame mentah (sudah include kolom target)

        Returns:
            DataSplit: siap dipakai ModelingPipeline
        """
        config = self.config
        target = config.data.target_column

        # ── Validasi input ─────────────────────────────────────────────
        if target not in df.columns:
            raise PipelineError(
                f"Target kolom '{target}' tidak ada di DataFrame",
                details={"available": list(df.columns)}
            )

        logger.info("=" * 55)
        logger.info(f"PREPROCESSING PIPELINE: {config.project.name}")
        logger.info(f"Target: {target} | Task: {config.modeling.task}")
        logger.info(f"Input shape: {df.shape}")
        logger.info("=" * 55)

        # ── Step 1: Drop kolom ID ──────────────────────────────────────
        drop_cols = [c for c in config.data.id_columns if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
            logger.info(f"[1/9] Dropped ID columns: {drop_cols}")
        else:
            logger.info("[1/9] Tidak ada ID columns untuk di-drop")

        # ── Step 2: Pisah X dan y ──────────────────────────────────────
        X = df.drop(columns=[target])
        y = df[target]
        logger.info(f"[2/9] X shape: {X.shape} | y shape: {y.shape}")

        # ── Step 3: Split train / val / test ───────────────────────────
        logger.info("[3/9] Splitting data...")
        X_train, X_val, X_test, y_train, y_val, y_test = \
            self.splitter.split_holdout(X, y)

        # ── Step 4: Missing values (FIT pada train saja!) ──────────────
        logger.info("[4/9] Handling missing values...")
        self.missing_handler.fit(X_train)
        X_train = self.missing_handler.transform(X_train)
        X_val   = self.missing_handler.transform(X_val)
        X_test  = self.missing_handler.transform(X_test)
        validate_no_missing(X_train)

        # ── Step 5: Outlier handling ───────────────────────────────────
        logger.info("[5/9] Handling outliers...")
        self.outlier_handler.fit(X_train)
        X_train = self.outlier_handler.transform(X_train)
        X_val   = self.outlier_handler.transform(X_val)
        X_test  = self.outlier_handler.transform(X_test)

        # ── Step 6: Encoding ───────────────────────────────────────────
        logger.info("[6/9] Encoding categorical features...")
        self.encoder.fit(X_train, y_train)
        X_train = self.encoder.transform(X_train)
        X_val   = self.encoder.transform(X_val)
        X_test  = self.encoder.transform(X_test)

        # ── Step 7: Scaling ────────────────────────────────────────────
        logger.info("[7/9] Scaling features...")
        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        X_val   = self.scaler.transform(X_val)
        X_test  = self.scaler.transform(X_test)

        # ── Step 8: Feature selection ──────────────────────────────────
        logger.info("[8/9] Selecting features...")
        self.selector.fit(X_train, y_train)
        X_train = self.selector.transform(X_train)
        X_val   = self.selector.transform(X_val)
        X_test  = self.selector.transform(X_test)

        # ── Step 9: Imbalanced handling (HANYA train!) ─────────────────
        if config.modeling.task == "classification":
            ratio = y_train.value_counts(normalize=True).min()
            if ratio < 0.3:
                logger.info(
                    f"[9/9] Imbalanced detected (minority={ratio:.1%}), resampling..."
                )
                X_train, y_train = self.imbalanced_handler.fit_resample(
                    X_train, y_train
                )
            else:
                logger.info(
                    f"[9/9] Kelas cukup seimbang ({ratio:.1%}), skip resampling"
                )
        else:
            logger.info("[9/9] Regression task, skip imbalanced handling")

        # ── Cek leakage ────────────────────────────────────────────────
        check_data_leakage(X_train, X_test)

        # ── Buat DataSplit ─────────────────────────────────────────────
        self._is_fitted = True
        feature_names   = X_train.columns.tolist()

        split = DataSplit(
            X_train      = X_train.reset_index(drop=True),
            X_val        = X_val.reset_index(drop=True),
            X_test       = X_test.reset_index(drop=True),
            y_train      = y_train.reset_index(drop=True),
            y_val        = y_val.reset_index(drop=True),
            y_test       = y_test.reset_index(drop=True),
            feature_names= feature_names,
            target_name  = target,
            task         = config.modeling.task,
        )

        logger.success("=" * 55)
        logger.success("PREPROCESSING SELESAI!")
        logger.success(split.summary())
        logger.success("=" * 55)

        return split

    def transform_new(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data baru (production inference) menggunakan
        pipeline yang sudah di-fit.

        Args:
            df: DataFrame baru tanpa kolom target

        Returns:
            DataFrame yang sudah dipreprocess
        """
        if not self._is_fitted:
            raise PipelineNotFittedError(
                "Pipeline belum di-fit. Jalankan .run() dulu."
            )

        config   = self.config
        drop_cols = [c for c in config.data.id_columns if c in df.columns]
        X = df.drop(columns=drop_cols, errors="ignore")

        X = self.missing_handler.transform(X)
        X = self.outlier_handler.transform(X)
        X = self.encoder.transform(X)
        X = self.scaler.transform(X)
        X = self.selector.transform(X)

        return X