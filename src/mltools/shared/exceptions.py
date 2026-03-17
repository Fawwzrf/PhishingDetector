# src/mltools/shared/exceptions.py


class MLToolsError(Exception):
    """Base exception untuk seluruh mltools package."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(
                f"{k}={v}" for k, v in self.details.items()
            )
            return f"{self.message} [{detail_str}]"
        return self.message


# ── CONFIG ────────────────────────────────────────────────────
class ConfigError(MLToolsError):
    """Error saat load atau validasi config."""
    pass


# ── DATA ──────────────────────────────────────────────────────
class DataError(MLToolsError):
    """Base error untuk semua masalah data."""
    pass

class DataShapeError(DataError):
    def __init__(self, expected, actual):
        super().__init__(
            "Shape tidak sesuai",
            details={"expected": str(expected), "actual": str(actual)}
        )

class DataLeakageError(DataError):
    """Terdeteksi potential data leakage."""
    pass

class DataTypeError(DataError):
    """Tipe data tidak sesuai yang diharapkan."""
    pass


# ── PREPROCESSING ─────────────────────────────────────────────
class PreprocessingError(MLToolsError):
    """Base error untuk preprocessing."""
    pass

class MissingHandlerError(PreprocessingError):
    pass

class OutlierHandlerError(PreprocessingError):
    pass

class EncodingError(PreprocessingError):
    pass

class ScalingError(PreprocessingError):
    pass

class FeatureSelectionError(PreprocessingError):
    pass

class SplittingError(PreprocessingError):
    pass


# ── MODELING ──────────────────────────────────────────────────
class ModelingError(MLToolsError):
    """Base error untuk modeling."""
    pass

class ModelNotFittedError(ModelingError):
    def __init__(self, model_name: str):
        super().__init__(
            f"Model '{model_name}' belum di-fit. "
            "Panggil .fit() dulu.",
            details={"model_name": model_name}
        )

class ModelNotFoundError(ModelingError):
    def __init__(self, model_name: str, available: list):
        super().__init__(
            f"Model '{model_name}' tidak dikenal.",
            details={"available": str(available)}
        )

class TuningError(ModelingError):
    pass

class EvaluationError(ModelingError):
    pass


# ── PIPELINE ──────────────────────────────────────────────────
class PipelineError(MLToolsError):
    """Error di level pipeline."""
    pass

class PipelineNotFittedError(PipelineError):
    pass


# ── SERVING ───────────────────────────────────────────────────
class ServingError(MLToolsError):
    """Error saat model serving / inference."""
    pass