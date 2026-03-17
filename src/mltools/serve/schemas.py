# src/mltools/serve/schemas.py

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class PredictionRequest(BaseModel):
    """Request body untuk prediksi satu URL."""

    features: Dict[str, float] = Field(
        ...,
        description="Dict fitur — key=nama fitur, value=nilai numerik",
        json_schema_extra={
            "example": {
                "qty_dot_directory_was_missing": 0.0,
                "qty_equal_file": 1.0,
                "length_url_bin": 2.0,
                "qty_underline_file": 0.0,
                "time_domain_activation": 365.0,
                "qty_hyphen_file": 1.0,
                "directory_length": 25.0,
                "qty_equal_directory": 0.0,
                "qty_slash_directory": 3.0,
                "qty_comma_file": 0.0,
                "qty_dot_domain": 1.0,
                "qty_and_file": 0.0,
                "qty_space_file": 0.0,
                "qty_underline_directory": 0.0,
                "qty_plus_file": 0.0,
                "qty_dot_directory": 2.0,
                "qty_hyphen_directory": 0.0,
                "qty_exclamation_file": 0.0,
                "time_domain_expiration": 365.0,
                "file_length": 15.0,
                "domain_spf": 1.0,
                "time_domain_activation_was_missing": 0.0,
                "qty_space_directory": 0.0,
                "qty_dot_file": 1.0,
            }
        },
    )


class PredictionResponse(BaseModel):
    """Response body untuk prediksi satu URL."""

    prediction : int   = Field(..., description="0 = legit, 1 = phishing")
    probability: float = Field(..., description="Probabilitas phishing (0-1)")
    label      : str   = Field(..., description="'legit' atau 'phishing'")
    threshold  : float = Field(..., description="Threshold yang digunakan")


class BatchPredictionRequest(BaseModel):
    """Request body untuk prediksi batch (banyak URL sekaligus)."""

    samples: List[Dict[str, float]] = Field(
        ...,
        description="List of feature dicts, satu dict per sample",
    )


class BatchPredictionResponse(BaseModel):
    """Response body untuk prediksi batch."""

    predictions: List[PredictionResponse]
    total      : int


class HealthResponse(BaseModel):
    """Response body untuk health check."""

    status       : str
    model_name   : str
    model_version: str
    n_features   : int
    test_metrics : Dict[str, float]
