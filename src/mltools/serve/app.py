# src/mltools/serve/app.py

"""
FastAPI application untuk serving model Phishing Detector.

Jalankan:
    uvicorn mltools.serve.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import json
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from loguru import logger

from mltools.serve.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)

# ── Global state ──────────────────────────────────────────────
_state: Dict[str, Any] = {}

META_PATH = Path("models/modeling_meta.json")


def _load_model_and_meta() -> None:
    """Load champion model & metadata saat startup."""

    if not META_PATH.exists():
        raise FileNotFoundError(
            f"modeling_meta.json tidak ditemukan di {META_PATH}. "
            "Pastikan Anda sudah menjalankan tahap modelling."
        )

    with open(META_PATH) as f:
        meta = json.load(f)

    # ── Load model via registry ───────────────────────────────
    from mltools.registry.model_registry import ModelRegistry

    registry = ModelRegistry(base_dir="models")
    model = registry.load(name=meta["champion_name"], version="champion")

    _state["model"] = model
    _state["meta"] = meta
    _state["feature_names"] = meta["feature_names"]
    _state["threshold"] = meta.get("optimal_threshold", 0.5)
    _state["champion_info"] = registry.get_champion()

    logger.success(
        f"Model loaded: {meta['champion_name']} "
        f"({len(meta['feature_names'])} features, "
        f"threshold={_state['threshold']:.4f})"
    )


# ── Lifespan ──────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model saat startup, cleanup saat shutdown."""
    logger.info("Loading champion model...")
    _load_model_and_meta()
    yield
    logger.info("Shutting down serving app.")


# ── FastAPI App ───────────────────────────────────────────────

app = FastAPI(
    title="🎣 Phishing Detector API",
    description=(
        "API untuk mendeteksi website phishing menggunakan model "
        "LightGBM champion yang sudah di-tuning dengan Optuna."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Helper ────────────────────────────────────────────────────


def _predict_single(features: Dict[str, float]) -> PredictionResponse:
    """Prediksi satu sample."""
    model = _state["model"]
    feature_names = _state["feature_names"]
    threshold = _state["threshold"]

    # Validasi fitur
    missing = set(feature_names) - set(features.keys())
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Fitur yang hilang: {sorted(missing)}",
        )

    # Buat DataFrame dengan urutan kolom yang benar
    df = pd.DataFrame([features])[feature_names]

    # Prediksi — lgb.Booster.predict() mengembalikan probabilitas langsung
    proba = model.predict(df)[0]

    # Jika model mengembalikan array 2D (misal dari sklearn), ambil kolom ke-2
    if hasattr(proba, "__len__") and len(proba) > 1:
        proba = float(proba[1])
    else:
        proba = float(proba)

    prediction = 1 if proba >= threshold else 0
    label = "phishing" if prediction == 1 else "legit"

    return PredictionResponse(
        prediction=prediction,
        probability=round(proba, 6),
        label=label,
        threshold=round(threshold, 6),
    )


# ── Endpoints ─────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Cek status server & informasi model yang sedang digunakan."""
    meta = _state.get("meta")
    champion = _state.get("champion_info", {})

    if meta is None:
        raise HTTPException(status_code=503, detail="Model belum dimuat.")

    return HealthResponse(
        status="ok",
        model_name=meta["champion_name"],
        model_version=champion.get("version", "unknown"),
        n_features=meta["n_features"],
        test_metrics=meta["test_metrics"],
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Prediksi apakah satu URL adalah phishing atau legit.

    Kirimkan dict `features` dengan 24 fitur numerik yang diperlukan.
    """
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model belum dimuat.")

    return _predict_single(request.features)


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
)
async def predict_batch(request: BatchPredictionRequest):
    """
    Prediksi batch — kirimkan list of feature dicts.

    Maksimum 1000 samples per request.
    """
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model belum dimuat.")

    if len(request.samples) > 1000:
        raise HTTPException(
            status_code=422,
            detail="Maksimum 1000 samples per batch request.",
        )

    results = [_predict_single(sample) for sample in request.samples]

    return BatchPredictionResponse(
        predictions=results,
        total=len(results),
    )
