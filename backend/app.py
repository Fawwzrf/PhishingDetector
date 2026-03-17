from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time
import uuid

from backend.schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    FeatureSchemaResponse,
    FeatureSchema
)
from backend.extractor import extract_all_features
from backend.predictor import PredictorPipeline
from backend.mlflow_logger import log_prediction_async

# Global state to hold model pipeline
model_pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_pipeline
    print("Loading PredictorPipeline...")
    model_pipeline = PredictorPipeline()
    yield
    print("Shutting down API...")
    model_pipeline = None

app = FastAPI(
    title="PhishingDetector API",
    description="Backend API for predicting phishing URLs in real-time.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration for Frontend to communicate with Backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, lock this down
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint to verify model loaded properly."""
    if not model_pipeline:
        raise HTTPException(status_code=503, detail="Model belum siap")
        
    return HealthResponse(
        status="healthy",
        model=model_pipeline.meta.get("champion_name", "lightgbm"),
        model_version=model_pipeline.meta.get("champion_name", "v1.0.0"),
        threshold=model_pipeline.meta.get("optimal_threshold", 0.522)
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_url(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Main URL prediction endpoint."""
    if not model_pipeline:
        raise HTTPException(status_code=503, detail="Model belum siap")
        
    url = request.url
    if not url.startswith("http://") and not url.startswith("https://"):
        raise HTTPException(status_code=422, detail="URL tidak valid. Pastikan dimulai dengan http:// atau https://")
        
    start_time = time.perf_counter()
    req_id = f"req_{uuid.uuid4().hex[:8]}"
    
    # 1. Extract raw features
    all_features, url_analysis = await extract_all_features(url)
    
    # 2. Inference via Pipeline
    try:
        prediction_output = model_pipeline.predict(all_features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
        
    # 3. Wrap response
    end_time = time.perf_counter()
    inference_time_ms = int((end_time - start_time) * 1000)
    
    prediction_output["meta"].inference_time_ms = inference_time_ms
    prediction_output["meta"].request_id = req_id
    
    response = PredictionResponse(
        result=prediction_output["result"],
        url_analysis=url_analysis,
        top_features=prediction_output["top_features"],
        all_features=all_features,
        meta=prediction_output["meta"]
    )
    
    # 4. Log to MLflow asynchronously
    background_tasks.add_task(log_prediction_async, url, response)
    
    return response

@app.get("/features/schema", response_model=FeatureSchemaResponse)
def get_features_schema():
    if not model_pipeline:
        raise HTTPException(status_code=503, detail="Model belum siap")
        
    features = []
    for f in model_pipeline.meta.get("feature_names", []):
        features.append(FeatureSchema(
            name=f,
            description="extracted feature",
            category="unknown",
            type="numeric"
        ))
        
    return FeatureSchemaResponse(
        total_features=len(features),
        features=features
    )
