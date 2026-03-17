from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

class PredictionRequest(BaseModel):
    url: str = Field(..., description="URL yang akan dianalisis", json_schema_extra={"example": "https://example.com/login"})

class PredictionResult(BaseModel):
    label: str = Field(..., description="'LEGITIMATE' atau 'PHISHING'")
    probability: float = Field(..., description="Probabilitas phishing (0-1)")
    confidence: str = Field(..., description="'HIGH', 'MEDIUM', atau 'LOW'")
    threshold: float = Field(..., description="Threshold yang digunakan")

class URLAnalysis(BaseModel):
    url_original: str
    url_decoded: str
    is_punycode: bool
    punycode_warning: Optional[str] = None

class TopFeature(BaseModel):
    feature: str
    value: float
    shap: float
    direction: str = Field(..., description="'phishing' atau 'legit'")

class PredictionMeta(BaseModel):
    model_version: str
    inference_time_ms: int
    request_id: str

class PredictionResponse(BaseModel):
    result: PredictionResult
    url_analysis: URLAnalysis
    top_features: List[TopFeature]
    all_features: Dict[str, float]
    meta: PredictionMeta

class FeatureSchema(BaseModel):
    name: str
    description: str = ""
    category: str = ""
    type: str = "numeric"

class FeatureSchemaResponse(BaseModel):
    total_features: int
    features: List[FeatureSchema]

class HealthResponse(BaseModel):
    status: str
    model: str
    model_version: str
    threshold: float
