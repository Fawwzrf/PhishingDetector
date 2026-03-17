import mlflow
import hashlib
import time
from backend.schemas import PredictionResponse

def log_prediction_async(url: str, response: PredictionResponse):
    """Logs the prediction features, result, and meta to MLflow sequentially (should be run in BackgroundTasks)."""
    try:
        mlflow.set_experiment("phishing_detector_inference")
        
        with mlflow.start_run(run_name=f"predict_{int(time.time())}"):
            # Hash URL to maintain privacy per requirements
            hashed_url = hashlib.md5(url.encode()).hexdigest()
            mlflow.set_tag("url_hash", hashed_url)
            mlflow.set_tag("env", "production")
            mlflow.set_tag("model_version", response.meta.model_version)
            mlflow.set_tag("is_punycode", str(response.url_analysis.is_punycode))
            
            # Log all features as params
            mlflow.log_params(response.all_features)
            
            # Log metrics
            mlflow.log_metric("probability", response.result.probability)
            mlflow.log_metric("inference_ms", response.meta.inference_time_ms)
            mlflow.log_metric("is_phishing", 1 if response.result.label == "PHISHING" else 0)
    except Exception as e:
        print(f"MLflow logging failed: {e}")
