import json
import joblib
import pandas as pd
import numpy as np
import shap
from typing import Dict, Any, List
from pathlib import Path

from backend.schemas import PredictionResult, TopFeature, PredictionMeta

# Make sure mltools is accessible
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from mltools.pipeline import FullMLPipeline

class PredictorPipeline:
    def __init__(self, base_dir: str = "models"):
        self.base_dir = Path(base_dir)
        self.meta = {}
        self.model = None
        self.explainer = None
        
        self.full_pipeline = None
        
        self._load_artifacts()

    def _load_artifacts(self):
        # Load FullMLPipeline
        pipeline_path = self.base_dir / "full_pipeline.joblib"
        self.full_pipeline = FullMLPipeline.load(str(pipeline_path))
        
        # Load feature list expected by extraction
        with open("all_features.json", "r") as f:
            self.raw_features = json.load(f)
            if "phishing" in self.raw_features:
                self.raw_features.remove("phishing")
        
        feature_names_path = self.base_dir / "feature_names.json"
        if feature_names_path.exists():
            with open(feature_names_path, "r") as f:
                 self.meta = json.load(f)
        else:
            self.meta = {"champion": "lightgbm", "optimal_threshold": 0.5, "feature_names": self.full_pipeline.result_.feature_names}
            
        self.model = self.full_pipeline.result_.champion_model
        
        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)

    def predict(self, raw_features: Dict[str, float]) -> Dict[str, Any]:
        """Runs the whole pipeline and returns the final predictions + SHAP values."""
        # Pad any unextracted features with NaN
        for feat in self.raw_features:
            if feat not in raw_features:
                raw_features[feat] = np.nan
        
        # Convert dictionary to single-row dataframe
        df_raw = pd.DataFrame([raw_features])
        
        # 1. Pipeline transformations directly via transform_new
        df_final = self.full_pipeline.transform_new(df_raw)
        feature_names = self.meta.get("feature_names", self.full_pipeline.result_.feature_names)
        
        # 2. Inference
        proba_array = self.model.predict(df_final)
        if hasattr(proba_array[0], '__len__'):
            probability = float(proba_array[0][1])
        else:
            probability = float(proba_array[0])
            
        threshold = self.meta.get("optimal_threshold", 0.5)
        is_phishing = probability >= threshold
        
        # 3. Confidence formulation
        if is_phishing:
            if probability > 0.90: confidence = "HIGH"
            elif probability > 0.70: confidence = "MEDIUM"
            else: confidence = "LOW"
        else:
            if probability < 0.10: confidence = "HIGH"
            elif probability < 0.30: confidence = "MEDIUM"
            else: confidence = "LOW"
            
        result = PredictionResult(
            label="PHISHING" if is_phishing else "LEGITIMATE",
            probability=probability,
            confidence=confidence,
            threshold=threshold
        )
        
        # 4. SHAP Explanation
        shap_values = self.explainer.shap_values(df_final)[0]
        if isinstance(shap_values, list):
            shap_values = shap_values[1][0] 
            
        top_indices = np.argsort(np.abs(shap_values))[-5:][::-1]
        
        top_features = []
        for idx in top_indices:
            feat_name = feature_names[idx]
            feat_val = float(df_final.iloc[0, idx])
            shap_val = float(shap_values[idx])
            direction = "phishing" if shap_val > 0 else "legit"
            
            top_features.append(TopFeature(
                feature=feat_name,
                value=feat_val,
                shap=shap_val,
                direction=direction
            ))
            
        return {
            "result": result,
            "top_features": top_features,
            "meta": PredictionMeta(
                model_version=self.meta.get("champion", "lightgbm"),
                inference_time_ms=0, 
                request_id=""
            )
        }
