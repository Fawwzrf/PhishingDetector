import json
import joblib
import pandas as pd
import numpy as np
import shap
from typing import Dict, Any, List
from pathlib import Path

from backend.schemas import PredictionResult, TopFeature, PredictionMeta

class PredictorPipeline:
    def __init__(self, base_dir: str = "models"):
        self.base_dir = Path(base_dir)
        self.meta = {}
        self.model = None
        self.explainer = None
        
        # Preprocessing steps
        self.missing_handler = None
        self.outlier_handler = None
        self.feature_engineer = None
        self.scaler = None
        self.selector = None
        
        self._load_artifacts()

    def _load_artifacts(self):
        # Load Meta
        with open(self.base_dir / "modeling_meta.json", "r") as f:
            self.meta = json.load(f)
            
        with open("all_features.json", "r") as f:
            self.raw_features = json.load(f)
            if "phishing" in self.raw_features:
                self.raw_features.remove("phishing")
            
        # Load preprocessing pipeline sequentially (needed to match what model was trained on)
        prep_dir = self.base_dir / "preprocessing"
        self.missing_handler = joblib.load(prep_dir / "missing_handler.joblib")
        self.outlier_handler = joblib.load(prep_dir / "outlier_handler.joblib")
        self.feature_engineer = joblib.load(prep_dir / "feature_engineer.joblib")
        self.scaler = joblib.load(prep_dir / "scaler.joblib")
        self.selector = joblib.load(prep_dir / "selector.joblib")
        
        # Load LightGBM model
        self.model = joblib.load(self.base_dir / "lightgbm_champion.joblib")
        
        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)

    def predict(self, raw_features: Dict[str, float]) -> Dict[str, Any]:
        """Runs the whole pipeline and returns the final predictions + SHAP values."""
        # Pad any unextracted features with NaN
        for feat in self.raw_features:
            if feat not in raw_features:
                raw_features[feat] = np.nan
        
        df_raw = pd.DataFrame([raw_features])[self.raw_features]
        
        # 1. Pipeline transformations
        df_proc = self.missing_handler.transform(df_raw)
        df_proc = self.outlier_handler.transform(df_proc)
        df_proc = self.feature_engineer.transform(df_proc)
        df_proc = self.scaler.transform(df_proc)
        df_proc = self.selector.transform(df_proc)
        
        # Ensure correct column order expected by the model
        feature_names = self.meta["feature_names"]
        df_final = df_proc[feature_names]
        
        # 2. Inference
        proba_array = self.model.predict(df_final)
        # Handle different predict returns (could be 1D or 2D)
        if hasattr(proba_array[0], '__len__'):
            probability = float(proba_array[0][1])  # Assuming class 1 is phishing
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
        # LightGBM SHAP returns raw log-odds. If shap_values is a list (multiclass format though binary), handle it
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
                model_version=self.meta.get("champion_name", "lightgbm_v1"),
                inference_time_ms=0, # Computed at the API level
                request_id=""
            )
        }
