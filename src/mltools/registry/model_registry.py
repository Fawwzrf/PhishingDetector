# src/mltools/registry/model_registry.py

from __future__ import annotations

import json
import joblib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from mltools.shared.exceptions import ModelingError
from mltools.shared.schemas    import TrainingResult


class ModelRegistry:
    """
    Model registry untuk versioning dan manajemen model.

    Fitur:
    - Save model dengan versi otomatis berdasarkan timestamp
    - Load model berdasarkan versi atau 'latest'
    - Tandai model sebagai champion
    - List semua model yang tersimpan
    - Simpan metadata lengkap per model

    Cara pakai:
        registry = ModelRegistry()
        registry.save(result)
        model    = registry.load("lightgbm")
        df       = registry.list_models()
    """

    REGISTRY_FILE = "models/registry.json"

    def __init__(self, base_dir: str = "models"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._load_registry()

    # ── SAVE ─────────────────────────────────────────────────

    def save(
        self,
        result      : TrainingResult,
        is_champion : bool = True,
        compress    : int  = 3,
    ) -> str:
        """
        Save model dari TrainingResult dengan versioning otomatis.

        Args:
            result      : TrainingResult dari ModelingPipeline
            is_champion : Tandai sebagai champion
            compress    : Level kompresi joblib (0-9)

        Returns:
            Path ke file model yang disimpan
        """
        name      = result.champion_name
        version   = datetime.now().strftime("v_%Y%m%d_%H%M%S")
        model_dir = self.base_dir / name / version
        model_dir.mkdir(parents=True, exist_ok=True)

        # ── Pilih format berdasarkan tipe model ───────────────
        model     = result.champion_model
        model_type= type(model).__name__.lower()
        path      = self._save_model_file(
            model, model_type, model_dir
        )

        # ── Simpan metadata ───────────────────────────────────
        metadata = {
            "name"         : name,
            "version"      : version,
            "model_type"   : type(model).__name__,
            "metrics"      : {
                k: round(float(v), 6)
                for k, v in result.test_metrics.items()
            },
            "best_params"  : result.best_params,
            "feature_names": result.feature_names,
            "n_features"   : len(result.feature_names),
            "path"         : str(path),
            "timestamp"    : version,
            "is_champion"  : is_champion,
        }

        meta_path = model_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        # ── Update registry ───────────────────────────────────
        if name not in self.registry_["models"]:
            self.registry_["models"][name] = []

        self.registry_["models"][name].append(metadata)

        if is_champion:
            self.registry_["champion"] = {
                "name"   : name,
                "version": version,
                "path"   : str(path),
            }

        self._save_registry()

        logger.success(f"Model saved: {path}")
        logger.info(f"  Name    : {name}")
        logger.info(f"  Version : {version}")
        logger.info(
            f"  Metrics : "
            + ", ".join(
                f"{k}={v:.4f}"
                for k, v in result.test_metrics.items()
            )
        )

        return str(path)

    # ── LOAD ─────────────────────────────────────────────────

    def load(
        self,
        name   : str,
        version: str = "latest",
    ) -> Any:
        """
        Load model dari registry.

        Args:
            name    : Nama model (misal 'lightgbm')
            version : 'latest' / 'champion' / 'v_20240101_120000'

        Returns:
            Model object
        """
        path = self._resolve_path(name, version)
        logger.info(f"Loading model dari: {path}")

        model = self._load_model_file(path)
        logger.success(f"Model loaded: {type(model).__name__}")
        return model

    def load_metadata(
        self,
        name   : str,
        version: str = "latest",
    ) -> dict:
        """Load metadata model tanpa load model itu sendiri."""
        path     = Path(self._resolve_path(name, version))
        meta_path = path.parent / "metadata.json"

        if not meta_path.exists():
            raise ModelingError(f"Metadata tidak ditemukan: {meta_path}")

        with open(meta_path) as f:
            return json.load(f)

    # ── LISTING ───────────────────────────────────────────────

    def list_models(self) -> pd.DataFrame:
        """
        List semua model di registry sebagai DataFrame.

        Returns:
            DataFrame dengan kolom: name, version, model_type,
            metrics, is_champion, timestamp
        """
        rows = []
        for name, versions in self.registry_["models"].items():
            for v in versions:
                metrics_str = ", ".join(
                    f"{k}={val:.4f}"
                    for k, val in v.get("metrics", {}).items()
                )
                rows.append({
                    "name"       : name,
                    "version"    : v["version"],
                    "model_type" : v.get("model_type", ""),
                    "metrics"    : metrics_str,
                    "n_features" : v.get("n_features", ""),
                    "is_champion": v.get("is_champion", False),
                    "timestamp"  : v["timestamp"],
                })

        if not rows:
            logger.info("Registry kosong — belum ada model yang disimpan")
            return pd.DataFrame()

        df = pd.DataFrame(rows).sort_values(
            "timestamp", ascending=False
        )
        print(df.to_string(index=False))
        return df

    def get_champion(self) -> Optional[dict]:
        """Return metadata champion model saat ini."""
        return self.registry_.get("champion")

    # ── INTERNAL ─────────────────────────────────────────────

    def _save_model_file(
        self,
        model     ,
        model_type: str,
        model_dir : Path,
    ) -> Path:
        """Pilih format save yang tepat berdasarkan tipe model."""

        if any(t in model_type for t in ["lgbm", "lightgbm"]):
            path = model_dir / "model.txt"
            if hasattr(model, "booster_"):
                model.booster_.save_model(str(path))
            else:
                model.save_model(str(path))

        elif "catboost" in model_type:
            path = model_dir / "model.cbm"
            model.save_model(str(path))

        elif any(t in model_type for t in ["xgb", "xgboost"]):
            path = model_dir / "model.json"
            model.save_model(str(path))

        else:
            path = model_dir / "model.joblib"
            joblib.dump(model, str(path), compress=3)

        return path

    def _load_model_file(self, path: str) -> Any:
        """Load model dari path berdasarkan ekstensi."""
        p = Path(path)

        if p.suffix == ".txt":
            import lightgbm as lgb
            return lgb.Booster(model_file=str(p))

        elif p.suffix == ".cbm":
            from catboost import CatBoostClassifier
            m = CatBoostClassifier()
            m.load_model(str(p))
            return m

        elif p.suffix == ".json":
            import xgboost as xgb
            m = xgb.XGBClassifier()
            m.load_model(str(p))
            return m

        else:
            return joblib.load(str(p))

    def _resolve_path(self, name: str, version: str) -> str:
        """Resolve version string ke path aktual."""

        if version == "champion":
            champ = self.registry_.get("champion")
            if champ is None:
                raise ModelingError("Belum ada champion model")
            return champ["path"]

        if name not in self.registry_["models"]:
            available = list(self.registry_["models"].keys())
            raise ModelingError(
                f"Model '{name}' tidak ada di registry",
                details={"available": available},
            )

        versions = self.registry_["models"][name]

        if version == "latest":
            return versions[-1]["path"]

        matches = [v for v in versions if v["version"] == version]
        if not matches:
            avail = [v["version"] for v in versions]
            raise ModelingError(
                f"Version '{version}' tidak ditemukan untuk '{name}'",
                details={"available_versions": avail},
            )
        return matches[0]["path"]

    def _load_registry(self):
        """Load atau buat registry file."""
        reg_path = Path(self.REGISTRY_FILE)
        reg_path.parent.mkdir(parents=True, exist_ok=True)

        if reg_path.exists():
            with open(reg_path) as f:
                self.registry_ = json.load(f)
        else:
            self.registry_ = {"models": {}, "champion": None}

    def _save_registry(self):
        with open(self.REGISTRY_FILE, "w") as f:
            json.dump(self.registry_, f, indent=2, default=str)