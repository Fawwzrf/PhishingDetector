# src/mltools/interpretation/shap_analysis.py

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from loguru import logger

from mltools.shared.exceptions import ModelingError
from mltools.shared.schemas    import DataSplit, TrainingResult


class SHAPAnalyzer:
    """
    SHAP analysis expert-grade untuk semua jenis model.

    Fitur:
    - Auto-detect explainer yang tepat (TreeExplainer, LinearExplainer, KernelExplainer)
    - Global importance: bar plot + beeswarm
    - Local explanation: waterfall per sampel
    - Dependence plot: hubungan satu fitur dengan SHAP-nya
    - Export importance ke DataFrame

    Cara pakai:
        analyzer = SHAPAnalyzer.from_result(result, split)
        analyzer.compute()
        analyzer.full_analysis()
    """

    def __init__(
        self,
        model,
        X_background  : pd.DataFrame,
        feature_names : Optional[List[str]] = None,
        model_type    : str  = "auto",
        n_background  : int  = 200,
        task          : str  = "classification",
    ):
        self.task          = task
        self.feature_names = (
            feature_names or list(X_background.columns)
        )
        self.shap_values_  = None
        self.X_explained_  = None

        # ── Auto-detect explainer ─────────────────────────────
        if model_type == "auto":
            name = type(model).__name__.lower()
            if any(k in name for k in
                   ["forest", "tree", "xgb", "lgbm", "lgb",
                    "catboost", "gradient", "boost"]):
                model_type = "tree"
            elif any(k in name for k in
                     ["linear", "ridge", "lasso", "logistic"]):
                model_type = "linear"
            else:
                model_type = "kernel"

        logger.info(f"SHAP Explainer: {model_type}")
        self.model_type_ = model_type

        background = shap.sample(X_background, n_background)

        if model_type == "tree":
            self.explainer_ = shap.TreeExplainer(
                model,
                data                = background,
                feature_perturbation= "interventional",
            )
        elif model_type == "linear":
            self.explainer_ = shap.LinearExplainer(
                model, background,
                feature_perturbation="correlation_dependent",
            )
        else:
            predict_fn = (
                model.predict_proba
                if hasattr(model, "predict_proba")
                else model.predict
            )
            self.explainer_ = shap.KernelExplainer(
                predict_fn,
                shap.sample(background, min(50, len(background))),
            )

    @classmethod
    def from_result(
        cls,
        result : TrainingResult,
        split  : DataSplit,
        **kwargs,
    ) -> "SHAPAnalyzer":
        """
        Buat SHAPAnalyzer langsung dari TrainingResult + DataSplit.

        Cara pakai:
            analyzer = SHAPAnalyzer.from_result(result, split)
        """
        return cls(
            model         = result.champion_model,
            X_background  = split.X_train,
            feature_names = split.feature_names,
            task          = split.task,
            **kwargs,
        )

    # ── COMPUTE ───────────────────────────────────────────────

    def compute(
        self,
        X          : pd.DataFrame,
        max_samples: int = 2000,
    ) -> np.ndarray:
        """
        Hitung SHAP values untuk dataset X.

        Args:
            X           : DataFrame untuk dijelaskan
            max_samples : Subsample jika terlalu besar

        Returns:
            ndarray shape (n_samples, n_features)
        """
        if len(X) > max_samples:
            logger.warning(
                f"Subsampling dari {len(X):,} ke {max_samples:,} "
                "untuk efisiensi SHAP"
            )
            X = X.sample(max_samples, random_state=42)

        logger.info(f"Computing SHAP values untuk {len(X):,} samples...")

        raw = self.explainer_.shap_values(X)

        # Normalisasi format → selalu (n_samples, n_features)
        if isinstance(raw, list):
            # Binary → ambil kelas positif [1]
            if len(raw) == 2:
                shap_values = raw[1]
            else:
                # Multi-class → rata-rata absolut per kelas
                shap_values = np.mean(
                    [np.abs(s) for s in raw], axis=0
                )
        else:
            shap_values = raw
            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]

        self.shap_values_ = np.array(shap_values)
        self.X_explained_ = X

        logger.success(
            f"SHAP values computed: {self.shap_values_.shape}"
        )
        return self.shap_values_

    # ── VISUALISASI ───────────────────────────────────────────

    def plot_importance(
        self,
        top_n: int  = 20,
        save : bool = True,
    ):
        """Bar plot: mean |SHAP| per fitur — Global Importance."""
        self._check_computed()
        Path("reports").mkdir(parents=True, exist_ok=True)

        shap.summary_plot(
            self.shap_values_,
            self.X_explained_,
            feature_names = self.feature_names,
            plot_type     = "bar",
            max_display   = top_n,
            show          = False,
        )
        plt.title(
            f"Global Feature Importance (SHAP) — Top {top_n}\n"
            "Mean |SHAP| across all samples"
        )
        if save:
            plt.savefig(
                "reports/shap_importance.png",
                dpi=150, bbox_inches="tight",
            )
        plt.show()

    def plot_beeswarm(
        self,
        top_n: int  = 20,
        save : bool = True,
    ):
        """
        Beeswarm plot: distribusi SHAP per fitur.
        Lebih informatif dari bar plot.

        Cara baca:
        - Posisi X   : SHAP value (kanan = dorong prediksi naik)
        - Warna merah: nilai fitur tinggi
        - Warna biru : nilai fitur rendah
        """
        self._check_computed()

        shap.summary_plot(
            self.shap_values_,
            self.X_explained_,
            feature_names = self.feature_names,
            plot_type     = "dot",
            max_display   = top_n,
            alpha         = 0.5,
            show          = False,
        )
        plt.title(
            f"SHAP Beeswarm — Top {top_n}\n"
            "Merah=nilai tinggi | Biru=nilai rendah"
        )
        if save:
            plt.savefig(
                "reports/shap_beeswarm.png",
                dpi=150, bbox_inches="tight",
            )
        plt.show()

    def plot_waterfall(
        self,
        sample_idx: int  = 0,
        save      : bool = True,
    ):
        """
        Waterfall plot: jelaskan prediksi SATU sampel.

        Cara baca:
        - E[f(x)] = base value (rata-rata prediksi semua data)
        - f(x)    = prediksi untuk sampel ini
        - Bar merah  = fitur mendorong prediksi naik
        - Bar biru   = fitur mendorong prediksi turun
        """
        self._check_computed()

        base_val = self.explainer_.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = base_val[1] if len(base_val) == 2 else base_val[0]

        explanation = shap.Explanation(
            values        = self.shap_values_[sample_idx],
            base_values   = float(base_val),
            data          = self.X_explained_.iloc[sample_idx].values,
            feature_names = self.feature_names,
        )

        shap.waterfall_plot(explanation, max_display=15, show=False)
        plt.title(
            f"Local Explanation — Sample #{sample_idx}\n"
            "Mengapa model prediksi nilai ini?"
        )
        if save:
            plt.savefig(
                f"reports/shap_waterfall_{sample_idx}.png",
                dpi=150, bbox_inches="tight",
            )
        plt.show()

    def plot_dependence(
        self,
        feature    : str,
        interaction: str  = "auto",
        save       : bool = True,
    ):
        """
        Dependence plot: hubungan satu fitur dengan SHAP value-nya.

        Cara baca:
        - X axis: nilai fitur
        - Y axis: SHAP value (kontribusi ke prediksi)
        - Warna : fitur interaksi terkuat
        """
        self._check_computed()

        if feature not in self.X_explained_.columns:
            raise ModelingError(
                f"Fitur '{feature}' tidak ada di data",
                details={"available": list(self.X_explained_.columns)},
            )

        shap.dependence_plot(
            feature,
            self.shap_values_,
            self.X_explained_,
            interaction_index = interaction,
            alpha             = 0.5,
            show              = False,
        )
        plt.title(
            f"SHAP Dependence: {feature}\n"
            "Warna = fitur interaksi terkuat"
        )
        if save:
            plt.savefig(
                f"reports/shap_dependence_{feature}.png",
                dpi=150, bbox_inches="tight",
            )
        plt.show()

    # ── EXPORT ───────────────────────────────────────────────

    def get_importance_df(self) -> pd.DataFrame:
        """
        Return DataFrame feature importance berdasarkan mean |SHAP|.

        Returns:
            DataFrame dengan kolom: feature, mean_abs_shap,
            mean_shap, positive_impact_pct
        """
        self._check_computed()

        mean_abs = np.abs(self.shap_values_).mean(axis=0)
        mean_shap = self.shap_values_.mean(axis=0)
        pos_pct  = (self.shap_values_ > 0).mean(axis=0) * 100

        df = pd.DataFrame({
            "feature"          : self.feature_names,
            "mean_abs_shap"    : mean_abs,
            "mean_shap"        : mean_shap,
            "positive_impact_pct": pos_pct,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        df["rank"] = range(1, len(df) + 1)
        return df

    # ── FULL ANALYSIS ─────────────────────────────────────────

    def full_analysis(
        self,
        X    : pd.DataFrame,
        top_n: int = 20,
    ):
        """
        Jalankan semua visualisasi SHAP sekaligus.
        Simpan semua ke reports/.

        Args:
            X     : DataFrame untuk dijelaskan (biasanya X_test)
            top_n : jumlah fitur teratas yang ditampilkan
        """
        self.compute(X)

        logger.info("Running full SHAP analysis...")
        Path("reports").mkdir(parents=True, exist_ok=True)

        self.plot_importance(top_n)
        self.plot_beeswarm(top_n)

        # Waterfall untuk sampel pertama, tengah, terakhir
        n = len(self.X_explained_)
        for idx in [0, n // 2, n - 1]:
            self.plot_waterfall(sample_idx=idx)

        # Dependence untuk top 3 fitur
        importance = self.get_importance_df()
        for feat in importance["feature"].head(3):
            try:
                self.plot_dependence(feat)
            except Exception as e:
                logger.warning(f"Dependence plot gagal untuk {feat}: {e}")

        # Export importance ke CSV
        imp_path = "reports/shap_importance.csv"
        importance.to_csv(imp_path, index=False)
        logger.success(f"SHAP importance saved: {imp_path}")
        logger.success("Full SHAP analysis selesai! Check reports/")

    # ── HELPERS ───────────────────────────────────────────────

    def _check_computed(self):
        if self.shap_values_ is None:
            raise ModelingError(
                "SHAP values belum dihitung. "
                "Panggil .compute(X) dulu."
            )