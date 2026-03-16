# src/mltools/data/eda.py

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from mltools.shared.exceptions import DataError


class EDAVisualizer:
    """
    Visualisasi EDA manual untuk analisis mendalam.

    Melengkapi Auto EDA dengan visualisasi yang bisa
    di-customize dan disimpan secara individual.

    Cara pakai:
        eda = EDAVisualizer(df, target="phishing")
        eda.run_full_eda()        # Jalankan semua, simpan ke reports/

        # Atau per visualisasi:
        eda.plot_missing_heatmap()
        eda.plot_distributions()
        eda.plot_correlation_matrix()
    """

    def __init__(
        self,
        df          : pd.DataFrame,
        target      : Optional[str] = None,
        report_dir  : str           = "reports",
        figsize_base: tuple         = (14, 8),
    ):
        self.df          = df
        self.target      = target
        self.report_dir  = Path(report_dir)
        self.figsize_base= figsize_base

        self.report_dir.mkdir(parents=True, exist_ok=True)

        # Pisahkan tipe kolom
        self._num_cols = df.select_dtypes(include=np.number).columns.tolist()
        self._cat_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        if target and target in self._num_cols:
            self._num_cols.remove(target)
        if target and target in self._cat_cols:
            self._cat_cols.remove(target)

        logger.info(
            f"EDAVisualizer init: {df.shape} | "
            f"num={len(self._num_cols)} cat={len(self._cat_cols)} "
            f"target={target}"
        )

    # ── MISSING VALUES ────────────────────────────────────────

    def plot_missing_heatmap(self, save: bool = True) -> None:
        """
        Heatmap pola missing values menggunakan missingno-style.
        Menunjukkan apakah missing terjadi bersama-sama (MNAR clue).
        """
        missing_cols = self.df.columns[self.df.isnull().any()].tolist()

        if not missing_cols:
            logger.info("Tidak ada missing values — skip heatmap")
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # ── Bar chart % missing ───────────────────────────────
        miss_pct = (self.df[missing_cols].isnull().mean() * 100
                   ).sort_values(ascending=True)
        colors   = ["#E74C3C" if v > 50 else "#F39C12" if v > 20 else "#3498DB"
                   for v in miss_pct]
        axes[0].barh(miss_pct.index, miss_pct.values, color=colors)
        axes[0].set_xlabel("% Missing")
        axes[0].set_title(
            f"Missing Values per Kolom\n"
            f"({len(missing_cols)} kolom dari {self.df.shape[1]} total)"
        )
        for i, v in enumerate(miss_pct.values):
            axes[0].text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)

        # ── Heatmap korelasi missing pattern ─────────────────
        miss_df  = self.df[missing_cols].isnull().astype(int)
        corr     = miss_df.corr()
        mask     = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, ax=axes[1], mask=mask,
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            annot=len(missing_cols) <= 15,
            fmt=".1f", linewidths=0.5,
        )
        axes[1].set_title(
            "Korelasi Pola Missing\n"
            "(Merah=sering hilang bersamaan → MNAR clue)"
        )

        plt.tight_layout()
        if save:
            path = self.report_dir / "missing_heatmap.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved: {path}")
        plt.show()

    # ── DISTRIBUSI ────────────────────────────────────────────

    def plot_distributions(
        self,
        cols    : Optional[List[str]] = None,
        max_cols: int                 = 24,
        save    : bool                = True,
    ) -> None:
        """
        Histogram distribusi semua fitur numerik.
        Warna merah = skewness tinggi (butuh transformasi).
        """
        cols = cols or self._num_cols
        cols = cols[:max_cols]

        if not cols:
            logger.info("Tidak ada kolom numerik")
            return

        n_cols = 4
        n_rows = (len(cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(16, n_rows * 3.2)
        )
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes

        for i, col in enumerate(cols):
            ax      = axes[i]
            data    = self.df[col].dropna()
            skew    = data.skew()
            color   = "#E74C3C" if abs(skew) > 2 else (
                      "#F39C12" if abs(skew) > 1 else "#3498DB")

            ax.hist(data, bins=40, color=color, alpha=0.75, edgecolor="white")
            ax.set_title(f"{col}\nskew={skew:.2f}", fontsize=8)
            ax.tick_params(labelsize=7)

        # Sembunyikan axis kosong
        for j in range(len(cols), len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(
            f"Distribusi Fitur Numerik (n={len(cols)})\n"
            "Merah=skew>2 | Orange=skew>1 | Biru=OK",
            fontsize=12
        )
        plt.tight_layout()
        if save:
            path = self.report_dir / "distributions.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved: {path}")
        plt.show()

    # ── CORRELATION MATRIX ────────────────────────────────────

    def plot_correlation_matrix(
        self,
        threshold   : float = 0.7,
        max_features: int   = 40,
        save        : bool  = True,
    ) -> None:
        """
        Heatmap korelasi Pearson.
        Highlight pasangan dengan korelasi > threshold.
        """
        num_cols = self._num_cols[:max_features]
        if len(num_cols) < 2:
            logger.info("Terlalu sedikit kolom numerik untuk korelasi")
            return

        corr = self.df[num_cols].corr()

        # Hitung jumlah pasangan berkorelasi tinggi
        mask     = np.triu(np.ones_like(corr, dtype=bool))
        high_corr= (corr.abs().where(~mask).fillna(0) > threshold)
        n_high   = high_corr.sum().sum()

        size = max(10, len(num_cols) * 0.4)
        fig, ax = plt.subplots(figsize=(size, size * 0.85))

        annot = len(num_cols) <= 20
        sns.heatmap(
            corr,
            ax       = ax,
            mask     = mask,
            cmap     = "RdBu_r",
            center   = 0,
            vmin     = -1,
            vmax     = 1,
            annot    = annot,
            fmt      = ".1f",
            linewidths= 0.3,
            square   = True,
        )
        ax.set_title(
            f"Correlation Matrix — {len(num_cols)} fitur\n"
            f"Pasangan korelasi |r| > {threshold}: {n_high}",
            fontsize=12
        )

        plt.tight_layout()
        if save:
            path = self.report_dir / "correlation_matrix.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved: {path}")
        plt.show()

        # Print pasangan yang sangat berkorelasi
        if n_high > 0:
            logger.warning(
                f"Ditemukan {n_high} pasangan fitur dengan |r| > {threshold}:"
            )
            for c1 in corr.columns:
                for c2 in corr.columns:
                    if c1 < c2:
                        r = corr.loc[c1, c2]
                        if abs(r) > threshold:
                            logger.warning(
                                f"  {c1} ↔ {c2}: r={r:.4f}"
                            )

    # ── CATEGORICAL ───────────────────────────────────────────

    def plot_categorical_distributions(
        self,
        cols    : Optional[List[str]] = None,
        max_cols: int                 = 12,
        top_n   : int                 = 15,
        save    : bool                = True,
    ) -> None:
        """Bar chart distribusi fitur kategorikal."""
        cols = cols or self._cat_cols
        cols = cols[:max_cols]

        if not cols:
            logger.info("Tidak ada kolom kategorikal")
            return

        n_cols = 3
        n_rows = (len(cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(16, n_rows * 4)
        )
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for i, col in enumerate(cols):
            ax = axes[i]
            vc = self.df[col].value_counts().head(top_n)
            vc.plot(kind="barh", ax=ax, color="#3498DB", edgecolor="white")
            ax.set_title(f"{col}\n({self.df[col].nunique()} unique)", fontsize=9)
            ax.tick_params(labelsize=7)

        for j in range(len(cols), len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(f"Distribusi Fitur Kategorikal (n={len(cols)})", fontsize=12)
        plt.tight_layout()
        if save:
            path = self.report_dir / "categorical_distributions.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved: {path}")
        plt.show()

    # ── TARGET ANALYSIS ───────────────────────────────────────

    def plot_target_vs_features(
        self,
        cols    : Optional[List[str]] = None,
        max_cols: int                 = 16,
        save    : bool                = True,
    ) -> None:
        """
        Distribusi setiap fitur numerik per kelas target.
        Berguna untuk lihat fitur mana yang paling diskriminatif.
        """
        if not self.target:
            raise DataError("target harus diset di constructor")

        if self.target not in self.df.columns:
            raise DataError(f"Target '{self.target}' tidak ada di DataFrame")

        cols  = cols or self._num_cols
        cols  = [c for c in cols if c != self.target][:max_cols]
        y     = self.df[self.target]
        cats  = sorted(y.unique())
        n_cats= len(cats)

        palette = sns.color_palette("Set2", n_cats)

        n_cols = 4
        n_rows = (len(cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(16, n_rows * 3.5)
        )
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for i, col in enumerate(cols):
            ax = axes[i]
            for j, cat in enumerate(cats):
                data = self.df.loc[y == cat, col].dropna()
                ax.hist(
                    data, bins=30, alpha=0.6,
                    color=palette[j], label=str(cat),
                    density=True
                )
            ax.set_title(col, fontsize=8)
            ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)

        for j in range(len(cols), len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(
            f"Distribusi per Kelas Target: '{self.target}'\n"
            f"(Top {len(cols)} fitur numerik)",
            fontsize=12
        )
        plt.tight_layout()
        if save:
            path = self.report_dir / "target_vs_features.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved: {path}")
        plt.show()

    def plot_target_distribution(self, save: bool = True) -> None:
        """Distribusi kolom target."""
        if not self.target:
            raise DataError("target harus diset di constructor")

        y  = self.df[self.target]
        vc = y.value_counts().sort_index()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Bar chart
        colors = sns.color_palette("Set2", len(vc))
        bars   = axes[0].bar(
            vc.index.astype(str), vc.values,
            color=colors, edgecolor="black", alpha=0.85
        )
        for bar, val in zip(bars, vc.values):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + vc.max() * 0.01,
                f"{val:,}\n({val/len(y)*100:.1f}%)",
                ha="center", fontsize=10
            )
        axes[0].set_title(f"Distribusi Target: {self.target}")
        axes[0].set_xlabel("Kelas")
        axes[0].set_ylabel("Jumlah")

        # Pie chart
        axes[1].pie(
            vc.values, labels=vc.index.astype(str),
            colors=colors, autopct="%1.1f%%",
            startangle=90
        )
        axes[1].set_title("Proporsi Kelas")

        imbalance = vc.max() / vc.min() if vc.min() > 0 else float("inf")
        if imbalance > 3:
            logger.warning(
                f"Imbalance ratio: {imbalance:.1f}x — "
                "pertimbangkan SMOTE atau class_weight"
            )
        else:
            logger.info(f"Imbalance ratio: {imbalance:.1f}x — cukup seimbang")

        plt.suptitle(
            f"Target: {self.target} | n={len(y):,} | "
            f"Imbalance: {imbalance:.1f}x",
            fontsize=12
        )
        plt.tight_layout()
        if save:
            path = self.report_dir / "target_distribution.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved: {path}")
        plt.show()

    # ── OUTLIER OVERVIEW ──────────────────────────────────────

    def plot_boxplots(
        self,
        cols    : Optional[List[str]] = None,
        max_cols: int                 = 20,
        save    : bool                = True,
    ) -> None:
        """Boxplot fitur numerik untuk identifikasi outlier visual."""
        cols  = cols or self._num_cols
        cols  = cols[:max_cols]

        if not cols:
            logger.info("Tidak ada kolom numerik")
            return

        n_cols = 5
        n_rows = (len(cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(18, n_rows * 3)
        )
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for i, col in enumerate(cols):
            data = self.df[col].dropna()
            axes[i].boxplot(data, vert=True, patch_artist=True,
                           boxprops=dict(facecolor="#3498DB", alpha=0.7))
            axes[i].set_title(col, fontsize=8)
            axes[i].tick_params(labelsize=7)

        for j in range(len(cols), len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(f"Boxplot — Identifikasi Outlier (n={len(cols)})", fontsize=12)
        plt.tight_layout()
        if save:
            path = self.report_dir / "boxplots.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved: {path}")
        plt.show()

    # ── FULL EDA ──────────────────────────────────────────────

    def run_full_eda(
        self,
        max_features: int = 24,
    ) -> None:
        """
        Jalankan semua visualisasi EDA sekaligus.
        Semua plot disimpan otomatis ke reports/.
        """
        logger.info("Running Full EDA Visualization...")
        logger.info(f"Output → {self.report_dir}/")

        self.plot_missing_heatmap()

        if self.target:
            self.plot_target_distribution()

        self.plot_distributions(max_cols=max_features)
        self.plot_boxplots(max_cols=max_features)
        self.plot_correlation_matrix()

        if self._cat_cols:
            self.plot_categorical_distributions()

        if self.target:
            self.plot_target_vs_features(max_cols=max_features)

        logger.success(
            f"Full EDA selesai! Semua plot tersimpan di {self.report_dir}/"
        )