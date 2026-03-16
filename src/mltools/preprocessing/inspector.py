# src/data/inspector.py

import pandas as pd
import numpy as np
from loguru import logger
from mltools.shared.exceptions import DataError
from IPython.display import display


class DataInspector:
    """
    Inspeksi data komprehensif — Expert Level.
    Memberikan gambaran lengkap kondisi data sebelum preprocessing.
    """

    def __init__(self, df: pd.DataFrame, target: str = None):
        self.df = df
        self.target = target
        self._report = {}

    def full_report(self) -> dict:
        """Jalankan semua inspeksi sekaligus.

        Example
        -------
        >>> inspector = DataInspector(df, target="churn")
        >>> report = inspector.full_report()
        >>> print(report["basic"])
        """
        logger.info("=" * 60)
        logger.info("MEMULAI FULL DATA INSPECTION")
        logger.info("=" * 60)
        self.basic_info()
        self.dtype_analysis()
        self.missing_analysis()
        self.duplicate_analysis()
        self.cardinality_analysis()
        self.statistical_summary()
        if self.target:
            self.target_analysis()
        logger.info("=" * 60)
        logger.info("INSPEKSI SELESAI")
        logger.info("=" * 60)
        return self._report

    def basic_info(self):
        """Informasi dasar dataset.

        Example
        -------
        >>> inspector = DataInspector(df)
        >>> info = inspector.basic_info()
        >>> print(info["shape"])        # (10000, 25)
        """
        print("\n" + "━" * 60)
        print("📋 BASIC INFORMATION")
        print("━" * 60)
        info = {
            "shape"           : self.df.shape,
            "n_rows"          : self.df.shape[0],
            "n_cols"          : self.df.shape[1],
            "memory_usage_mb" : round(self.df.memory_usage(deep=True).sum() / 1e6, 2),
            "dtypes_count"    : self.df.dtypes.value_counts().to_dict(),
        }
        for k, v in info.items():
            print(f"  {k:25s}: {v}")
        self._report["basic"] = info
        return info

    def dtype_analysis(self):
        """Analisis tipe data setiap kolom.

        Example
        -------
        >>> inspector = DataInspector(df)
        >>> dtype_df = inspector.dtype_analysis()
        """
        print("\n" + "━" * 60)
        print("🔢 DTYPE ANALYSIS")
        print("━" * 60)
        dtype_df = pd.DataFrame({
            "column"     : self.df.columns,
            "dtype"      : self.df.dtypes.values,
            "pandas_type": [self._classify_dtype(d) for d in self.df.dtypes],
            "memory_mb"  : [
                round(self.df[c].memory_usage(deep=True) / 1e6, 4)
                for c in self.df.columns
            ],
        })
        display(dtype_df)
        self._report["dtypes"] = dtype_df
        return dtype_df

    def missing_analysis(self):
        """Analisis nilai hilang.

        Example
        -------
        >>> inspector = DataInspector(df)
        >>> missing = inspector.missing_analysis()
        >>> print(missing["missing_pct"].max())   # kolom paling banyak missing
        """
        print("\n" + "━" * 60)
        print("❓ MISSING VALUES ANALYSIS")
        print("━" * 60)
        missing = pd.DataFrame({
            "column"       : self.df.columns,
            "missing_count": self.df.isnull().sum().values,
            "missing_pct"  : (self.df.isnull().sum() / len(self.df) * 100).round(2).values,
            "dtype"        : self.df.dtypes.values,
        }).sort_values("missing_pct", ascending=False)
        missing = missing[missing["missing_count"] > 0]
        if missing.empty:
            print("  ✅ Tidak ada missing values!")
        else:
            print(f"  ⚠  {len(missing)} kolom memiliki missing values")
            display(missing)
        total_cells = self.df.shape[0] * self.df.shape[1]
        total_missing = self.df.isnull().sum().sum()
        print(
            f"\n  Total missing cells : {total_missing:,} / {total_cells:,} "
            f"({total_missing / total_cells * 100:.2f}%)"
        )
        self._report["missing"] = missing
        return missing

    def duplicate_analysis(self):
        """Analisis duplikat.

        Example
        -------
        >>> inspector = DataInspector(df)
        >>> inspector.duplicate_analysis()
        """
        print("\n" + "━" * 60)
        print("🔁 DUPLICATE ANALYSIS")
        print("━" * 60)
        n_dup_rows = self.df.duplicated().sum()
        n_dup_pct = n_dup_rows / len(self.df) * 100
        print(f"  Duplicate rows : {n_dup_rows:,} ({n_dup_pct:.2f}%)")
        # Cek duplikat per kolom
        dup_cols = {}
        for col in self.df.columns:
            n = self.df[col].duplicated().sum()
            dup_cols[col] = n
        self._report["duplicates"] = {
            "n_dup_rows" : n_dup_rows,
            "pct_dup_rows": n_dup_pct,
            "dup_cols"   : dup_cols,
        }

    def cardinality_analysis(self):
        """Analisis kardinalitas (unique values).

        Example
        -------
        >>> inspector = DataInspector(df)
        >>> cardinality = inspector.cardinality_analysis()
        """
        print("\n" + "━" * 60)
        print("📊 CARDINALITY ANALYSIS")
        print("━" * 60)
        cardinality = pd.DataFrame({
            "column"     : self.df.columns,
            "n_unique"   : [self.df[c].nunique() for c in self.df.columns],
            "unique_pct" : [
                round(self.df[c].nunique() / len(self.df) * 100, 2)
                for c in self.df.columns
            ],
            "dtype"      : self.df.dtypes.values,
            "cardinality": [
                self._classify_cardinality(self.df[c]) for c in self.df.columns
            ],
        }).sort_values("n_unique", ascending=False)
        display(cardinality)
        self._report["cardinality"] = cardinality
        return cardinality

    def statistical_summary(self):
        """Summary statistik extended.

        Example
        -------
        >>> inspector = DataInspector(df)
        >>> inspector.statistical_summary()   # prints describe + skewness + kurtosis
        """
        print("\n" + "━" * 60)
        print("📈 STATISTICAL SUMMARY")
        print("━" * 60)
        num_cols = self.df.select_dtypes(include=np.number).columns
        if len(num_cols) > 0:
            stats = self.df[num_cols].describe(
                percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
            )
            stats.loc["skewness"] = self.df[num_cols].skew()
            stats.loc["kurtosis"] = self.df[num_cols].kurtosis()
            display(stats.round(4))
            self._report["stats"] = stats

    def target_analysis(self):
        """Analisis target variable.

        Example
        -------
        >>> inspector = DataInspector(df, target="churn")
        >>> inspector.target_analysis()   # prints class dist & imbalance ratio
        """
        print("\n" + "━" * 60)
        print(f"🎯 TARGET VARIABLE ANALYSIS: '{self.target}'")
        print("━" * 60)
        y = self.df[self.target]
        if y.dtype == "object" or y.nunique() < 20:
            # Klasifikasi
            print("  Task Type: CLASSIFICATION")
            vc = y.value_counts()
            print(f"  Class Distribution:\n{vc}")
            imbalance_ratio = vc.max() / vc.min()
            print(f"\n  Imbalance Ratio: {imbalance_ratio:.2f}x")
            if imbalance_ratio > 10:
                print("  ⚠  SEVERE IMBALANCE — Perlu SMOTE / Class Weights!")
            elif imbalance_ratio > 3:
                print("  ⚠  MODERATE IMBALANCE — Pertimbangkan resampling")
        else:
            # Regresi
            print("  Task Type: REGRESSION")
            print(f"  Mean     : {y.mean():.4f}")
            print(f"  Std      : {y.std():.4f}")
            print(f"  Skewness : {y.skew():.4f}")
            if abs(y.skew()) > 1:
                print("  ⚠  Target SKEWED — Pertimbangkan log/sqrt transform!")

    @staticmethod
    def _classify_dtype(dtype) -> str:
        dtype_str = str(dtype)
        if "int"      in dtype_str: return "integer"
        if "float"    in dtype_str: return "float"
        if "object"   in dtype_str: return "string/mixed"
        if "bool"     in dtype_str: return "boolean"
        if "datetime" in dtype_str: return "datetime"
        if "category" in dtype_str: return "category"
        return "other"

    @staticmethod
    def _classify_cardinality(series: pd.Series) -> str:
        n_unique = series.nunique()
        n_total  = len(series)
        ratio    = n_unique / n_total
        if n_unique == 1    : return "CONSTANT ⛔"
        if n_unique == 2    : return "BINARY ✓"
        if n_unique <= 10   : return "LOW CARD ✓"
        if n_unique <= 50   : return "MED CARD ~"
        if ratio    > 0.95  : return "QUASI-ID ⚠"
        return "HIGH CARD ⚠"


# ── Contoh Penggunaan ─────────────────────────────────────────────────────────
# inspector = DataInspector(df, target="churn")
# report = inspector.full_report()