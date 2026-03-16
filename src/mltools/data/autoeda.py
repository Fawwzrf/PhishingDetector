# src/mltools/data/autoeda.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


def generate_eda_report(
    df          : pd.DataFrame,
    title       : str           = "EDA Report",
    output_file : str           = "reports/eda_report.html",
    target_col  : Optional[str] = None,
    minimal     : bool          = False,
    dark_mode   : bool          = False,
    explorative : bool          = True,
) -> object:
    """
    Generate laporan EDA HTML komprehensif menggunakan ydata-profiling.

    Laporan mencakup:
    - Overview dataset (shape, memory, missing, duplicates)
    - Distribusi setiap variabel + statistik lengkap
    - Analisis korelasi (Pearson, Spearman, Kendall, Phik)
    - Deteksi missing values dan pola
    - Peringatan otomatis (high cardinality, skewed, dll.)
    - Interaksi antar variabel (jika explorative=True)

    Args:
        df          : DataFrame yang akan dianalisis
        title       : Judul laporan HTML
        output_file : Path output file HTML
        target_col  : Kolom target untuk analisis tambahan
        minimal     : True untuk dataset > 500K baris (lebih cepat)
        dark_mode   : True untuk tema gelap
        explorative : True untuk analisis interaksi antar variabel

    Returns:
        ProfileReport object

    Cara pakai:
        profile = generate_eda_report(
            df,
            title      = "Phishing Detection EDA",
            output_file= "reports/eda_report.html",
            target_col = "phishing",
        )
        # Buka reports/eda_report.html di browser
    """

    try:
        from ydata_profiling import ProfileReport
    except ImportError:
        raise ImportError(
            "ydata-profiling belum terinstall. "
            "Jalankan: pip install ydata-profiling"
        )

    # Buat direktori output jika belum ada
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating EDA report: '{title}'")
    logger.info(f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} cols")
    logger.info(f"Mode: {'minimal' if minimal else 'full'}")

    # Build config
    profile_config = {
        "title"      : title,
        "explorative": explorative,
        "dark_mode"  : dark_mode,
        "minimal"    : minimal,
        "samples"    : {
            "head"  : 10,
            "tail"  : 10,
        },
        "correlations": {
            "auto"   : {"calculate": True},
            "pearson": {"calculate": True},
            "spearman": {"calculate": not minimal},
            "kendall": {"calculate": not minimal},
            "phi_k"  : {"calculate": not minimal},
        },
    }

    # Target column config
    if target_col and target_col in df.columns:
        profile_config["y_axis"] = target_col
        logger.info(f"Target column: {target_col}")

    # Generate report
    profile = ProfileReport(df, **profile_config)

    # Save ke HTML
    profile.to_file(output_file)
    size_mb = output_path.stat().st_size / 1e6

    logger.success(
        f"EDA report saved: {output_file} ({size_mb:.1f} MB)\n"
        f"Buka di browser: file:///{output_path.resolve()}"
    )

    return profile


def quick_eda(df: pd.DataFrame, target: Optional[str] = None) -> None:
    """
    EDA cepat di terminal — tanpa HTML, langsung ke console.
    Berguna saat eksplorasi awal sebelum generate full report.

    Args:
        df     : DataFrame
        target : Nama kolom target (opsional)
    """
    print("\n" + "═" * 60)
    print("  QUICK EDA SUMMARY")
    print("═" * 60)

    # Basic info
    print(f"\n  Shape     : {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"  Memory    : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"  Dtypes    : {dict(df.dtypes.value_counts())}")

    # Missing
    n_missing = df.isnull().sum().sum()
    if n_missing > 0:
        miss_pct  = n_missing / (df.shape[0] * df.shape[1]) * 100
        miss_cols = df.columns[df.isnull().any()].tolist()
        print(f"\n  Missing   : {n_missing:,} cells ({miss_pct:.2f}%)")
        print(f"  Cols w/NA : {len(miss_cols)}")
        # Top 5 missing
        top_miss = df.isnull().mean().sort_values(ascending=False).head(5)
        for col, pct in top_miss.items():
            if pct > 0:
                print(f"    {col:<35}: {pct*100:.1f}%")
    else:
        print("\n  Missing   : Tidak ada")

    # Duplicates
    n_dup = df.duplicated().sum()
    print(f"\n  Duplicates: {n_dup:,} ({n_dup/len(df)*100:.2f}%)")

    # Constant cols
    const_cols = [c for c in df.columns if df[c].nunique() <= 1]
    if const_cols:
        print(f"\n  Constant cols ({len(const_cols)}): {const_cols}")

    # High cardinality
    high_card = [c for c in df.select_dtypes("object").columns
                 if df[c].nunique() / len(df) > 0.9]
    if high_card:
        print(f"\n  Quasi-ID cols: {high_card}")

    # Numeric stats
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        skewed = df[num_cols].skew()
        high_skew = skewed[skewed.abs() > 2]
        if len(high_skew) > 0:
            print(f"\n  Highly skewed (|skew|>2): {len(high_skew)} cols")
            for col, sk in high_skew.sort_values(key=abs, ascending=False).head(5).items():
                print(f"    {col:<35}: {sk:.2f}")

    # Target analysis
    if target and target in df.columns:
        y  = df[target]
        vc = y.value_counts()
        print(f"\n  Target: {target}")
        for cls, cnt in vc.items():
            print(f"    Class {cls}: {cnt:,} ({cnt/len(y)*100:.1f}%)")
        if y.nunique() > 1:
            imb = vc.max() / vc.min()
            level = "SEVERE" if imb > 10 else "MODERATE" if imb > 3 else "OK"
            print(f"    Imbalance: {imb:.1f}x [{level}]")

    print("\n" + "═" * 60)