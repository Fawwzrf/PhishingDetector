# src/mltools/data/loader.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, Iterator

import pandas as pd
import numpy as np
from loguru import logger

from mltools.shared.exceptions import DataError


class DataLoader:
    """
    Universal data loader untuk berbagai format file.

    Format yang didukung:
        CSV, Parquet, Excel (.xlsx/.xls), JSON,
        Feather, Pickle (.pkl/.pickle)

    Cara pakai:
        df = DataLoader.load("data/raw/dataset.csv")
        df = DataLoader.load("data/raw/dataset.csv", sample_frac=0.2)

        # Dataset sangat besar
        for chunk in DataLoader.load_chunks("data/raw/big.csv"):
            process(chunk)
    """

    # ── LOAD TUNGGAL ──────────────────────────────────────────

    @staticmethod
    def load(
        path         : str,
        sample_frac  : Optional[float] = None,
        sample_n     : Optional[int]   = None,
        random_state : int             = 42,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load file ke DataFrame.

        Args:
            path        : Path ke file data
            sample_frac : Ambil fraksi acak (0.0–1.0). None = load semua
            sample_n    : Ambil N baris acak. None = load semua
            random_state: Seed untuk sampling
            **kwargs    : Diteruskan ke pd.read_* function

        Returns:
            pd.DataFrame
        """
        path_obj = Path(path)

        if not path_obj.exists():
            raise DataError(
                f"File tidak ditemukan: {path}",
                details={"path": str(path_obj.resolve())}
            )

        ext    = path_obj.suffix.lower()
        logger.info(f"Loading data dari {path} (format: {ext})")

        loaders = {
            ".csv"    : lambda p, kw: pd.read_csv(p, **kw),
            ".parquet": lambda p, kw: pd.read_parquet(p, **kw),
            ".xlsx"   : lambda p, kw: pd.read_excel(p, **kw),
            ".xls"    : lambda p, kw: pd.read_excel(p, **kw),
            ".json"   : lambda p, kw: pd.read_json(p, **kw),
            ".feather" : lambda p, kw: pd.read_feather(p, **kw),
            ".pkl"    : lambda p, kw: pd.read_pickle(p),
            ".pickle" : lambda p, kw: pd.read_pickle(p),
        }

        if ext not in loaders:
            raise DataError(
                f"Format tidak didukung: {ext}",
                details={"supported": list(loaders.keys())}
            )

        df = loaders[ext](path, kwargs)

        # Sampling opsional
        if sample_frac is not None:
            if not 0.0 < sample_frac <= 1.0:
                raise DataError(
                    f"sample_frac harus antara 0 dan 1, dapat: {sample_frac}"
                )
            df = df.sample(frac=sample_frac, random_state=random_state)
            logger.info(f"Sampling: {sample_frac:.0%} = {len(df):,} baris")

        elif sample_n is not None:
            df = df.sample(n=min(sample_n, len(df)),
                          random_state=random_state)
            logger.info(f"Sampling: {len(df):,} baris")

        df = df.reset_index(drop=True)

        logger.success(
            f"Data loaded: {df.shape[0]:,} rows × {df.shape[1]} cols "
            f"| {df.memory_usage(deep=True).sum() / 1e6:.1f} MB"
        )
        return df

    # ── CHUNKED LOADING ───────────────────────────────────────

    @staticmethod
    def load_chunks(
        path      : str,
        chunksize : int = 50_000,
        **kwargs,
    ) -> Iterator[pd.DataFrame]:
        """
        Load file CSV besar secara chunk-by-chunk.
        Hemat RAM — tidak load semua ke memori sekaligus.

        Args:
            path      : Path ke CSV file
            chunksize : Jumlah baris per chunk

        Yields:
            pd.DataFrame per chunk

        Cara pakai:
            chunks = []
            for chunk in DataLoader.load_chunks("big.csv", chunksize=100_000):
                chunks.append(preprocess(chunk))
            df = pd.concat(chunks, ignore_index=True)
        """
        path_obj = Path(path)

        if not path_obj.exists():
            raise DataError(f"File tidak ditemukan: {path}")

        if path_obj.suffix.lower() != ".csv":
            raise DataError(
                "load_chunks hanya support CSV. "
                "Untuk Parquet pakai pd.read_parquet dengan filters."
            )

        logger.info(
            f"Chunked loading: {path} "
            f"(chunksize={chunksize:,})"
        )

        total_rows = 0
        for i, chunk in enumerate(
            pd.read_csv(path, chunksize=chunksize, **kwargs)
        ):
            total_rows += len(chunk)
            logger.debug(f"  Chunk {i+1}: {len(chunk):,} baris")
            yield chunk

        logger.success(f"Chunked load selesai: {total_rows:,} rows total")

    # ── LOAD LARGE (concat semua chunk) ───────────────────────

    @staticmethod
    def load_large(
        path      : str,
        chunksize : int            = 100_000,
        preprocess: callable       = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load CSV besar dengan concat semua chunk setelah opsional preprocessing.

        Args:
            path       : Path ke CSV file
            chunksize  : Baris per chunk
            preprocess : Fungsi opsional yang diapply ke setiap chunk
                         Signature: (chunk: pd.DataFrame) -> pd.DataFrame

        Returns:
            pd.DataFrame gabungan semua chunk
        """
        chunks = []
        for chunk in DataLoader.load_chunks(path, chunksize, **kwargs):
            if preprocess:
                chunk = preprocess(chunk)
            chunks.append(chunk)

        df = pd.concat(chunks, ignore_index=True)
        logger.success(
            f"Load large selesai: {df.shape[0]:,} rows × {df.shape[1]} cols"
        )
        return df

    # ── SAVE ──────────────────────────────────────────────────

    @staticmethod
    def save(
        df   : pd.DataFrame,
        path : str,
        **kwargs,
    ) -> None:
        """
        Simpan DataFrame ke file.
        Format ditentukan dari ekstensi file.

        Args:
            df   : DataFrame yang akan disimpan
            path : Path output file

        Cara pakai:
            DataLoader.save(df, "data/processed/clean.parquet")
            DataLoader.save(df, "data/processed/clean.csv", index=False)
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        ext = path_obj.suffix.lower()

        savers = {
            ".csv"    : lambda d, p, kw: d.to_csv(p, index=False, **kw),
            ".parquet": lambda d, p, kw: d.to_parquet(p, index=False, **kw),
            ".xlsx"   : lambda d, p, kw: d.to_excel(p, index=False, **kw),
            ".feather" : lambda d, p, kw: d.to_feather(p, **kw),
            ".pkl"    : lambda d, p, kw: d.to_pickle(p),
            ".pickle" : lambda d, p, kw: d.to_pickle(p),
        }

        if ext not in savers:
            raise DataError(
                f"Format save tidak didukung: {ext}",
                details={"supported": list(savers.keys())}
            )

        savers[ext](df, path, kwargs)
        size_mb = path_obj.stat().st_size / 1e6
        logger.success(f"Saved: {path} ({size_mb:.1f} MB)")

    # ── INFO ──────────────────────────────────────────────────

    @staticmethod
    def info(df: pd.DataFrame) -> dict:
        """
        Ringkasan cepat DataFrame.

        Returns:
            dict dengan shape, memory, dtypes, dll.
        """
        mem_mb = df.memory_usage(deep=True).sum() / 1e6
        info = {
            "shape"         : df.shape,
            "n_rows"        : df.shape[0],
            "n_cols"        : df.shape[1],
            "memory_mb"     : round(mem_mb, 2),
            "n_missing"     : int(df.isnull().sum().sum()),
            "n_duplicates"  : int(df.duplicated().sum()),
            "dtypes"        : df.dtypes.value_counts().to_dict(),
        }
        for k, v in info.items():
            logger.info(f"  {k:<20}: {v}")
        return info