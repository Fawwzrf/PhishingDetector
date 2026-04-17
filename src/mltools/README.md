# 🤖 mltools — Production ML Pipeline

> **Reusable, production-grade ML library** untuk data tabular — mencakup seluruh alur dari raw data hingga model deployed.  
> Dibangun dengan prinsip **no data leakage**, **modular**, **sklearn-compatible**, dan **reproducible**.

---

## 📋 Daftar Isi

1. [Gambaran Umum](#-gambaran-umum)
2. [Struktur Direktori](#-struktur-direktori)
3. [Instalasi & Setup](#-instalasi--setup)
4. [Konfigurasi](#-konfigurasi)
5. [Quick Start — Full Pipeline](#-quick-start--full-pipeline-1-menit)
6. [Pipeline Lengkap — Step by Step](#-pipeline-lengkap--step-by-step)
   - [FASE 0 — Setup & Persiapan](#fase-0--setup--persiapan)
   - [FASE 1 — Data Loading & Inspeksi](#fase-1--data-loading--inspeksi)
   - [FASE 2 — EDA](#fase-2--exploratory-data-analysis)
   - [FASE 3 — Preprocessing Manual](#fase-3--preprocessing-manual-step-by-step)
   - [FASE 4 — Modeling](#fase-4--modeling)
   - [FASE 5 — Interpretasi & SHAP](#fase-5--interpretasi--shap)
   - [FASE 6 — Save, Registry & Inference](#fase-6--save-registry--inference)
   - [FASE 7 — Serving dengan FastAPI](#fase-7--serving-dengan-fastapi)
7. [Flow Diagram Lengkap](#-flow-diagram-lengkap)
8. [Kasus Penggunaan Khusus](#-kasus-penggunaan-khusus)
9. [Kompetisi ML — Tips Expert](#-kompetisi-ml--tips-expert)
10. [Anti-Pattern yang Wajib Dihindari](#-anti-pattern-yang-wajib-dihindari)
11. [Checklist Production-Ready](#-checklist-production-ready)
12. [Dependensi](#-dependensi)

---

## 🏗️ Gambaran Umum

`mltools` adalah library ML end-to-end yang bisa dipakai ulang di berbagai project. Satu konfigurasi YAML mengontrol seluruh pipeline dari raw data sampai model tersimpan.

```
Raw Data  →  Inspect  →  EDA  →  Preprocess  →  Model  →  Interpret  →  Deploy
```

**Dua cara pakai:**

```python
# ── Cara 1: Full otomatis (1 baris) ──────────────────────────
from mltools import FullMLPipeline, MLConfig
result = FullMLPipeline(MLConfig.from_yaml("configs/ml_config.yaml")).run(df)

# ── Cara 2: Manual per komponen (kontrol penuh) ───────────────
from mltools.preprocessing import PreprocessingPipeline
from mltools.modeling      import ModelingPipeline
split  = PreprocessingPipeline(config).run(df)
result = ModelingPipeline(config).run(split)
```

---

## 📂 Struktur Direktori

```
ml_pipeline/
│
├── configs/
│   └── ml_config.yaml              # Satu config untuk seluruh pipeline
│
├── data/
│   ├── raw/                        # ⛔ JANGAN pernah modifikasi
│   ├── interim/                    # Data hasil cleaning sementara
│   ├── processed/                  # Data final siap modeling (Parquet)
│   └── external/                   # Data dari sumber eksternal
│
├── models/                         # Model tersimpan + metadata JSON
│   ├── registry.json               # Daftar semua versi model
│   └── feature_names.json          # Fitur yang dipakai saat training
│
├── notebooks/
│   ├── 01_eda.ipynb                # Inspeksi & EDA (Fase 1-2)
│   ├── 02_preprocessing.ipynb      # Preprocessing manual (Fase 3)
│   ├── 03_modeling.ipynb           # Training & evaluasi (Fase 4)
│   ├── 04_interpretation.ipynb     # SHAP & insight (Fase 5)
│   └── 05_inference.ipynb          # Prediksi data baru (Fase 6)
│
├── reports/                        # Plot, HTML report, SHAP visualisasi
│
├── logs/                           # Log file per experiment
│
├── src/
│   └── mltools/
│       ├── __init__.py             # Public API
│       ├── pipeline.py             # FullMLPipeline
│       │
│       ├── shared/
│       │   ├── config.py           # MLConfig dari YAML
│       │   ├── exceptions.py       # Custom exception hierarchy
│       │   ├── schemas.py          # DataSplit, TrainingResult
│       │   └── logging.py          # setup_logging()
│       │
│       ├── preprocessing/
│       │   ├── pipeline.py         # PreprocessingPipeline → DataSplit
│       │   ├── inspector.py        # DataInspector
│       │   ├── missing_handler.py  # ExpertMissingHandler
│       │   ├── outlier_handler.py  # ExpertOutlierHandler
│       │   ├── encoder.py          # ExpertCategoricalEncoder
│       │   ├── scaler.py           # ExpertScalerTransformer
│       │   ├── engineer.py         # ExpertFeatureEngineer
│       │   ├── selector.py         # ExpertFeatureSelector
│       │   ├── splitter.py         # ExpertDataSplitter
│       │   └── imbalanced_handler.py
│       │
│       ├── modeling/
│       │   ├── pipeline.py         # ModelingPipeline(DataSplit) → TrainingResult
│       │   ├── baseline.py         # BaselineModel
│       │   ├── linear_models.py    # ExpertLogisticRegression
│       │   ├── tree_models.py      # ExpertDecisionTree, ExpertRandomForest
│       │   ├── boosting_models.py  # ExpertXGBoost, ExpertLightGBM, ExpertCatBoost
│       │   ├── ensemble.py         # VotingEnsembler, StackingEnsembler
│       │   ├── neural_models.py    # ExpertMLPClassifier
│       │   ├── evaluator.py        # ModelEvaluator
│       │   ├── cross_validator.py  # CrossValidator
│       │   └── tuner.py            # OptunaTuner
│       │
│       ├── interpretation/
│       │   └── shap_analysis.py    # SHAPAnalyzer
│       │
│       └── registry/
│           └── model_registry.py   # ModelRegistry
│
├── tests/
│   ├── conftest.py
│   ├── test_shared.py
│   ├── test_preprocessing.py
│   ├── test_modeling.py
│   └── test_pipeline.py
│
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## ⚙️ Instalasi & Setup

```bash
# 1. Clone repositori
git clone <repo-url>
cd ml_pipeline

# 2. Buat virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# 3. Install library sebagai package (editable mode)
pip install -e .

# 4. Atau install dari requirements.txt saja
pip install -r requirements.txt

# 5. Buat direktori yang dibutuhkan
mkdir -p reports models logs data/raw data/processed data/interim

# 6. Setup pre-commit hooks (opsional, untuk code quality)
pre-commit install

# 7. Verifikasi instalasi
python -c "import mltools; print(mltools.__version__)"
# Output: 1.0.0
```

### Setup MLflow (Experiment Tracking)

```bash
# Terminal 1: Jalankan MLflow server
mlflow ui --host 0.0.0.0 --port 5000

# Buka browser: http://localhost:5000
# Semua experiment akan ter-log otomatis saat pipeline dijalankan
```

---

## 🔧 Konfigurasi

Satu file YAML mengontrol seluruh pipeline. Edit sesuai dataset kamu:

```yaml
# configs/ml_config.yaml

project:
  name        : "phishing_detection"   # Nama experiment di MLflow
  version     : "1.0.0"
  random_state: 42
  log_level   : "INFO"                 # DEBUG / INFO / WARNING

data:
  target_column: "phishing"            # Kolom target
  id_columns   : ["id"]               # Kolom ID — tidak dipakai sebagai fitur
  date_columns : ["created_at"]       # Kolom datetime — diekstrak otomatis

preprocessing:
  missing_values:
    threshold_drop_column  : 0.6      # Drop kolom jika missing > 60%
    threshold_drop_row     : 0.5      # Drop baris jika missing > 50%
    strategy_numerical     : "median" # mean / median / knn / iterative
    strategy_categorical   : "most_frequent"
  outliers:
    method   : "iqr"                  # iqr / zscore / isolation_forest / lof
    threshold: 1.5
    treatment: "cap"                  # cap / remove / flag / cap_and_flag
  encoding:
    high_cardinality_threshold: 20
    default_strategy          : "target" # onehot / ordinal / target / woe
  scaling:
    strategy: "robust"               # standard / minmax / robust / power
  feature_selection:
    variance_threshold   : 0.01
    correlation_threshold: 0.95
    n_features_to_select : "auto"

modeling:
  task        : "classification"      # classification / regression
  metric      : "roc_auc"
  n_cv_folds  : 5
  random_state: 42
  baseline:
    strategy: "most_frequent"
  models_to_try:
    - logistic_regression             # Linear benchmark
    - random_forest
    - xgboost
    - lightgbm                        # Biasanya champion
  tuning:
    n_trials: 100                     # 0 = skip tuning
    timeout : 3600                    # Detik, null = tanpa batas
    sampler : "tpe"

mlflow:
  experiment_name: "phishing_detection"
  tracking_uri   : "http://localhost:5000"
```

---

## ⚡ Quick Start — Full Pipeline (1 Menit)

```python
import pandas as pd
from mltools import FullMLPipeline, MLConfig

# 1. Load config
config = MLConfig.from_yaml("configs/ml_config.yaml")

# 2. Load data mentah
df = pd.read_csv("data/raw/dataset.csv")

# 3. Jalankan full pipeline — dari raw data sampai model tersimpan
pipeline = FullMLPipeline(config)
result   = pipeline.run(df)

# 4. Lihat hasil
print(result.summary())
# Output:
#   Champion  : lightgbm
#   roc_auc   : 0.9423
#   f1        : 0.8917
#   Model path: models/lightgbm_champion.joblib

# 5. Prediksi data baru
df_new = pd.read_csv("data/raw/new_data.csv")
predictions = pipeline.predict(df_new)
proba       = pipeline.predict_proba(df_new)

# 6. Simpan pipeline untuk deployment
pipeline.save("models/full_pipeline.joblib")
```

---

## 🔄 Pipeline Lengkap — Step by Step

---

### FASE 0 — Setup & Persiapan

```python
from mltools                import MLConfig
from mltools.shared.logging import setup_logging

# Load config
config = MLConfig.from_yaml("configs/ml_config.yaml")
config.validate()
print(config.summary())

# Setup logging ke file + console
setup_logging(
    log_level  = config.project.log_level,
    log_dir    = "logs",
    experiment = config.project.name,
)

# Shortcut dari config
TARGET    = config.data.target_column    # "phishing"
ID_COLS   = config.data.id_columns      # ["id"]
DATE_COLS = config.data.date_columns    # ["created_at"]
TASK      = config.modeling.task        # "classification"
```

---

### FASE 1 — Data Loading & Inspeksi

#### 1.1 Load Data

```python
import pandas as pd

# ── Opsi 1: Standard load ───────────────────────────────────────────────────
df = pd.read_csv("data/raw/dataset.csv")
# df = pd.read_parquet("data/raw/dataset.parquet")

# ── Opsi 2: Sampling cepat untuk eksplorasi ─────────────────────────────────
df = pd.read_csv("data/raw/dataset.csv").sample(frac=0.2, random_state=42)

# ── Opsi 3: Dataset sangat besar → chunked loading ──────────────────────────
chunks = []
for chunk in pd.read_csv("data/raw/huge_dataset.csv", chunksize=100_000):
    chunks.append(chunk)
df = pd.concat(chunks, ignore_index=True)

# ── Opsi 4: Hemat RAM → Polars ───────────────────────────────────────────────
import polars as pl
df = pl.read_csv("data/raw/dataset.csv").to_pandas()

print(f"Shape  : {df.shape}")
print(f"Memory : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
```

> **⚠️ Aturan Emas:** Data di `data/raw/` **tidak pernah dimodifikasi**. Semua output ke `data/interim/` atau `data/processed/`.

#### 1.2 Data Inspection

Jalankan **sebelum** melakukan perubahan apapun.

```python
from mltools.preprocessing import DataInspector

inspector = DataInspector(df, target=TARGET)
report    = inspector.full_report()

# Atau per bagian:
inspector.basic_info()           # shape, memory, dtype count
inspector.dtype_analysis()       # tipe data + memory per kolom
inspector.missing_analysis()     # % missing + recommendation
inspector.duplicate_analysis()   # baris duplikat
inspector.cardinality_analysis() # unique values, QUASI-ID, CONSTANT
inspector.statistical_summary()  # describe + skewness + kurtosis
inspector.target_analysis()      # distribusi target + imbalance ratio
```

**Keputusan dari hasil inspeksi:**

| Temuan | Tindakan |
|--------|----------|
| Kolom missing > 60% | Drop otomatis di preprocessing |
| Kolom CONSTANT (1 unique) | Drop sebelum modeling |
| Kolom QUASI-ID (unique > 95%) | Exclude dari fitur |
| Target imbalanced > 3x | Aktifkan resampling |
| Skewness abs > 1.0 | Transformasi di scaler |
| Ada kolom datetime | Aktifkan DatetimeFeatureExtractor |
| `memory_usage` besar pada object | Convert ke `category` dtype |

---

### FASE 2 — Exploratory Data Analysis

```python
# ── Auto EDA — laporan HTML komprehensif ─────────────────────────────────────
from ydata_profiling import ProfileReport

profile = ProfileReport(
    df,
    title   = f"{config.project.name} — EDA Report",
    minimal = False,  # True untuk dataset > 500K baris
)
profile.to_file("reports/eda_report.html")
# Buka reports/eda_report.html di browser

# ── Manual EDA — visualisasi terpisah ────────────────────────────────────────
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# Pola missing values
msno.matrix(df)
plt.savefig("reports/missing_matrix.png", dpi=150, bbox_inches="tight")

# Distribusi target
df[TARGET].value_counts().plot(kind="bar")
plt.title(f"Target Distribution: {TARGET}")
plt.savefig("reports/target_dist.png", dpi=150, bbox_inches="tight")

# Korelasi fitur numerik
num_cols = df.select_dtypes(include="number").columns
corr     = df[num_cols].corr()
sns.heatmap(corr, annot=False, cmap="RdBu_r", center=0)
plt.savefig("reports/correlation_heatmap.png", dpi=150, bbox_inches="tight")
```

> **💡 Tips:** Perhatikan korelasi > 0.85 — ini kandidat drop di feature selection Layer 2.

---

### FASE 3 — Preprocessing Manual (Step by Step)

> Gunakan ini untuk kontrol penuh dan pemahaman mendalam.  
> Untuk otomatis, gunakan `PreprocessingPipeline` (lihat di bawah).

#### Step 3.1 — Split Data (LAKUKAN PERTAMA!)

> **🚨 ATURAN PALING KRITIS:** Split sebelum fitting komponen apapun.

```python
from mltools.preprocessing.splitter import ExpertDataSplitter, check_data_leakage

X = df.drop(columns=[TARGET] + ID_COLS, errors="ignore")
y = df[TARGET]

# ── Holdout split (default) ──────────────────────────────────────────────────
splitter = ExpertDataSplitter(
    task        = TASK,
    test_size   = 0.15,
    val_size    = 0.15,
    random_state= config.project.random_state,
)
X_train, X_val, X_test, y_train, y_val, y_test = splitter.split_holdout(X, y)

# ── Time series split ────────────────────────────────────────────────────────
# splitter_ts = ExpertDataSplitter(task="regression", n_splits=5, time_col="date")
# for fold, X_tr, X_va, y_tr, y_va in splitter_ts.split_timeseries(X, y):
#     ...

# ── Group split (satu entitas hanya di satu fold) ───────────────────────────
# splitter_grp = ExpertDataSplitter(task=TASK, n_splits=5, group_col="customer_id")
# for fold, X_tr, X_va, y_tr, y_va in splitter_grp.split_group(X, y):
#     ...

# Validasi no leakage
assert not check_data_leakage(X_train, X_test), "Leakage terdeteksi!"
print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
```

#### Step 3.2 — Handle Missing Values

```python
from mltools.preprocessing.missing_handler import (
    ExpertMissingHandler, analyze_missing_pattern, validate_no_missing
)

# Analisis pola dulu
pattern = analyze_missing_pattern(X_train)
print(pattern)

# Buat handler
handler = ExpertMissingHandler(
    drop_col_threshold    = 0.6,
    num_strategy          = "median",  # mean / median / knn / iterative
    cat_strategy          = "most_frequent",
    add_missing_indicator = True,      # Tambah _was_missing flag
)

# Fit HANYA pada train
handler.fit(X_train)

# Transform semua split
X_train = handler.transform(X_train)
X_val   = handler.transform(X_val)
X_test  = handler.transform(X_test)

# Verifikasi
validate_no_missing(X_train)
```

| % Missing | Strategi Numerik |
|-----------|-----------------|
| < 5% | `median` |
| 5–30%, skewed | `median` |
| 5–30%, normal | `mean` |
| > 30% | `knn` atau `iterative` |
| > 60% | Drop otomatis |

#### Step 3.3 — Handle Outliers

```python
from mltools.preprocessing.outlier_handler import (
    ExpertOutlierHandler, plot_outlier_before_after
)

handler_out = ExpertOutlierHandler(
    method    = "iqr",   # iqr / zscore / modified_zscore / isolation_forest / lof
    treatment = "cap",   # cap / remove / flag / cap_and_flag
    threshold = 1.5,
)

handler_out.fit(X_train)
report_out = handler_out.get_outlier_report(X_train)
print(report_out)

X_train_before = X_train.copy()
X_train = handler_out.transform(X_train)
X_val   = handler_out.transform(X_val)
X_test  = handler_out.transform(X_test)

# Visualisasi
plot_outlier_before_after(
    X_train_before, X_train,
    columns=report_out["column"].head(6).tolist()
)
```

#### Step 3.4 — Handle Imbalanced Data

> **⚠️ HANYA untuk training set!**

```python
from mltools.preprocessing.imbalanced_handler import ExpertImbalancedHandler

vc = y_train.value_counts()
print(f"Imbalance ratio: {vc.max() / vc.min():.1f}x")

# Resampling
handler_imb = ExpertImbalancedHandler(strategy="smotetomek")
X_train_res, y_train_res = handler_imb.fit_resample(X_train, y_train)
handler_imb.plot_distribution(y_train, y_train_res)

# Class weights (alternatif, tidak ubah data)
# handler_cw = ExpertImbalancedHandler(strategy="class_weight")
# _, _ = handler_cw.fit_resample(X_train, y_train)
# weights = handler_cw.class_weights_
```

| Situasi | Strategi |
|---------|----------|
| Imbalance 2–5x | `smotetomek` |
| Imbalance > 10x | `smoteenn` |
| Data kecil | `class_weight` |
| Ada kolom kategorikal | `smotenc` |
| Fraud < 1% | `class_weight` + `isolation_forest` |

#### Step 3.5 — Feature Engineering

```python
from mltools.preprocessing.engineer import (
    ExpertFeatureEngineer, DatetimeFeatureExtractor, create_rfm_features
)

# Fitur umum
fe = ExpertFeatureEngineer(
    add_polynomial  = True,
    poly_cols       = ["feat_a", "feat_b"],
    add_ratios      = True,
    ratio_pairs     = [("income", "debt"), ("amount", "limit")],
    add_bins        = True,
    n_bins          = 5,
    add_interaction = True,
    interaction_pairs = [("age", "income")],
    add_group_agg   = True,
    group_agg_config= [
        {"group_col": "category", "agg_col": "amount", "funcs": ["mean", "std"]}
    ],
)

fe.fit(X_train)
X_train = fe.transform(X_train)
X_val   = fe.transform(X_val)
X_test  = fe.transform(X_test)

# Fitur datetime (jika ada)
if DATE_COLS:
    dte = DatetimeFeatureExtractor(
        date_cols    = DATE_COLS,
        add_cyclical = True,       # sin/cos untuk jam, bulan, hari
    )
    dte.fit(X_train)
    X_train = dte.transform(X_train)
    X_val   = dte.transform(X_val)
    X_test  = dte.transform(X_test)
```

#### Step 3.6 — Encoding Kategorikal

```python
from mltools.preprocessing.encoder import ExpertCategoricalEncoder

encoder = ExpertCategoricalEncoder(
    model_type            = "tree",    # "tree" atau "linear"
    cardinality_threshold = 20,
    high_card_method      = "target",  # target / woe / count / hash
    low_card_method       = "onehot",
    target_smoothing      = 10.0,
    handle_rare           = True,
    rare_tol              = 0.03,
)

encoder.fit(X_train, y_train)   # y diperlukan untuk target encoding
X_train = encoder.transform(X_train)
X_val   = encoder.transform(X_val)
X_test  = encoder.transform(X_test)
```

| Model | Strategi |
|-------|---------|
| Tree-based | `target` (high card), `onehot` (low card) |
| Linear | `ordinal` untuk semua |
| Finance/kredit | `woe` |
| High card > 1000 | `hash` |

#### Step 3.7 — Scaling

```python
from mltools.preprocessing.scaler import ExpertScalerTransformer

# Cek skewness dulu
scaler = ExpertScalerTransformer(
    scaler           = "robust",
    auto_transform   = True,
    transform_method = "yeojohnson",
    skew_threshold   = 1.0,
    exclude_cols     = ["is_weekend", "is_fraud"],  # jangan scale flag binary
)

print(scaler.get_skewness_report(X_train))

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)
```

> **Catatan:** Untuk tree-based (XGBoost, LightGBM, CatBoost), scaling **tidak diperlukan** secara teknis. Tetapi Yeo-Johnson transform untuk fitur sangat skewed tetap berguna.

#### Step 3.8 — Feature Selection

```python
from mltools.preprocessing.selector import (
    ExpertFeatureSelector, compute_permutation_importance
)

selector = ExpertFeatureSelector(
    task              = TASK,
    variance_thr      = 0.01,   # Layer 1: hapus quasi-constant
    corr_threshold    = 0.95,   # Layer 2: hapus highly correlated
    importance_method = "shap", # Layer 3: shap / tree / lasso / mutual_info
    top_n_pct         = 0.6,    # Ambil top 60% fitur terpenting
    use_rfecv         = False,  # Layer 4: aktifkan untuk hasil optimal
)

selector.fit(X_train, y_train)
selector.get_selection_report()
selector.plot_importance(top_n=30)

X_train = selector.transform(X_train)
X_val   = selector.transform(X_val)
X_test  = selector.transform(X_test)

print(f"Fitur setelah seleksi: {len(selector.selected_features_)}")
```

#### Step 3.9 — Gunakan PreprocessingPipeline (Alternatif Otomatis)

```python
from mltools.preprocessing import PreprocessingPipeline
from mltools              import MLConfig

config = MLConfig.from_yaml("configs/ml_config.yaml")
split  = PreprocessingPipeline(config).run(df)

# split adalah DataSplit dengan atribut:
print(split.summary())
# X_train, X_val, X_test, y_train, y_val, y_test
# feature_names, task, n_classes, is_binary, class_balance
```

---

### FASE 4 — Modeling

#### Step 4.1 — Baseline (WAJIB)

```python
from mltools.modeling.baseline import BaselineModel

baseline = BaselineModel(task=TASK)
baseline.evaluate_from_split(split)
# Output:
#   Strategy : most_frequent
#   accuracy : 0.3733
#   roc_auc  : 0.5000   ← ini floor minimum
```

#### Step 4.2 — Train Model Manual

```python
from mltools.modeling.boosting_models import ExpertLightGBM, ExpertXGBoost

# LightGBM — champion untuk sebagian besar dataset tabular
lgbm = ExpertLightGBM(
    n_estimators          = 2000,
    learning_rate         = 0.05,
    num_leaves            = 63,
    early_stopping_rounds = 50,
    is_unbalance          = True,
)
lgbm.fit(split.X_train, split.y_train, split.X_val, split.y_val)

# XGBoost
xgb = ExpertXGBoost(
    n_estimators          = 2000,
    learning_rate         = 0.05,
    early_stopping_rounds = 50,
)
xgb.fit(split.X_train, split.y_train, split.X_val, split.y_val,
        auto_imbalance=True)
```

#### Step 4.3 — Cross-Validation

```python
from mltools.modeling.cross_validator import CrossValidator

cv = CrossValidator.from_config(config)
scores = cv.score(lgbm.model, split.X_train, split.y_train,
                  scoring="roc_auc")
print(f"CV ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")
```

#### Step 4.4 — Evaluasi

```python
from mltools.modeling.evaluator import ModelEvaluator

evaluator = ModelEvaluator(task=TASK)

# Evaluasi di val set
val_metrics = evaluator.evaluate_from_split(lgbm, split, split_name="val")
print(val_metrics)

# Cek vs baseline
baseline.is_better(val_metrics["roc_auc"], metric="roc_auc")

# Confusion matrix
evaluator.plot_confusion(split.y_val, lgbm.predict(split.X_val))
evaluator.print_report(split.y_val, lgbm.predict(split.X_val))
```

#### Step 4.5 — Hyperparameter Tuning dengan Optuna

```python
from mltools.modeling.tuner import OptunaTuner
import lightgbm as lgb

# Definisi search space
param_space = {
    "n_estimators"    : lambda t: t.suggest_int("n_estimators", 200, 2000, step=100),
    "learning_rate"   : lambda t: t.suggest_float("learning_rate", 1e-4, 0.3, log=True),
    "num_leaves"      : lambda t: t.suggest_int("num_leaves", 20, 255),
    "min_data_in_leaf": lambda t: t.suggest_int("min_data_in_leaf", 5, 100),
    "lambda_l1"       : lambda t: t.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
    "lambda_l2"       : lambda t: t.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    "feature_fraction": lambda t: t.suggest_float("feature_fraction", 0.4, 1.0),
    "verbose"         : lambda t: -1,
}

tuner = OptunaTuner.from_config(
    model_class = lgb.LGBMClassifier,
    param_space = param_space,
    config      = config,   # Ambil n_trials, timeout, metric dari config
)

best_params, best_score = tuner.tune(split.X_train, split.y_train)
print(f"Best ROC-AUC: {best_score:.4f}")
print(f"Best params : {best_params}")

# Refit dengan best params
final_model = lgb.LGBMClassifier(**best_params, verbose=-1)
final_model.fit(
    split.X_train, split.y_train,
    eval_set  = [(split.X_val, split.y_val)],
    callbacks = [lgb.early_stopping(50, verbose=False)],
)
```

#### Step 4.6 — Evaluasi Final di Test Set

> **🚨 Test set hanya boleh dievaluasi SEKALI — di paling akhir!**

```python
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, f1_score
)

y_pred_test = final_model.predict(split.X_test)
y_prob_test = final_model.predict_proba(split.X_test)[:, 1]

print("=" * 50)
print("FINAL TEST SET RESULTS")
print("=" * 50)
print(f"ROC-AUC  : {roc_auc_score(split.y_test, y_prob_test):.4f}")
print(f"F1-Score : {f1_score(split.y_test, y_pred_test):.4f}")
print()
print(classification_report(split.y_test, y_pred_test))

# Threshold optimization (untuk imbalanced)
from mltools.preprocessing.imbalanced_handler import ExpertImbalancedHandler
optimal_thr, best_f1 = ExpertImbalancedHandler.find_optimal_threshold(
    split.y_val, final_model.predict_proba(split.X_val)[:, 1], metric="f1"
)
print(f"Optimal threshold: {optimal_thr:.4f} (F1={best_f1:.4f})")

# Apply optimal threshold ke test
y_pred_opt = (y_prob_test >= optimal_thr).astype(int)
print(f"F1 dengan threshold {optimal_thr:.2f}: {f1_score(split.y_test, y_pred_opt):.4f}")
```

#### Step 4.7 — Gunakan ModelingPipeline (Alternatif Otomatis)

```python
from mltools.modeling import ModelingPipeline

pipeline = ModelingPipeline(config)
result   = pipeline.run(split)   # Terima DataSplit, return TrainingResult

print(result.summary())
# Champion  : lightgbm
# roc_auc   : 0.9423
# All scores: lightgbm=0.94, xgboost=0.91, random_forest=0.88
```

---

### FASE 5 — Interpretasi & SHAP

```python
from mltools.interpretation import SHAPAnalyzer

# Inisialisasi langsung dari hasil pipeline
analyzer = SHAPAnalyzer.from_result(result, split)

# Atau manual
# analyzer = SHAPAnalyzer(
#     model        = final_model,
#     X_background = split.X_train,
#     feature_names= split.feature_names,
#     task         = TASK,
# )

# Jalankan full analysis — semua visualisasi disimpan ke reports/
analyzer.full_analysis(split.X_test, top_n=20)

# Atau per visualisasi:
shap_values = analyzer.compute(split.X_test)
analyzer.plot_importance(top_n=20)      # Bar chart global importance
analyzer.plot_beeswarm(top_n=20)        # Distribusi SHAP per fitur
analyzer.plot_waterfall(sample_idx=0)   # Penjelasan lokal satu sampel
analyzer.plot_dependence("url_length")  # Hubungan fitur dengan prediksi

# Export ke DataFrame
importance_df = analyzer.get_importance_df()
importance_df.to_csv("reports/shap_importance.csv", index=False)
print(importance_df.head(10))
```

**Cara baca SHAP:**
- `mean_abs_shap` tinggi → fitur sangat berpengaruh ke prediksi
- SHAP positif → fitur mendorong prediksi ke kelas 1
- SHAP negatif → fitur mendorong prediksi ke kelas 0
- Beeswarm merah → nilai fitur tinggi, biru → nilai fitur rendah

---

### FASE 6 — Save, Registry & Inference

#### 6.1 Simpan ke Model Registry

```python
from mltools.registry import ModelRegistry
from mltools.shared.schemas import TrainingResult

registry = ModelRegistry()

# Save dari TrainingResult (otomatis dapat versi dan metadata)
model_path = registry.save(result, is_champion=True)

# Atau save manual
import joblib
joblib.dump(final_model, "models/lgbm_v1.joblib", compress=3)

# Simpan fitur yang dipakai (wajib untuk inference!)
import json
with open("models/feature_names.json", "w") as f:
    json.dump({"feature_names": split.feature_names}, f)
```

#### 6.2 List Model di Registry

```python
registry.list_models()
# name       version              model_type  metrics           is_champion
# lightgbm   v_20240315_143022    LGBMClass   roc_auc=0.9423    True
# xgboost    v_20240315_141503    XGBClass    roc_auc=0.9105    False
```

#### 6.3 Load Model

```python
# Load champion
champion_model = registry.load("lightgbm", version="champion")

# Load versi spesifik
old_model = registry.load("lightgbm", version="v_20240315_143022")

# Load preprocessing pipeline
pipeline = FullMLPipeline.load("models/full_pipeline.joblib")
```

#### 6.4 Inference Data Baru

```python
import pandas as pd

df_new = pd.read_csv("data/raw/new_data.csv")

# ── Cara 1: Pakai FullMLPipeline (paling mudah) ──────────────────────────────
pipeline    = FullMLPipeline.load("models/full_pipeline.joblib")
predictions = pipeline.predict(df_new)
proba       = pipeline.predict_proba(df_new)

# ── Cara 2: Manual dengan preprocessing components ───────────────────────────
X_new = df_new.drop(columns=ID_COLS, errors="ignore")

# Apply semua transformer dari training (FIT sudah tersimpan)
X_new = handler.transform(X_new)
X_new = handler_out.transform(X_new)
X_new = encoder.transform(X_new)
X_new = scaler.transform(X_new)
X_new = selector.transform(X_new)

y_pred = champion_model.predict(X_new)
y_prob = champion_model.predict_proba(X_new)[:, 1]

# Simpan hasil
df_new["predicted_label"] = y_pred
df_new["predicted_prob"]  = y_prob.round(4)
df_new.to_csv("data/processed/predictions.csv", index=False)
print(f"Prediksi selesai: {len(df_new):,} baris")
```

---

### FASE 7 — Serving dengan FastAPI

```bash
# 1. Pastikan model sudah tersimpan
ls models/
# full_pipeline.joblib
# feature_names.json

# 2. Jalankan API server
uvicorn mltools.serve.app:app --host 0.0.0.0 --port 8000 --reload

# 3. Buka dokumentasi interaktif
# http://localhost:8000/docs
```

```python
# Test endpoint dari Python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())
# {"status": "healthy", "model_loaded": true, "model_type": "LGBMClassifier"}

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "features": {
            "url_length" : 120,
            "n_dots"     : 4,
            "has_https"  : 0,
            "domain_age" : 15,
        },
        "return_proba": True,
    }
)
print(response.json())
# {"prediction": 1, "probability": 0.9234, "inference_time_ms": 3.2}

# Batch prediction
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={
        "records": [
            {"url_length": 120, "n_dots": 4, "has_https": 0},
            {"url_length": 45,  "n_dots": 1, "has_https": 1},
        ]
    }
)
print(response.json())
# {"predictions": [1, 0], "probabilities": [0.92, 0.08], "n_records": 2}
```

---

## 📊 Flow Diagram Lengkap

```
┌──────────────────────────────────────────────────────────────┐
│                      RAW DATA (CSV/Parquet)                   │
└────────────────────────────┬─────────────────────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │   FASE 1: Inspection         │  DataInspector.full_report()
              │   Shape, dtype, missing,     │  memory_usage(deep=True)
              │   cardinality, target dist   │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │   FASE 2: EDA                │  ydata_profiling / missingno
              │   Distribusi, korelasi,      │  seaborn heatmap
              │   pola missing               │
              └──────────────┬──────────────┘
                             │
┌────────────────────────────▼──────────────────────────────────┐
│  FASE 3: PREPROCESSING — FIT HANYA PADA TRAIN!                │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│  ① ExpertDataSplitter      → Train / Val / Test               │
│       ↓ (fit zone dimulai)                                     │
│  ② ExpertMissingHandler    → Impute missing                   │
│  ③ ExpertOutlierHandler    → Cap / remove outlier             │
│  ④ ExpertImbalancedHandler → SMOTE (train only!)              │
│  ⑤ ExpertFeatureEngineer   → Polynomial, ratio, group agg    │
│  ⑥ DatetimeFeatureExtractor→ sin/cos datetime features        │
│  ⑦ ExpertCategoricalEncoder→ Target / OHE / WoE encoding      │
│  ⑧ ExpertScalerTransformer → RobustScaler + Yeo-Johnson       │
│  ⑨ ExpertFeatureSelector   → 4-layer filter                   │
│                                                                │
│       OUTPUT: DataSplit(X_train, X_val, X_test, ...)          │
└────────────────────────────┬──────────────────────────────────┘
                             │
┌────────────────────────────▼──────────────────────────────────┐
│  FASE 4: MODELING                                              │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│  ① BaselineModel           → Floor minimum (roc_auc ~0.5)    │
│  ② ExpertLogisticRegression→ Linear benchmark                 │
│  ③ ExpertRandomForest      → Ensemble baseline               │
│  ④ ExpertXGBoost           → Gradient boosting               │
│  ⑤ ExpertLightGBM          → Champion untuk tabular          │
│  ⑥ CrossValidator          → 5-fold stratified CV            │
│  ⑦ OptunaTuner             → 100 trials Bayesian HPO         │
│  ⑧ ModelEvaluator          → Test set final evaluation       │
│                                                                │
│       OUTPUT: TrainingResult(champion_model, metrics, ...)    │
└────────────────────────────┬──────────────────────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │   FASE 5: Interpretasi       │  SHAPAnalyzer.full_analysis()
              │   Global + local SHAP,       │  → reports/shap_*.png
              │   dependence plots           │  → reports/shap_importance.csv
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │   FASE 6: Registry & Save    │  ModelRegistry.save(result)
              │   Versioning, metadata,      │  FullMLPipeline.save()
              │   feature names              │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │   FASE 7: Serving            │  FastAPI /predict
              │   REST API inference,        │  /predict/batch
              │   health check               │  /health
              └─────────────────────────────┘
```

---

## 🔧 Kasus Penggunaan Khusus

### Phishing Detection (Dataset Project Ini)

```python
config = MLConfig.from_yaml("configs/ml_config.yaml")
# target: "phishing", task: "classification"

# Pipeline otomatis
pipeline = FullMLPipeline(config)
result   = pipeline.run(df)

# Atau manual dengan isolation_forest untuk anomaly features
handler_out = ExpertOutlierHandler(
    method="isolation_forest", treatment="cap_and_flag", contamination=0.05
)
```

### Data Time Series

```python
splitter = ExpertDataSplitter(strategy="timeseries", time_col="date", n_splits=5)

# WAJIB ffill untuk missing time series
from mltools.preprocessing.missing_handler import handle_timeseries_missing
df = handle_timeseries_missing(df, date_col="date", method="ffill")

dte = DatetimeFeatureExtractor(date_cols=["date"], add_cyclical=True)
```

### Credit Scoring / Finance

```python
encoder = ExpertCategoricalEncoder(
    high_card_method = "woe",   # Weight of Evidence — interpretable
    low_card_method  = "ordinal",
)
handler_out = ExpertOutlierHandler(method="modified_zscore", treatment="cap")
```

### Fraud Detection

```python
handler_out = ExpertOutlierHandler(
    method="isolation_forest", treatment="cap_and_flag", contamination=0.02
)
handler_imb = ExpertImbalancedHandler(strategy="smoteenn", sampling_strategy=0.3)
```

### E-Commerce / CLV

```python
rfm = create_rfm_features(df, customer_col="user_id",
                          date_col="order_date", amount_col="order_value")
fe = ExpertFeatureEngineer(
    add_group_agg=True,
    group_agg_config=[
        {"group_col": "product_category", "agg_col": "price", "funcs": ["mean","std"]},
    ],
)
```

---

## 🏆 Kompetisi ML — Tips Expert

### Adversarial Validation

```python
# Deteksi distribusi shift antara train dan test SEBELUM modeling
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import cross_val_score

X_adv = pd.concat([
    X_train.assign(is_test=0),
    X_test.assign(is_test=1)
], ignore_index=True)
y_adv = X_adv.pop("is_test")

adv_model = lgb.LGBMClassifier(verbose=-1)
auc = cross_val_score(adv_model, X_adv, y_adv, scoring="roc_auc", cv=5).mean()

print(f"Adversarial AUC: {auc:.4f}")
# < 0.55 → Distribusi aman, CV standard
# 0.55–0.70 → Drop fitur bermasalah
# > 0.70 → Distribusi shift serius, buat adversarial val set
```

### Pseudo-Labeling

```python
# Train model → prediksi test → ambil confident → tambah ke train → retrain
model.fit(X_train, y_train)
test_proba = model.predict_proba(X_test)

# Ambil hanya prediksi sangat confident (≥95%)
confident_mask = (test_proba.max(axis=1) >= 0.95)
X_pseudo = X_test[confident_mask]
y_pseudo  = (test_proba[confident_mask].argmax(axis=1))

# Gabungkan dan retrain
X_new = pd.concat([X_train, X_pseudo])
y_new = pd.concat([y_train, pd.Series(y_pseudo)])
model.fit(X_new, y_new)
```

### Threshold Optimization

```python
# Default 0.5 hampir tidak pernah optimal untuk imbalanced data
from mltools.preprocessing.imbalanced_handler import ExpertImbalancedHandler

y_prob = model.predict_proba(X_val)[:, 1]
optimal_thr, best_f1 = ExpertImbalancedHandler.find_optimal_threshold(
    y_val, y_prob, metric="f1"
)
y_pred_final = (y_prob >= optimal_thr).astype(int)
```

---

## ⛔ Anti-Pattern yang Wajib Dihindari

```python
# ❌ SALAH: Fit pada seluruh data sebelum split
scaler.fit(df_full)           # DATA LEAKAGE!

# ✅ BENAR: Fit hanya pada train setelah split
X_train, X_val, X_test, ... = splitter.split_holdout(X, y)
scaler.fit(X_train)

# ──────────────────────────────────────────────────────────────

# ❌ SALAH: Resampling pada val atau test
X_val_res, _ = handler_imb.fit_resample(X_val, y_val)   # SALAH!

# ✅ BENAR: Resampling HANYA pada train
X_train_res, y_train_res = handler_imb.fit_resample(X_train, y_train)

# ──────────────────────────────────────────────────────────────

# ❌ SALAH: Evaluasi test set berkali-kali
auc_v1 = roc_auc_score(y_test, model_v1.predict(X_test))  # Round 1
auc_v2 = roc_auc_score(y_test, model_v2.predict(X_test))  # Round 2 — OVERFITTING!

# ✅ BENAR: Test set HANYA sekali di akhir
# Semua keputusan berdasarkan val set, test set hanya untuk laporan final

# ──────────────────────────────────────────────────────────────

# ❌ SALAH: Feature selection berdasarkan seluruh data
corr = df.corr()    # Include info dari test set!

# ✅ BENAR: Semua keputusan berdasarkan train set
selector.fit(X_train, y_train)

# ──────────────────────────────────────────────────────────────

# ❌ SALAH: KFold biasa untuk klasifikasi imbalanced
from sklearn.model_selection import KFold          # Bisa unequal class dist!

# ✅ BENAR: StratifiedKFold selalu untuk klasifikasi
from sklearn.model_selection import StratifiedKFold
```

---

## ✅ Checklist Production-Ready

```
PREPROCESSING:
□ Split dilakukan SEBELUM fitting komponen apapun
□ validate_no_missing() mengembalikan True
□ check_data_leakage() mengembalikan False
□ Resampling HANYA pada training set
□ Semua komponen di-fit pada X_train, bukan X_full

MODELING:
□ Baseline dijalankan dan didokumentasikan sebagai floor
□ Semua model menggunakan StratifiedKFold (bukan KFold)
□ Val score dan test score tidak terlalu berbeda (< 5% gap)
□ Test set hanya dievaluasi SEKALI di akhir
□ Semua runs ter-log di MLflow dengan params + metrics

INTERPRETASI:
□ SHAP analysis dijalankan → laporan ada di reports/
□ Fitur paling penting masuk akal secara bisnis
□ Tidak ada fitur yang "terlalu bagus" (suspect leakage)

DEPLOYMENT:
□ FullMLPipeline disimpan ke models/full_pipeline.joblib
□ feature_names.json tersimpan untuk validasi input
□ ModelRegistry mencatat semua versi model
□ FastAPI /health endpoint berjalan
□ /predict endpoint ditest dengan data nyata
□ Versi library dicatat di requirements.txt

DOKUMENTASI:
□ Setiap keputusan preprocessing didokumentasikan
□ Config YAML sudah di-commit ke repo
□ README up to date
```

---

## 📦 Dependensi

| Library | Versi | Fungsi |
|---------|-------|--------|
| `pandas` | ≥ 2.1.0 | Manipulasi data |
| `scikit-learn` | ≥ 1.4.0 | ML base + transformers |
| `lightgbm` | ≥ 4.2.0 | Champion boosting model |
| `xgboost` | ≥ 2.0.0 | Gradient boosting |
| `catboost` | ≥ 1.2.0 | Categorical boosting |
| `category_encoders` | ≥ 2.6.0 | Target/WoE/Hash encoding |
| `feature-engine` | ≥ 1.6.2 | RareLabelEncoder |
| `imbalanced-learn` | ≥ 0.12.0 | SMOTE, SMOTETomek |
| `shap` | ≥ 0.44.0 | Model interpretability |
| `optuna` | ≥ 3.5.0 | Bayesian HPO |
| `mlflow` | ≥ 2.10.0 | Experiment tracking |
| `fastapi` | ≥ 0.110.0 | Model serving API |
| `pydantic` | ≥ 2.0.0 | Data validation |
| `loguru` | ≥ 0.7.2 | Production logging |
| `ydata-profiling` | ≥ 4.6.0 | Auto EDA HTML report |

```bash
pip install -r requirements.txt
pip install -e .
```

---

<div align="center">

**Dibuat untuk data scientist dan ML engineer yang peduli reproducibility dan production-readiness.**

`mltools v1.0.0` · Python 3.10+ · MIT License

</div>