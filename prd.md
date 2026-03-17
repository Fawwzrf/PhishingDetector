# PRD — Phishing URL Detector Web System
**Product Requirements Document**
Version 1.0 | PhishingDetector Monorepo

---

## 1. Overview

### 1.1 Latar Belakang

Phishing adalah serangan siber di mana penyerang membuat URL yang menyerupai situs legitimate untuk mencuri kredensial. Salah satu teknik yang semakin umum adalah **Punycode/IDN Homograph Attack** — menggunakan karakter Unicode yang identik secara visual dengan karakter ASCII. Contoh: `аpple.com` (huruf `а` Cyrillic) vs `apple.com` (ASCII), keduanya terlihat sama di browser.

Sistem ini membangun website yang memungkinkan pengguna mendeteksi URL phishing secara real-time dengan pipeline:

```
Input URL (raw) → Decode Punycode → Ekstrak 112 Fitur → Model ML → Hasil Deteksi
```

### 1.2 Tujuan Produk

- Memberikan alat deteksi phishing yang mudah digunakan oleh siapapun
- Mengedukasi pengguna tentang karakteristik URL phishing
- Menyediakan transparansi hasil prediksi (fitur apa yang paling berpengaruh)
- Logging setiap prediksi ke MLflow untuk monitoring model drift

### 1.3 Scope

| In Scope | Out of Scope |
|----------|-------------|
| Deteksi URL berbasis fitur struktural | Scanning konten halaman web |
| Punycode/IDN decoding | Real-time crawling website |
| Penjelasan hasil (SHAP top features) | Browser extension |
| MLflow logging per request | User authentication |
| Riwayat pencarian (localStorage) | Database pengguna |

---

## 2. User & Persona

### Persona 1 — Pengguna Awam
- **Siapa:** Karyawan, mahasiswa, pengguna internet umum
- **Masalah:** Menerima link mencurigakan via WhatsApp/email, tidak tahu cara verifikasi
- **Kebutuhan:** Antarmuka simpel, hasil jelas (aman/berbahaya), tidak perlu teknis
- **Ekspektasi:** Respons < 3 detik, penjelasan sederhana

### Persona 2 — Security Analyst
- **Siapa:** IT staff, researcher keamanan siber
- **Masalah:** Perlu verifikasi batch URL, butuh detail teknis
- **Kebutuhan:** Lihat semua fitur yang diekstrak, skor per fitur, confidence score
- **Ekspektasi:** API endpoint tersedia, detail teknis bisa diakses

---

## 3. User Stories

```
US-01  Sebagai pengguna, saya ingin menempelkan URL dan mendapat
       hasil deteksi dalam < 3 detik.

US-02  Sebagai pengguna, saya ingin tahu MENGAPA URL dianggap phishing
       (fitur apa yang paling berpengaruh).

US-03  Sebagai pengguna, saya ingin melihat URL asli vs hasil
       decode Punycode untuk mendeteksi IDN homograph attack.

US-04  Sebagai pengguna, saya ingin melihat riwayat URL yang
       pernah saya cek (tersimpan di browser).

US-05  Sebagai security analyst, saya ingin mengakses API langsung
       tanpa melalui UI.

US-06  Sebagai admin, saya ingin setiap prediksi ter-log di MLflow
       untuk monitoring performa model.
```

---

## 4. Fitur Sistem

### 4.1 Core Features

#### F-01: URL Input & Validation
- Input field untuk satu URL
- Validasi format URL dasar sebelum dikirim ke backend
- Deteksi dan tampilkan warning jika URL mengandung karakter non-ASCII (Punycode suspect)
- Tombol paste dari clipboard

#### F-02: Punycode/IDN Decoding
- Setiap URL diproses melalui pipeline decoding sebelum analisis
- Tampilkan URL asli vs URL setelah decode
- Highlight perbedaan jika ada (karakter Cyrillic, Greek, dll. yang mirip ASCII)

```
Input   : https://xn--pple-43d.com/login
Decoded : https://аpple.com/login  ← 'а' adalah Cyrillic!
Warning : "URL mengandung karakter Unicode yang mencurigakan"
```

#### F-03: Feature Extraction (112 fitur)
Ekstraksi otomatis dari URL string:

| Kategori | Fitur | Metode Ekstraksi |
|----------|-------|-----------------|
| URL structure | length_url, qty_dot_url, qty_hyphen_url, ... | String parsing |
| Domain | length_hostname, ip_present, https_present | urllib.parse |
| Path | qty_slash_url, qty_equal_url, qty_questionmark_url | String parsing |
| Punycode | is_punycode, n_unicode_chars, homograph_score | idna library |
| Network* | time_response, qty_redirects, ttl_hostname | HTTP request (async) |
| DNS* | qty_ip_resolved, qty_nameservers | DNS lookup (async) |
| WHOIS* | time_domain_activation, time_domain_expiration | python-whois |

> `*` Fitur network/DNS diambil secara async dengan timeout 5 detik.
> Jika gagal/timeout → nilai diset ke median dari training data (dari `feature_names.json`).

#### F-04: Prediksi Model
- Model LightGBM champion dari pipeline training
- Output: label (PHISHING/LEGITIMATE), probabilitas, confidence level
- Threshold optimal dari training (`modeling_meta.json`)

#### F-05: Explainability
- Tampilkan top 5 fitur yang paling berpengaruh untuk prediksi ini
- Nilai SHAP per fitur (positif = mendorong ke phishing, negatif = mendorong ke legitimate)
- Visualisasi bar horizontal sederhana

#### F-06: Riwayat Pencarian
- Simpan 10 pencarian terakhir di localStorage browser
- Tampilkan: URL, hasil, probabilitas, timestamp
- Tombol hapus riwayat

#### F-07: MLflow Logging
Setiap request ke `/predict` dicatat:
- URL yang dicek (di-hash untuk privacy)
- Fitur yang diekstrak
- Hasil prediksi (label, probabilitas)
- Waktu inferensi (ms)
- Versi model yang digunakan

---

## 5. Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────┐
│                    MONOREPO                              │
│                                                          │
│  ┌──────────────┐    HTTP/REST    ┌─────────────────┐   │
│  │   FRONTEND   │ ─────────────► │    BACKEND      │   │
│  │  Next.js 14  │ ◄───────────── │    FastAPI      │   │
│  │  TypeScript  │                │    Python       │   │
│  │  Tailwind CSS│                └────────┬────────┘   │
│  └──────────────┘                         │            │
│                                  ┌────────▼────────┐   │
│                                  │  ML PIPELINE    │   │
│                                  │  - Preprocessors│   │
│                                  │  - LightGBM     │   │
│                                  │  - SHAP         │   │
│                                  └────────┬────────┘   │
│                                           │            │
│                                  ┌────────▼────────┐   │
│                                  │    MLFLOW       │   │
│                                  │  Experiment Log │   │
│                                  └─────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 5.1 Struktur Folder Monorepo

```
PhishingDetector/
│
├── frontend/                        ← Next.js app
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx             ← Halaman utama
│   │   │   ├── layout.tsx
│   │   │   └── globals.css
│   │   ├── components/
│   │   │   ├── URLInput.tsx         ← Input + validasi
│   │   │   ├── ResultCard.tsx       ← Hasil prediksi
│   │   │   ├── FeatureDetails.tsx   ← Breakdown fitur
│   │   │   ├── SHAPChart.tsx        ← Top 5 SHAP features
│   │   │   ├── PunycodeAlert.tsx    ← Warning IDN homograph
│   │   │   └── History.tsx          ← Riwayat pencarian
│   │   ├── lib/
│   │   │   ├── api.ts               ← Fetch ke FastAPI
│   │   │   └── history.ts           ← localStorage utils
│   │   └── types/
│   │       └── index.ts             ← TypeScript interfaces
│   ├── public/
│   ├── package.json
│   ├── tailwind.config.ts
│   └── next.config.ts
│
├── backend/                         ← FastAPI app
│   ├── app.py                       ← Main FastAPI app
│   ├── schemas.py                   ← Pydantic models
│   ├── extractor.py                 ← Feature extraction + Punycode
│   ├── predictor.py                 ← Model inference + SHAP
│   └── mlflow_logger.py             ← MLflow logging
│
├── src/mltools/                     ← Library (tidak berubah)
├── models/                          ← Artifacts training
├── notebooks/                       ← Notebook EDA + training
├── configs/
│   └── ml_config.yaml
├── requirements.txt                 ← Python deps
└── README.md
```

---

## 6. API Contract

### Base URL
```
Development : http://localhost:8000
Production  : https://api.phishingdetector.com (TBD)
```

### Endpoints

---

#### `GET /health`
Cek status server dan model.

**Response:**
```json
{
  "status"       : "healthy",
  "model"        : "lightgbm",
  "model_version": "v_20240315_143022",
  "threshold"    : 0.52
}
```

---

#### `POST /predict`
Prediksi URL.

**Request Body:**
```json
{
  "url": "https://xn--pple-43d.com/login?user=admin"
}
```

**Response 200:**
```json
{
  "result": {
    "label"      : "PHISHING",
    "probability": 0.9234,
    "confidence" : "HIGH",
    "threshold"  : 0.52
  },
  "url_analysis": {
    "url_original" : "https://xn--pple-43d.com/login",
    "url_decoded"  : "https://аpple.com/login",
    "is_punycode"  : true,
    "punycode_warning": "Domain mengandung karakter Cyrillic yang mirip ASCII"
  },
  "top_features": [
    { "feature": "qty_dot_url",     "value": 4.0, "shap": 0.312,  "direction": "phishing" },
    { "feature": "length_url",      "value": 89,  "shap": 0.198,  "direction": "phishing" },
    { "feature": "is_punycode",     "value": 1.0, "shap": 0.445,  "direction": "phishing" },
    { "feature": "https_present",   "value": 1.0, "shap": -0.087, "direction": "legit"    },
    { "feature": "time_response",   "value": 380, "shap": 0.134,  "direction": "phishing" }
  ],
  "all_features": {
    "length_url"     : 89,
    "qty_dot_url"    : 4,
    "is_punycode"    : 1,
    "https_present"  : 1,
    "time_response"  : 380
  },
  "meta": {
    "model_version"   : "v_20240315_143022",
    "inference_time_ms": 45,
    "request_id"      : "req_abc123"
  }
}
```

**Response 422 (URL tidak valid):**
```json
{
  "detail": "URL tidak valid. Pastikan dimulai dengan http:// atau https://"
}
```

**Response 503:**
```json
{
  "detail": "Model belum siap"
}
```

---

#### `GET /features/schema`
Daftar semua 112 fitur beserta deskripsinya.

**Response:**
```json
{
  "total_features": 112,
  "features": [
    {
      "name"       : "length_url",
      "description": "Panjang total karakter URL",
      "category"   : "url_structure",
      "type"       : "numeric"
    }
  ]
}
```

---

## 7. Desain UI/UX

### 7.1 Wireframe Halaman Utama

```
┌─────────────────────────────────────────────────────────┐
│  🎣 PhishingDetector                          [GitHub]  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│         Periksa apakah URL aman sebelum diklik          │
│                                                          │
│  ┌────────────────────────────────────────┐  [Deteksi]  │
│  │ https://contoh.com/login               │             │
│  └────────────────────────────────────────┘             │
│  [📋 Paste dari clipboard]                              │
│                                                          │
├─ Hasil Deteksi ─────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  🚨  PHISHING TERDETEKSI                         │   │
│  │                                                  │   │
│  │  Probabilitas: ████████████░░  92.3%             │   │
│  │  Confidence  : HIGH                              │   │
│  │                                                  │   │
│  │  ⚠️ URL Mencurigakan:                            │   │
│  │  Original : https://xn--pple-43d.com/login      │   │
│  │  Decoded  : https://аpple.com/login              │   │
│  │  ↑ Karakter 'а' adalah Cyrillic, bukan ASCII!    │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌─ Mengapa dianggap phishing? ────────────────────┐    │
│  │  is_punycode      ████████████  +0.445 ↑        │    │
│  │  qty_dot_url      ████████░░░░  +0.312 ↑        │    │
│  │  length_url       ██████░░░░░░  +0.198 ↑        │    │
│  │  time_response    ████░░░░░░░░  +0.134 ↑        │    │
│  │  https_present    ██░░░░░░░░░░  -0.087 ↓ (aman) │    │
│  └──────────────────────────────────────────────────┘    │
│                                                          │
│  [Lihat semua 21 fitur ▼]                               │
│                                                          │
├─ Riwayat Terakhir ──────────────────────────────────────┤
│  🚨 https://xn--pple-43d.com     92.3%   2 menit lalu  │
│  ✅ https://google.com           3.1%    5 menit lalu  │
│  [Hapus Riwayat]                                        │
└─────────────────────────────────────────────────────────┘
```

### 7.2 Color System

| Kondisi | Warna | Hex |
|---------|-------|-----|
| Phishing — HIGH | Merah terang | `#ef4444` |
| Phishing — MEDIUM | Oranye | `#f97316` |
| Legitimate — HIGH | Hijau | `#22c55e` |
| Legitimate — MEDIUM | Hijau muda | `#86efac` |
| Warning Punycode | Kuning | `#eab308` |
| Background | Abu sangat terang | `#f8fafc` |
| Card | Putih | `#ffffff` |

### 7.3 State UI

| State | Tampilan |
|-------|----------|
| Idle | Input kosong, placeholder text |
| Loading | Spinner + teks "Menganalisis URL..." |
| Phishing | Card merah + icon 🚨 + bar merah |
| Legitimate | Card hijau + icon ✅ + bar hijau |
| Punycode warning | Banner kuning di atas result |
| Error (invalid URL) | Inline error di bawah input |
| Error (server) | Toast notification merah |

---

## 8. Pipeline Teknis

### 8.1 Alur Request Lengkap

```
User input URL
      │
      ▼
[Frontend - URLInput.tsx]
  1. Trim whitespace
  2. Validasi format dasar (regex)
  3. POST ke /predict

      │
      ▼
[Backend - app.py]
  4. Validasi Pydantic schema

      │
      ▼
[extractor.py]
  5. Decode Punycode → Unicode
  6. Deteksi IDN homograph attack
  7. Ekstrak fitur URL (sinkron, ~1ms)
  8. Ekstrak fitur network (async, timeout 5s)
     - HTTP HEAD request → time_response, qty_redirects, https_present
     - DNS lookup → ttl_hostname, qty_ip_resolved
     - WHOIS → time_domain_activation, time_domain_expiration
  9. Fallback: fitur yang gagal → median dari training data

      │
      ▼
[predictor.py]
  10. Transform fitur → DataFrame
  11. Apply preprocessing pipeline:
      missing_handler.transform()
      outlier_handler.transform()
      feature_engineer.transform()
      scaler.transform()
      selector.transform()
  12. model.predict_proba()
  13. SHAP values untuk top 5 fitur
  14. Build response object

      │
      ▼
[mlflow_logger.py]  ← async, tidak block response
  15. Log ke MLflow:
      - URL hash (MD5, bukan URL asli untuk privacy)
      - Semua fitur
      - Hasil prediksi
      - Inference time

      │
      ▼
[Frontend - ResultCard.tsx]
  16. Render hasil
  17. Simpan ke localStorage (riwayat)
```

### 8.2 Punycode Handling Detail

```python
# Contoh kasus yang ditangani:
CASES = {
    # IDN Homograph Attack
    "https://xn--pple-43d.com"  : "https://аpple.com (Cyrillic а)",
    "https://xn--googl-0ra.com" : "https://gοοgl.com (Greek ο)",

    # Mixed script (Latin + non-Latin)
    "https://paypal-secure.xn--cm" : "Warning: mixed script domain",

    # Normal IDN (bukan serangan)
    "https://xn--80akhbyknj4f.xn--p1ai" : "https://пример.испытание (Russian, legitimate)",
}
```

---

## 9. MLflow Logging Schema

Setiap request dicatat sebagai MLflow run dalam experiment `phishing_detector_inference`:

```python
# Run name  : predict_{timestamp}
# Tags      : {"env": "production", "model_version": "..."}
# Params    : semua fitur (112 nilai)
# Metrics   : {
#     "probability"     : 0.9234,
#     "inference_ms"    : 45,
#     "is_phishing"     : 1,
#     "is_punycode"     : 1,
# }
# Artifacts : (tidak ada — untuk efisiensi)
```

**Dashboard MLflow yang perlu dipantau:**

| Metrik | Alert Threshold | Arti |
|--------|----------------|------|
| `probability` rata-rata per hari | Naik > 20% dari baseline | Mungkin ada serangan baru |
| `inference_ms` p95 | > 3000ms | Backend lambat |
| Rasio `is_phishing` per hari | > 80% | Possible adversarial input |
| Error rate | > 5% | Bug atau model issue |

---

## 10. Non-Functional Requirements

| Requirement | Target |
|-------------|--------|
| Response time (tanpa network features) | < 500ms |
| Response time (dengan network features) | < 5 detik |
| Uptime | 99% (development) |
| Browser support | Chrome 90+, Firefox 88+, Safari 14+ |
| Mobile responsive | Ya — mobile-first design |
| Aksesibilitas | WCAG 2.1 Level AA |

---

## 11. Tech Stack

| Layer | Teknologi | Versi |
|-------|-----------|-------|
| Frontend | Next.js | 14+ (App Router) |
| Frontend styling | Tailwind CSS | 3.x |
| Frontend language | TypeScript | 5.x |
| Backend | FastAPI | 0.110+ |
| Backend language | Python | 3.10+ |
| ML model | LightGBM | 4.x |
| Explainability | SHAP | 0.44+ |
| Experiment tracking | MLflow | 2.10+ |
| Punycode decoding | `idna` | 3.x |
| Network features | `aiohttp`, `dnspython` | latest |
| WHOIS | `python-whois` | latest |

---

## 12. Development Phases

### Phase 1 — Backend Core (2-3 hari)
- [ ] Setup FastAPI app dengan struktur monorepo
- [ ] Implementasi `extractor.py` — Punycode decoding + fitur URL (tanpa network)
- [ ] Implementasi `predictor.py` — load model + preprocessing pipeline + prediksi
- [ ] Endpoint `/predict` dan `/health` berjalan
- [ ] Unit test untuk extractor

### Phase 2 — Frontend Core (2-3 hari)
- [ ] Setup Next.js dengan Tailwind
- [ ] Komponen `URLInput` + validasi
- [ ] Komponen `ResultCard` (phishing/legitimate)
- [ ] Komponen `PunycodeAlert`
- [ ] Koneksi ke backend berjalan end-to-end

### Phase 3 — Advanced Features (2-3 hari)
- [ ] Network feature extraction (async + timeout fallback)
- [ ] Komponen `SHAPChart` — top 5 fitur
- [ ] Komponen `FeatureDetails` — semua fitur expandable
- [ ] MLflow logging per request
- [ ] Riwayat di localStorage

### Phase 4 — Polish (1-2 hari)
- [ ] Loading states dan error handling lengkap
- [ ] Mobile responsive
- [ ] README update
- [ ] (Opsional) Deploy ke Render/Railway + Vercel

---

## 13. Out of Scope (Versi 1.0)

Fitur berikut tidak masuk versi ini, bisa ditambahkan di versi berikutnya:

- **Batch URL check** — submit banyak URL sekaligus via CSV
- **Browser extension** — deteksi langsung di browser
- **Feedback loop** — user bisa report false positive/negative
- **API key authentication** — rate limiting per user
- **Database history** — riwayat tersimpan di server, bukan localStorage
- **Email/WhatsApp integration** — forward pesan, sistem ekstrak URL otomatis
- **Real-time page content scan** — bukan hanya URL, tapi konten halaman

---

*PRD ini adalah living document — update sesuai perkembangan development.*