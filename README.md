# 🎬 Multimodal Movie Recommender — Data → Baseline → Hybrid

A fast, modern pipeline for building a **next‑gen movie recommender**.
Start with a solid **data foundation**, then ship a **dense retriever + light reranker** baseline, and finally upgrade to **hybrid retrieval (dense + BM25) with language‑aware ranking**.

> **Attribution:** This product uses the TMDb API but is not endorsed or certified by TMDb.

---

## ✨ Features

- **Recursive windowing** to bypass TMDb’s 10k results/query cap
- **Async credits backfill** (runtime, full cast, directors) with controlled concurrency
- **Typed Parquet** + optional partitioning (fast loading, analytics‑friendly)
- **Baseline recommender**: embeddings + FAISS + light rerank (blend / CE / MMR)
- **Hybrid retrieval**: dense (embeddings) + sparse (BM25) fused via **RRF**
- **Language‑aware ranking**: auto‑detect language intent in query; **soft boost** or **hard filter**
- **Streamlit demo** (GPU‑aware on Windows 11) and **tiny offline eval**

---

## 🗂️ Project Layout

```
├─ data/
│  └─ processed/
│     ├─ movies.parquet                # full typed dataset
│     ├─ movies_parquet/               # partitioned by year
│     ├─ movies_sample.parquet         # small sample kept in Git
│     └─ artifacts/
│        ├─ text.index                 # FAISS index (cosine/IP)
│        ├─ text_idmap.parquet         # rowid ↔ movie_id
│        └─ search_payload.parquet     # lean fields for search/rerank
├─ reports/
│  ├─ data_profile.json
│  └─ checksums.txt
├─ notebooks/
│  └─ 01_eda_movies.ipynb              # portfolio‑style EDA
├─ src/
│  ├─ config.py                        # reads .env; paths; device detection
│  ├─ download_dataset.py              # Gradio UI to fetch CSVs
│  ├─ tmdb_api_test.py                 # quick API smoke test
│  ├─ data/
│  │  └─ data_preparation.py           # CSV → Parquet → sample → reports
│  ├─ recsys/
│  │  ├─ build_text_index.py           # Step 3A (doc builder + embeddings + FAISS)
│  │  ├─ search_and_rerank.py          # Step 3B + Step 4 (retrieval, CE, MMR, HYBRID)
│  │  ├─ eval_proxy.py                 # Step 3C proxy metrics
│  │  ├─ hybrid_sparse.py              # BM25 retriever
│  │  └─ hybrid_fusion.py              # RRF combiner
│  └─ app/
│     └─ demo.py                       # Streamlit demo
├─ requirements.txt
└─ README.md
```

---

## 🔑 Prerequisites

- Python **3.10+**
- A free TMDb API key → https://www.themoviedb.org/settings/api
- (Optional but recommended) NVIDIA GPU on **Windows 11** with CUDA for faster encoders

Create a **`.env`** in the project root:

```ini
TMDB_API_KEY=YOUR_TMDB_KEY_HERE

# Hugging Face (avoid 429s)
HUGGING_FACE_HUB_TOKEN=hf_xxx
HF_HOME=.hf_cache

# Models / perf (override if desired)
EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
CROSS_ENCODER_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2
ST_FP16=1
```

---

## 🧰 Setup

```bash
# create & activate venv (Windows PowerShell)
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate

# install CUDA PyTorch (Windows with CUDA 12.1; adjust if needed)
pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# install project deps
pip install -r requirements.txt
```

### ✅ Verify your TMDb API setup

```bash
# from repo root
python -m src.tmdb_api_test
```

---

## 🚀 Download a Dataset (Gradio UI)

```bash
python -m src.download_dataset
```
- Choose a date range (uses recursive windowing under-the-hood)
- Toggle **Include adult (18+)** if relevant
- **Concurrency** 12–20 is a good start
- Outputs CSV under `data/`

### 🔁 Backfill Missing Credits (CLI)

```bash
python -m src.scripts.backfill_credits --csv data/movies_YYYY-MM-DD_YYYY-MM-DD.csv --concurrency 12 --batch-size 800
```
What it does:
- Reads CSV
- Finds rows where **both** actors & directors are blank
- Fetches credits with retries + exponential backoff + jitter
- Writes progress back to the same CSV each batch (safe stop/resume)

---

## 🧱 Step 1 — Intake & Validation (CSV → Parquet)

Convert raw CSV → **typed Parquet**, plus a small sample and integrity reports.

```bash
python src/data/prepare_ds_release.py data/movies_2020-01-01_2025-08-08.csv
```

**Outputs:**
- `data/processed/movies.parquet`
- `data/processed/movies_parquet/` (partitioned)
- `data/processed/movies_sample.parquet`
- `reports/data_profile.json`, `reports/checksums.txt`

---

## 📊 Step 2 — Exploratory Data Analysis (EDA)

Open `notebooks/01_eda_movies.ipynb` for a compact EDA:
- Year trend 🗓️, Runtime ⏱️
- Language & Genre mix 🌍
- Popularity skew 📈
- Missingness heatmap 🧼
- Poster gallery 🎞️

**What it tells us:**
- Strong recent coverage → time‑based splits
- Popularity is skewed → add semantic retrieval to reduce bias
- Missingness is localized → impute/skip per‑feature

---

## ⚙️ Step 3 — Baseline “next‑gen” recommender (embeddings + light rerank)

Build a fast, modern baseline: **dense retrieval** over **rich docs** + tiny reranker.

### 3A — Build the text index
```bash
python -m src.recsys.build_text_index
```
Creates:
- `data/processed/artifacts/text.index` (FAISS, cosine/IP)
- `data/processed/artifacts/text_idmap.parquet`
- `data/processed/artifacts/search_payload.parquet` (includes the **doc**: title · overview · top cast/crew · genres · **language** · year)

### 3B — Search + rerank (CLI)
```bash
# Blend (retrieval + recency + popularity)
python -m src.recsys.search_and_rerank "smart heist thriller set in Europe" --k 20 --method blend

# Retrieval only (no rerank)
python -m src.recsys.search_and_rerank "lonely space survival drama" --k 20 --method retrieval

# Cross‑encoder rerank (small; uses GPU if available)
python -m src.recsys.search_and_rerank "neo‑noir crime with witty dialogue" --k 20 --method ce

# MMR diversity (good default λ≈0.2–0.3)
python -m src.recsys.search_and_rerank "cozy holiday romcom" --k 20 --method mmr --mmr_lambda 0.3
```

### 3C — Tiny proxy eval
```bash
python -m src.recsys.eval_proxy --k 10 --sample_n 200 --method blend
```
Outputs CSV + summary JSON in `artifacts/`.

### 3D — Streamlit demo
```bash
streamlit run src/app/demo.py
```
Use the sidebar to:
- Enter a taste query
- Choose **method** (blend / retrieval / ce / mmr / **hybrid**)
- Set **Top‑K**, **MMR λ** (for MMR), and **Min vote_count**

---

## 🧪 Step 4 — Hybrid retrieval (dense + BM25) with language‑aware ranking

**Why:** exact names/franchises/misspellings + strong semantic recall is the 2025 default.
**How:** fuse **dense (FAISS)** and **sparse (BM25)** with **RRF**; detect language intent and **boost** or **hard‑keep** that language.

### Use it
```bash
# RRF fusion + soft language boost (default)
python -m src.recsys.search_and_rerank "malayalam thriller" --method hybrid --k 20

# Hard language preference (keep Malayalam first, then top‑up)
python -m src.recsys.search_and_rerank "malayalam thriller" --method hybrid --k 20 --lang_policy auto-hard
```

**Knobs:**
- `--hybrid_dense_k / --hybrid_sparse_k` – candidate pool sizes (default 500/500)
- `--rrf_k` – RRF constant (typical 50–100; default 60)
- `--w_dense / --w_sparse` – source weights (default 1.0/1.0)
- `--lang_policy` – `off | auto-soft | auto-hard`

**What’s under the hood:**
- `hybrid_sparse.py` builds an in‑memory **BM25** over `doc`
- `search_and_rerank.py` returns **rowid** from FAISS; hybrid uses RRF on rowids
- Language intent is **data‑driven** (codes + spoken names discovered from payload)
- Query is augmented with “**Language: <Name>**” so both dense & BM25 see it
- Policy: **soft boost** (multiply scores) or **hard keep** (filter/top‑up)

---

## 🔎 How to search (good queries)

**Recipe:** `[genre] + [vibe] + [hook/theme] + [setting/locale] + [constraints]`

Examples (paste as‑is):
- “slow‑burn sci‑fi about isolation in space”
- “Indian Malayalam investigative thriller after 2020”
- “like ‘Drishyam’, tight family crime with twists”
- “anime coming‑of‑age with music and friendship”
- “Tamil neo‑noir crime in Chennai”
- “French heist comedy 2000s”

**Tips:**
- Prefer **natural language** over boolean syntax
- For variety: `--method mmr --mmr_lambda 0.25`
- For exact names/franchises: `--method hybrid`
- If results feel too niche: raise **Min vote_count** in the demo

---

## 🧩 Troubleshooting

- **HF 429 / auth** → add `HUGGING_FACE_HUB_TOKEN` to `.env` or pre‑download models; set `HF_HOME=.hf_cache`
- **FAISS error: `n, d = x.shape`** → ensure query vector is **(1, d) float32** (already handled in code)
- **pd.NA slicing** → code uses NA‑safe helpers; rebuild if you changed doc logic
- **Streamlit import error** → run from repo root or use the path bootstrap in `demo.py`
- **Few results with MMR + min votes** → set min votes to 0 or use the “top‑up” filter; increase MMR pool

---

## 🧭 Roadmap

- Poster caching on demand
- **Hybrid by default** in demo, with entity‑aware boosts (actors/directors)
- Explanations (“Because you liked… / shares actor Y / genre Z”)
- Personalization (seed titles → user vector)
- FastAPI + Docker for deployment
- Multimodal ranking (CLIP poster embeddings + text)

---

## 📜 License

MIT

