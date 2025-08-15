# 🎬 Multimodal Movie Recommender — Data → Baseline → Hybrid → Explanations → Personalization

A fast, modern pipeline for a **next-gen movie recommender**.
Start with a solid **data foundation**, ship a **dense retriever + light reranker** baseline, upgrade to **hybrid retrieval (dense + BM25) with language-aware ranking**, add **short explanations**, and finish with **personalization** from seed titles.

> **Attribution:** This product uses the TMDb API but is not endorsed or certified by TMDb.
---
## 📺 Demo
A Streamlit demo that lets you query the movie dataset, use seed titles for personalization, and see results ranked by a hybrid retrieval method.
![Demo – hybrid search + seeds](assets/demo.gif)

---

## 🚀 Quickstart (TL;DR)

```bash
# 1) Create venv
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate

# 2) Install deps (CPU is fine; CUDA optional on Windows)
pip install -r requirements.txt

# 3) Put your TMDb key in .env (see: Prerequisites)

# 4) Build the text index (Step 3A)
python -m src.recsys.build_text_index

# 5) Launch the Streamlit demo
streamlit run src/app/demo.py
```

**Recommended method:** `hybrid` (dense + BM25 + RRF).
**Try this:** Query = `malayalam investigative thriller after 2020`, Seeds = `Drishyam, Memories`, Query mode = `Both`.

---

## ✨ Features

- **Data foundation:** recursive windowing + async credits backfill → typed Parquet
- **Baseline recommender:** embeddings + FAISS + light rerank (Blend / CE / MMR)
- **Hybrid retrieval:** dense + BM25 fused via **RRF** (Reciprocal Rank Fusion)
- **Language-aware ranking:** detect language intent; **soft boost** or **hard keep**
- **Explanations:** short reasons (language/genre/people/year) per result
- **Personalization:** build a user vector from seed titles (“more like these”)
- **Streamlit demo**
- **Light CI:** build index on sample parquet and run one hybrid query

---

## 🗂️ Project Layout

```
├─ data/
│  └─ processed/
│     ├─ movies.parquet                # full typed dataset (not in Git)
│     ├─ movies_parquet/               # partitioned by year
│     ├─ movies_sample.parquet         # small sample kept in Git (50–200 rows)
│     └─ artifacts/
│        ├─ text.index                 # FAISS index (cosine/IP)
│        ├─ text_idmap.parquet         # rowid ↔ movie_id
│        └─ search_payload.parquet     # lean fields for search/rerank
├─ reports/                            # integrity / profiling
│  ├─ data_profile.json
│  └─ checksums.txt
├─ notebooks/
│  └─ 01_eda_movies.ipynb              # portfolio-style EDA
├─ src/
│  ├─ config.py                        # reads .env; paths; device detection
│  ├─ download_dataset.py              # Gradio UI to fetch CSVs
│  ├─ tmdb_api_test.py                 # quick API smoke test
│  ├─ data/prepare_ds_release.py       # CSV → Parquet → sample → reports
│  ├─ recsys/
│  │  ├─ build_text_index.py           # Step 3A (rich doc + embeddings + FAISS)
│  │  ├─ search_and_rerank.py          # Step 3B/4/5/6 (retrieval, CE, MMR, HYBRID, lang policy)
│  │  ├─ eval_proxy.py                 # Step 3C proxy metrics
│  │  ├─ hybrid_sparse.py              # BM25 retriever
│  │  ├─ hybrid_fusion.py              # RRF combiner
│  │  ├─ explanations.py               # Step 5 (reasons)
│  │  └─ user_profiles.py              # Step 6 (seeds → user vector)
│  └─ app/demo.py                      # Streamlit demo (query + seeds)
├─ .github/workflows/ci.yml            # Light CI
├─ requirements.txt
└─ README.md
```

---

## 🔑 Prerequisites

- Python **3.10+**
- Free TMDb API key → https://www.themoviedb.org/settings/api
- (Optional) NVIDIA GPU on **Windows 11** with CUDA for faster encoders

Create **`.env`** at project root:

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

## 🧱 Step 1 — Intake & Validation (CSV → Parquet)

Convert raw CSV → **typed Parquet**, plus a small sample and integrity reports.

```bash
python src/data/prepare_ds_release.py data/movies_2020-01-01_2025-08-08.csv
```

**Outputs:** `data/processed/movies.parquet`, `movies_parquet/` (optional), `movies_sample.parquet`, `reports/`

---

## 📊 Step 2 — EDA

Open `notebooks/01_eda_movies.ipynb`:
- Year trend, Language/Genre mix, Popularity skew, Missingness, Poster gallery

---

## ⚙️ Step 3 — Baseline (embeddings + light rerank)

### 3A — Build the text index
```bash
python -m src.recsys.build_text_index
```

### 3B — Search + rerank (CLI)
```bash
# Blend (retrieval + recency + popularity)
python -m src.recsys.search_and_rerank "smart heist thriller set in Europe" --k 20 --method blend

# Retrieval only
python -m src.recsys.search_and_rerank "lonely space survival drama" --k 20 --method retrieval

# Cross-encoder rerank (small; uses GPU if available)
python -m src.recsys.search_and_rerank "neo-noir crime with witty dialogue" --k 20 --method ce

# MMR diversity (λ≈0.2–0.3)
python -m src.recsys.search_and_rerank "cozy holiday romcom" --k 20 --method mmr --mmr_lambda 0.3
```

### 3C — Tiny proxy eval
```bash
python -m src.recsys.eval_proxy --k 10 --sample_n 200 --method blend
```

### 3D — Streamlit demo
```bash
streamlit run src/app/demo.py
```

---

## 🧪 Step 4 — Hybrid retrieval (dense + BM25) with language-aware ranking

```bash
# Soft language boost (default)
python -m src.recsys.search_and_rerank "malayalam thriller" --method hybrid --k 20

# Hard keep language (filter/top-up)
python -m src.recsys.search_and_rerank "malayalam thriller" --method hybrid --k 20 --lang_policy auto-hard
```
**Knobs:** `--hybrid_dense_k`, `--hybrid_sparse_k`, `--rrf_k`, `--w_dense`, `--w_sparse`, `--lang_policy`

---

## 💡 Step 5 — Explanations

Human-readable reasons per result: **language match**, **genre overlap**, **title keyword**, **year proximity**, with **cast/dir** fallbacks. Visible under each card in the demo.

---

## 🧑‍🤝‍🧑 Step 6 — Personalization (“More like these”)

- Type 2–5 titles in **Seed titles** (sidebar).
- Choose **Query mode**:
  - **Query only** → ignore seeds, use text
  - **Seeds only** → ignore text, build a user vector from seeds
  - **Both** → seeds steer taste; text focuses results (best all-around)
- Recommended **Method:** `hybrid`

---

## 🔎 Query tips

Use natural language: `[genre] + [vibe] + [hook/theme] + [setting] + [constraints]`

Examples:
- `malayalam investigative thriller after 2020`
- `tamil neo-noir crime in chennai`
- `like "Drishyam", tight family crime with twists`
- `heist comedy set in europe 2000s`
- `anime coming-of-age with music and friendship`

If results feel same-y → `--method mmr --mmr_lambda 0.25`
If too niche → set **Min vote_count = 0** in the demo.

---

## 🧩 Troubleshooting

- **HF 429 / auth** → add `HUGGING_FACE_HUB_TOKEN` to `.env` or pre-download models; set `HF_HOME=.hf_cache`
- **FAISS shape error** → vector must be `(1, d)` float32 (handled in code)
- **Few results with MMR + min votes** → set min votes to 0 (demo will top-up)
- **Streamlit import error** → run from repo root; `demo.py` bootstraps paths

---

## ✅ CI

Add `.github/workflows/ci.yml` with a tiny smoke test that:
1) Installs deps
2) Copies `movies_sample.parquet` → `movies.parquet`
3) Builds the index
4) Runs **one hybrid query** and uploads CSV results

---

## 🧭 Future Work

- Entity-aware boosts (actors/directors) during ranking
- API (FastAPI) + Docker
- Multimodal fusion (CLIP poster embeddings + text)
- Better small embedder/reranker (e.g., bge-m3)

---

## 📜 License

MIT
