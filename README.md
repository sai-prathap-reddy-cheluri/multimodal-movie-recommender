# 🎬 Multimodal Movie Recommender – Data Foundation

A fast, modern data pipeline for building a next‑gen movie recommender.
It uses async parallel fetching to gather TMDb movies for a time range, then backfills runtime, actors (full cast), and directors via credits. From there, we convert to typed Parquet, publish a small sample for reviewers, and keep the full dataset as a release asset. This repo is tuned for portfolio-readability and reproducibility.

Attribution: This product uses the TMDb API but is not endorsed or certified by TMDb.

## ✨ Features
- **Recursive windowing** to bypass TMDb’s 10k results/query cap
- **Async credits fetch** (runtime, full cast, directors) with controlled concurrency
- **Robust backfill** script: retries, exponential backoff, batch progress writes
- Clean, reproducible **Gradio UI** to download datasets quickly
- Typed Parquet + partitioning (fast loading, analytics‑friendly)

## 🗂️ Project Layout

```
├─ data/                    # Raw CSVs you generate (git‑ignored; keep only small samples)
│  └─ processed/
│     ├─ movies.parquet               # full typed dataset (Release asset, not in Git)
│     ├─ movies_parquet/              # partitioned by year/ (Release asset)
│     ├─ movies_sample.parquet        # small sample kept in Git
│     └─ splits/                      # optional time-based splits
├─ reports/
│  ├─ data_profile.json               # row counts, nulls, dtypes
│  └─ checksums.txt                   # sha256 for integrity verification
├─ notebooks/
│  └─ 01_eda_movies.ipynb             # visual EDA with short commentary
├─ src/
│  ├─ config.py                       # reads .env, defines DATA_DIR, etc.
│  ├─ download_dataset.py             # Gradio app (UI) to download datasets
│  ├─ tmdb_api_test.py                # quick API smoke test
│  ├─ data/
│  │  └─ data_preparation.py        # intake → Parquet → sample → reports
│  ├─ scripts/
│  │  └─ backfill_credits.py          # CLI backfill for blank actors/directors
├─ requirements.txt
└─ README.md
```

## 🔑 Prerequisites
- Python 3.10+
- A free TMDb API key: https://www.themoviedb.org/settings/api

Prerequisite: .env in the project root with:
```ini
TMDB_API_KEY=YOUR_TMDB_KEY_HERE
```
## 🧰 Setup
```bash
# create & activate venv (Windows PowerShell)
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate

# install deps
pip install -r requirements.txt
```

## ✅ Verify your TMDb API setup

You can quickly check that your API key and network access are working by running the test script **`src/tmdb_api_test.py`**. It will call TMDb and print the top 5 popular movies.

> Run these commands **from the project root** (where your `.env` lives).

**Windows / PyCharm terminal**
```powershell
python -m src.tmdb_api_test
# or
python src\tmdb_api_test.py
```

```bash
python -m src.tmdb_api_test
# or
python src/tmdb_api_test.py
```

## 🚀 Run the Downloader (Gradio UI)

Launch the UI and download a dataset for any date range; it will also enrich with runtime, full cast, and directors.
```bash
# from repo root
python -m src.download_dataset
```
* Choose a preset date range or set custom dates.
* Toggle Include adult (18+) as desired.
* Set Concurrency (parallel credit requests). Start with 12–20.
* Output CSV: data/movies_YYYY-MM-DD_YYYY-MM-DD.csv.

## 🔁 Backfill Missing Credits (CLI)

While fetching the actors and director if API times out and the actors and director list is blank,  then can backfill rows where both actors and directors are empty, run:

```bash
# Windows (PyCharm terminal uses the project interpreter)
python -m src.scripts.backfill_credits --csv data/movies_2020-01-01_2025-08-08.csv --concurrency 12 --batch-size 800

# macOS / Linux
python -m src.scripts.backfill_credits --csv data/movies_2020-01-01_2025-08-08.csv --concurrency 12 --batch-size 800
```

What it does
* Reads your CSV.
* Finds rows where both actors and directors are blank.
* Fetches credits with retries + exponential backoff + jitter.
* Writes progress back to the same CSV after each batch so you can safely stop/resume.

## ⚙️ Concurrency & Batch Size

* Concurrency (credits): start 12–16, cap around 20–24.
* Batch size: 500–1000 IDs per batch is a sweet spot.
* If you see 429 Too Many Requests or timeouts, lower concurrency and/or batch size.

### 🧪 Quick Smoke Test
```bash
# tiny range to validate everything
python -m src.scripts.backfill_credits --csv data/movies_2025-01-01_2025-01-31.csv --concurrency 8 --batch-size 300
```

## 🧱 Step 1 — Intake & Validation (convert CSV → Parquet)

Turn the raw CSV into a typed, analytics‑ready Parquet dataset, plus a 10k sample and integrity reports. This step also builds absolute poster/backdrop URLs and standardizes list fields to JSON strings.

```
# requires: pandas, pyarrow
python src/data/prepare_ds_release.py data/movies_2020-01-01_2025-08-08.csv
```

### Outputs:
* data/processed/movies.parquet
* data/processed/movies_parquet/ (partitioned by year)
* data/processed/movies_sample.parquet
* reports/data_profile.json, reports/checksums.txt

### Viewing Parquet
```
import pandas as pd, duckdb
print(pd.read_parquet('data/processed/movies.parquet').head())
con = duckdb.connect()
print(con.execute("SELECT year, COUNT(*) FROM 'data/processed/movies_parquet' GROUP BY year ORDER BY year").df())
```

## 📊 Step 2 — Exploratory Data Analysis (EDA)

A compact, portfolio-ready EDA to understand coverage, data quality, and biases before modeling.
### What you’ll see
- Year trend 🗓️ (coverage & recency), Runtime ⏱️ (typical lengths & outliers)
- Language & Genre mix 🌍
- Popularity skew (votes vs. popularity) 📈
- Missingness heatmap 🧼
- A mini poster gallery for quick eyeballing 🎞️

### Run it

Open `notebooks/01_eda_movies.ipynb`.

### ✅ What this EDA tells us
- We have strong recent coverage → use time-based splits.
- Runtimes have outliers → clip to p95 when featurizing.
- Popularity is skewed → add semantic retrieval to reduce popularity bias.
- Missingness is localized → impute/skip per-feature, don’t blanket-drop rows.

## Step 3 — Baseline “next-gen” recommender (embeddings + light rerank)

Build a fast, modern baseline that looks great in a portfolio: dense retrieval over rich movie text, then a tiny reranker for relevance + optional diversity.

### What you’ll build
- **Text retriever (dense):** create a rich doc per movie (title · overview · top cast/crew · genres · year), embed with a compact model, and index with FAISS.
- **Light reranker:** sort the top candidates with either a **blend** (retrieval score + recency + popularity) or a **small cross-encoder**.
- **Diversity (optional):** **MMR** (Maximal Marginal Relevance) to avoid near-duplicates.
- **Tiny eval:** quick proxy metrics to claim improvements (genre overlap@k, year gap, language match).
- **Mini demo:** Streamlit one-box search with posters.

> **Trend note (verified best practice):** Hybrid **dense+sparse** retrieval plus a **light reranker** is the current default pattern for modern recommenders/search. Add MMR for diversity and you’ve got a strong baseline.

---

### Artifacts produced
- `data/processed/artifacts/text.index` — FAISS index (cosine/IP).
- `data/processed/artifacts/text_idmap.parquet` — rowid ↔ movie_id map.
- `data/processed/artifacts/search_payload.parquet` — lean fields for display/rerank.
- `data/processed/artifacts/eval_proxy_[method].csv|json` — quick offline metrics.

---

### Run it (Step 3A–3D)

> **Windows 11 + CUDA (verified):** GPU is used for embeddings/rerank; keep FAISS on CPU.

**3A — Build the text index**
```bash
python -m src.recsys.build_text_index
```

**3B — Search + rerank
```bash
# Blend (default): retrieval + recency + popularity
python -m src.recsys.search_and_rerank "smart heist thriller set in Europe" --k 20 --method blend

# Retrieval only
python -m src.recsys.search_and_rerank "lonely space survival drama" --k 20 --method retrieval

# Cross-encoder rerank (small, runs on GPU)
python -m src.recsys.search_and_rerank "neo-noir crime with witty dialogue" --k 20 --method ce

# MMR diversity (more variety; λ≈0.2–0.3 is a good default)
python -m src.recsys.search_and_rerank "cozy holiday romcom" --k 20 --method mmr --mmr_lambda 0.3
```

**3C — Tiny proxy eval
```bash
python -m src.recsys.eval_proxy --k 10 --sample_n 200 --method blend
```

**3D — Streamlit demo
```bash
streamlit run src/app/demo.py
```
### Configuration

- Models (env-overridable):
    * EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
    * CROSS_ENCODER_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2

- Performance toggles:
    * ST_FP16=1 (use half precision on CUDA-capable GPUs)
    * HF_HOME=.hf_cache (shorter Windows paths)

- Hugging Face auth (avoid 429s):
    * .env: HUGGING_FACE_HUB_TOKEN=hf_xxx
    * (Optional) pre-download model and set EMBED_MODEL_NAME to the local folder

### How to search (good queries)

Use natural language: [genre] + [vibe] + [hook/theme] + [setting/locale] + [constraints]
- “slow-burn sci-fi about isolation in space”
- “Indian Malayalam investigative thriller after 2020”
- “like ‘Drishyam’, tight family crime with twists”
- “anime coming-of-age with music and friendship”

If results feel same-y → use --method mmr --mmr_lambda 0.25.
If results feel too niche → raise min vote_count in the demo or add broader vibe words.

## 🧭 Roadmap

* Poster caching on-demand
* Embeddings & vector store for RAG-augmented recommendations
* Multimodal ranking with poster/text features

## 📜 License

MIT

---
