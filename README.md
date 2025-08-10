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
├─ src/
│  ├─ config.py                       # reads .env, defines DATA_DIR, etc.
│  ├─ download_dataset.py             # Gradio app (UI) to download datasets
│  ├─ tmdb_api_test.py                # quick API smoke test
│  ├─ data/
│  │  └─ prepare_ds_release.py        # intake → Parquet → sample → reports
│  ├─ scripts/
│  │  ├─ backfill_credits.py          # CLI backfill for blank actors/directors
│  │  └─ make_splits.py               # create time‑based train/val/test
│  └─ models/
│     ├─ embed_movies.py              # text embeddings + ANN index (baseline)
│     └─ embed_posters.py             # (optional) CLIP poster embeddings
├─ .env                                # TMDB_API_KEY=...
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
python data/prepare_ds_release.py data/movies_2020-01-01_2025-08-08.csv
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

## 🧭 Roadmap

* Poster caching on-demand
* Embeddings & vector store for RAG-augmented recommendations
* Multimodal ranking with poster/text features

## 📜 License

MIT

---
