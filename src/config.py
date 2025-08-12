import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import torch

# ---------- Project & .env ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# Optional: expose a TMDB key if other scripts use it
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# ---------- Device / perf toggles ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")  # TF32 on Ampere+

# Half precision for embedding models (safe for MiniLM/bge-small)
USE_FP16 = os.getenv("ST_FP16", "1") == "1" and DEVICE == "cuda"

# FAISS-GPU is OFF by default on native Windows; keep CPU FAISS.
USE_FAISS_GPU = os.getenv("USE_FAISS_GPU", "0") == "1"
FAISS_FP16 = os.getenv("FAISS_FP16", "0") == "1"  # store/search in fp16 on GPU

# ---------- Models ----------
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
CROSS_ENCODER_NAME = os.getenv("CROSS_ENCODER_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# ---------- Data folders (Path objects) ----------
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROCESSED_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Files (Path objects) ----------
MOVIES_PARQUET = PROCESSED_DIR / "movies.parquet"
TEXT_INDEX = ARTIFACTS_DIR / "text.index"
TEXT_IDMAP = ARTIFACTS_DIR / "text_idmap.parquet"
SEARCH_PAYLOAD = ARTIFACTS_DIR / "search_payload.parquet"

# ---------- Misc ----------
YEAR_NOW = int(os.getenv("YEAR_NOW", datetime.now().year))

# ---------- Notes ----------
# - Keeping Path objects ensures build_text_index.py’s `MOVIES_PARQUET.exists()` works.
# - Encoders (SentenceTransformers/CrossEncoder) will use CUDA automatically.
# - If you run under WSL2 + Ubuntu and want FAISS-GPU, set USE_FAISS_GPU=1.
