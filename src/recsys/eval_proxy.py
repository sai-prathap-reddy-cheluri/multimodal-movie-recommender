from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd

from src.config import MOVIES_PARQUET, ARTIFACTS_DIR
from src.recsys.search_and_rerank import retrieve
from search_and_rerank import jlist

def genre_jaccard(a, b):
    """Jaccard similarity for genre lists, robust to JSON or list inputs."""
    a, b = set(jlist(a)), set(jlist(b))
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def eval_one(seed_row: pd.Series, k: int = 10, method: str = "blend"):
    """Evaluate a single movie row against the retrieval method."""
    q = seed_row.get("title") or seed_row.get("doc") or ""
    res = retrieve(q, k=k+1, method=method)  # ask for one extra to drop self
    # drop exact self if present
    mid = seed_row.get("movie_id")
    if mid is not None and "movie_id" in res.columns:
        res = res[res["movie_id"] != mid]

    g0 = seed_row.get("genres") or seed_row.get("genre_ids_json", "[]")
    lang0 = seed_row.get("original_language")
    y0 = seed_row.get("year")

    gj, gap, langmatch = [], [], []
    for _, rr in res.head(k).iterrows():
        gj.append(genre_jaccard(g0, rr.get("genres") or rr.get("genre_ids_json", "[]")))
        y = rr.get("year")
        if pd.notna(y0) and pd.notna(y):
            try:
                gap.append(abs(int(y0) - int(y)))
            except Exception:
                pass
        langmatch.append(1 if rr.get("original_language") == lang0 else 0)

    return {
        "genre@k": float(np.mean(gj)) if gj else np.nan,
        "year_gap@k": float(np.median(gap)) if gap else np.nan,
        "lang_match@k": float(np.mean(langmatch)) if langmatch else np.nan,
    }

def run_eval(k: int = 10, sample_n: int = 200, method: str = "blend", seed: int = 42) -> pd.DataFrame:
    """Run evaluation on a sample of movies."""
    df = pd.read_parquet(MOVIES_PARQUET)
    if len(df) == 0:
        raise RuntimeError("Empty movies parquet.")
    df = df.sample(min(sample_n, len(df)), random_state=seed)

    rows = []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        try:
            m = eval_one(row, k=k, method=method)
            m["i"] = i
            rows.append(m)
        except Exception as e:
            # skip bad row, continue
            rows.append({"i": i, "error": str(e)})
            continue
    return pd.DataFrame(rows)

def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--sample_n", type=int, default=200)
    ap.add_argument("--method", type=str, default="blend", choices=["blend","retrieval","ce","mmr","hybrid"])
    args = ap.parse_args()

    dfm = run_eval(k=args.k, sample_n=args.sample_n, method=args.method)
    out_csv = ARTIFACTS_DIR / f"eval_proxy_{args.method}.csv"
    dfm.to_csv(out_csv, index=False)

    # Simple summary
    good = dfm.dropna(subset=["genre@k","year_gap@k","lang_match@k"], how="any")
    summary = {
        "n": int(len(good)),
        "genre@k_mean": float(good["genre@k"].mean()) if "genre@k" in good else None,
        "year_gap@k_median": float(good["year_gap@k"].median()) if "year_gap@k" in good else None,
        "lang_match@k_mean": float(good["lang_match@k"].mean()) if "lang_match@k" in good else None,
    }
    out_json = ARTIFACTS_DIR / f"eval_proxy_{args.method}_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Saved {out_csv}\nSummary -> {out_json}\n{json.dumps(summary, indent=2)}")

if __name__ == "__main__":
    cli()