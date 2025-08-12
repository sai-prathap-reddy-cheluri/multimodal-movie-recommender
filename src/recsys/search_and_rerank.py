from __future__ import annotations
import argparse
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.config import (
    TEXT_INDEX, SEARCH_PAYLOAD, TEXT_IDMAP,
    EMBED_MODEL_NAME, CROSS_ENCODER_NAME,
    DEVICE, YEAR_NOW
)

def jlist(x):
    if isinstance(x, list):
        return x
    if x is None:
        return []
    try:
        if isinstance(x, str):
            y = json.loads(x)
        else:
            y = x
        return y if isinstance(y, list) else []
    except Exception:
        return []

@dataclass(frozen=True)
class SearchConfig:
    k: int = 20
    method: str = "blend"  # "retrieval" | "blend" | "ce" | "hybrid"
    mmr_lambda: float = 0.3  # used if method == "mmr"
    use_ce: bool = False      # load CE only if needed
    year_now: int = YEAR_NOW

def _coerce_cfg(cfg: Any) -> SearchConfig:
    if isinstance(cfg, SearchConfig):
        return cfg
    if isinstance(cfg, dict):
        return SearchConfig(**cfg)
    if isinstance(cfg, tuple):
        # assume tuple in canonical order
        keys = list(SearchConfig().__dict__.keys())
        return SearchConfig(**{k: v for k, v in zip(keys, cfg)})
    # last resort defaults
    return SearchConfig()

class TextRetriever:
    def __init__(self, cfg: SearchConfig):
        self.cfg = _coerce_cfg(cfg)
        # Load artifacts
        self.index = faiss.read_index(str(TEXT_INDEX))
        self.payload = pd.read_parquet(SEARCH_PAYLOAD)
        self.idmap = pd.read_parquet(TEXT_IDMAP) if Path(TEXT_IDMAP).exists() else None
        # Encoders
        self.embedder = SentenceTransformer(str(EMBED_MODEL_NAME), device=DEVICE)
        self._ce: Optional[CrossEncoder] = None

    def _ensure_ce(self):
        if self._ce is None:
            self._ce = CrossEncoder(str(CROSS_ENCODER_NAME), device=DEVICE)

    def encode_query(self, query: str) -> np.ndarray:
        qv = self.embedder.encode([query], normalize_embeddings=True)
        return qv.astype(np.float32)

    def search(self, query: str, k: int = 20) -> pd.DataFrame:
        qv = self.encode_query(query)
        scores, idx = self.index.search(qv, k)
        res = self.payload.iloc[idx[0]].copy()
        res["score"] = scores[0]
        return res

    @staticmethod
    def _mmr(query_vec: np.ndarray, cand_vecs: np.ndarray, lambda_mult: float = 0.3, top_k: int = 20) -> List[int]:
        """MMR over normalized vectors; returns indices of selected items."""
        # Similarity to query
        sim_to_query = (cand_vecs @ query_vec.T).reshape(-1)
        selected = []
        # Precompute candidate-to-candidate sims
        cc = cand_vecs @ cand_vecs.T  # cosine since normalized
        remaining = list(range(len(cand_vecs)))

        while remaining and len(selected) < top_k:
            if not selected:
                # pick best to query first
                best = int(np.argmax(sim_to_query[remaining]))
                selected.append(remaining.pop(best))
                continue
            # compute MMR for remaining
            sub = np.array(remaining)
            max_sim_to_selected = cc[np.ix_(sub, selected)].max(axis=1)
            mmr_score = lambda_mult * sim_to_query[sub] - (1 - lambda_mult) * max_sim_to_selected
            pick = int(np.argmax(mmr_score))
            selected.append(remaining.pop(pick))
        return selected

    def rerank(self, query: str, res: pd.DataFrame, method: str = "blend", mmr_lambda: float = 0.3) -> pd.DataFrame:
        method = (method or self.cfg.method or "blend").lower()
        year_now = getattr(self.cfg, "year_now", YEAR_NOW)
        out = res.copy()

        if method in ("retrieval", "none"):
            return out.sort_values("score", ascending=False)

        if method in ("ce", "cross", "cross_encoder"):
            self._ensure_ce()
            pairs = [[query, d] for d in out["doc"].tolist()]
            ce_scores = self._ce.predict(pairs, batch_size=64, convert_to_numpy=True, show_progress_bar=False)
            out["rerank"] = ce_scores
            return out.sort_values("rerank", ascending=False)

        # default: blend retrieval + simple priors (recency, popularity)
        y = pd.to_numeric(out.get("year"), errors="coerce").fillna(year_now)
        recency = 1.0 / (1.0 + np.maximum(0, year_now - y))
        pop = (pd.to_numeric(out.get("vote_count"), errors="coerce").fillna(0)).clip(0, 20000) / 20000.0
        out["blend"] = 0.6 * out["score"].astype(float) + 0.25 * recency + 0.15 * pop
        out = out.sort_values("blend", ascending=False)

        if method == "mmr":
            # Run MMR on top-200 using internal vectors for better diversity
            # Re-embed top-200 docs for diversity step
            topn = min(200, len(out))
            tops = out.head(topn)
            doc_vecs = self.embedder.encode(
                tops["doc"].tolist(), batch_size=128,
                convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True
            ).astype(np.float32)
            qv = self.encode_query(query)
            order_idx = self._mmr(qv, doc_vecs, lambda_mult=mmr_lambda, top_k=min(self.cfg.k, topn))
            out = tops.iloc[order_idx]
        return out

@lru_cache(maxsize=2)
def get_retriever(cfg_tuple: Tuple) -> TextRetriever:
    """LRU-cached retriever keyed by a tuple of config values (so we don't reload models repeatedly)."""
    cfg = _coerce_cfg(cfg_tuple)
    return TextRetriever(cfg)

def retrieve(query: str, k: int = 20, method: str = "blend", mmr_lambda: float = 0.3) -> pd.DataFrame:
    cfg = SearchConfig(k=k, method=method, mmr_lambda=mmr_lambda)
    tr = get_retriever(tuple(asdict(cfg).values()))
    res = tr.search(query, k=max(k, 200 if method == "mmr" else k))
    out = tr.rerank(query, res, method=method, mmr_lambda=mmr_lambda)
    return out.head(k)

def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", type=str, help="free-text taste or title")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--method", type=str, default="blend", choices=["retrieval","blend","ce","mmr"])
    ap.add_argument("--mmr_lambda", type=float, default=0.3)
    args = ap.parse_args()

    out = retrieve(args.query, k=args.k, method=args.method, mmr_lambda=args.mmr_lambda)
    cols = [c for c in ["title","year","score","blend","rerank","poster_url"] if c in out.columns]
    print(out[cols].head(args.k).to_string(index=False))

if __name__ == "__main__":
    cli()