from __future__ import annotations
import argparse
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union, Sequence
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
from src.recsys.hybrid_fusion import rrf_fuse
from src.recsys.hybrid_sparse import SparseBM25

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

def build_lang_aliases(df: pd.DataFrame) -> dict[str, tuple[str,str]]:
    """
    Returns {alias_lower: (iso_code, english_name)} from payload.
    Aliases include ISO codes and spoken language names observed in data.
    """
    aliases: dict[str, tuple[str,str]] = {}
    # From original_language codes
    if "original_language" in df.columns:
        for code in (
            df["original_language"]
            .dropna().astype(str).str.lower().unique().tolist()
        ):
            name = code
            try:
                from pycountry import languages
                rec = languages.get(alpha_2=code)
                if rec and rec.name:
                    name = rec.name
            except Exception:
                pass
            aliases[code] = (code, name)
            aliases[name.lower()] = (code, name)

    colname = "spoken_languages" if "spoken_languages" in df.columns else (
        "spoken_languages_json" if "spoken_languages_json" in df.columns else None
    )
    if colname:
        for cell in df[colname].dropna().head(5000):  # cap for speed
            lst = jlist(cell)
            if lst:
                # dict form
                if isinstance(lst[0], dict):
                    for d in lst:
                        nm = d.get("english_name") or d.get("name")
                        cd = (d.get("iso_639_1") or "").lower()
                        if nm:
                            aliases[nm.lower()] = (cd or aliases.get(nm.lower(), ("", nm))[0], nm)
                # string form
                elif isinstance(lst[0], str):
                    for nm in lst:
                        if nm:
                            key = nm.lower()
                            if key not in aliases:
                                aliases[key] = ("", nm)
    return aliases

def detect_lang_intent(text: str, aliases: dict[str, tuple[str,str]]):
    t = text.lower()
    # exact token contains (fast path)
    for k, (code, name) in aliases.items():
        if k and k in t:
            return code, name
    return None, None

def apply_lang_policy(
    df: pd.DataFrame,
    lang_code: str | None,
    lang_name: str | None,
    k: int,
    strict: bool = False,
    boost: float = 1.35,
) -> pd.DataFrame:
    """
    If strict=True: keep rows matching language first (hard preference), then top-up.
    Else: softly boost matching rows on the first available score column.
    """
    if df is None or df.empty or (not lang_code and not lang_name):
        return df.head(k)

    code_mask = False
    if lang_code and "original_language" in df.columns:
        code_mask = df["original_language"].astype(str).str.lower() == lang_code

    name_mask = False
    if lang_name and "doc" in df.columns:
        name_mask = df["doc"].astype(str).str.contains(fr"\b{lang_name}\b", case=False, regex=True)

    if isinstance(code_mask, bool) and isinstance(name_mask, bool):
        # neither column exists; nothing to do
        return df.head(k)

    mask = (code_mask | name_mask)

    if strict:
        prefer = df[mask]
        if len(prefer) >= k:
            return prefer.head(k)
        need = k - len(prefer)
        rest = df[~mask].head(need)
        return pd.concat([prefer, rest], ignore_index=True)

    # soft boost on the first available score column
    out = df.copy()
    bonus = np.where(mask, boost, 1.0)
    for col in ("hybrid", "blend", "rerank", "rrf", "score"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[col] = out[col] * bonus
            out = out.sort_values(col, ascending=False)
            break
    return out.head(k)

@dataclass(frozen=True)
class SearchConfig:
    k: int = 20
    method: str = "blend"  # "retrieval" | "blend" | "ce" | "hybrid"
    mmr_lambda: float = 0.3  # used if method == "mmr"
    use_ce: bool = False      # load CE only if needed
    year_now: int = YEAR_NOW

def coerce_cfg(cfg: Any) -> SearchConfig:
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

def qtext(q: Union[str, Sequence[str]]) -> str:
    if isinstance(q, (list, tuple)):
        return ". ".join(map(str, q))  # multi-query: join
    return str(q)

class TextRetriever:
    def __init__(self, cfg: SearchConfig):
        self.cfg = coerce_cfg(cfg)
        # Load artifacts
        self.index = faiss.read_index(str(TEXT_INDEX))
        self.payload = pd.read_parquet(SEARCH_PAYLOAD)
        self.lang_aliases = build_lang_aliases(self.payload)
        self.idmap = pd.read_parquet(TEXT_IDMAP) if Path(TEXT_IDMAP).exists() else None
        # Encoders
        self.embedder = SentenceTransformer(str(EMBED_MODEL_NAME), device=DEVICE)
        self._ce: Optional[CrossEncoder] = None
        self._sparse: Optional[SparseBM25] = None

    def ensure_ce(self):
        if self._ce is None:
            self._ce = CrossEncoder(str(CROSS_ENCODER_NAME), device=DEVICE)

    def ensure_sparse(self):
        if self._sparse is None:
            self._sparse = SparseBM25(self.payload)

    def encode_query(self,  query: Union[str, Sequence[str]]) -> np.ndarray:
        vec = self.embedder.encode(
            qtext(query),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        vec = np.asarray(vec, dtype=np.float32)
        vec = np.atleast_2d(vec)
        return np.ascontiguousarray(vec, dtype=np.float32)

    def search(self,  query: Union[str, Sequence[str]], k: int = 20) -> pd.DataFrame:
        qv = self.encode_query(query)
        qv = np.ascontiguousarray(qv, dtype=np.float32)
        scores, idx = self.index.search(qv, k)
        res = self.payload.iloc[idx[0]].copy()
        res["score"] = scores[0]
        # Add rowid to align with sparse results
        res["rowid"] = idx[0].astype(np.int32)
        return res

    @staticmethod
    def mmr(
            query_vec: np.ndarray,
            cand_vecs: np.ndarray,
            lambda_mult: float = 0.3,
            top_k: int = 20
    ) -> List[int]:
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

    def rerank(
            self,
            query: Union[str, Sequence[str]],
            res: pd.DataFrame,
            method: str = "blend",
            mmr_lambda: float = 0.3
    ) -> pd.DataFrame:
        method = (method or self.cfg.method or "blend").lower()
        year_now = getattr(self.cfg, "year_now", YEAR_NOW)
        out = res.copy()

        if method in ("retrieval", "none"):
            return out.sort_values("score", ascending=False)

        if method in ("ce", "cross", "cross_encoder"):
            self.ensure_ce()
            pairs = [[qtext(query), d] for d in out["doc"].tolist()]
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
            # Re-embed top-200 docs for a diversity step
            topn = min(200, len(out))
            tops = out.head(topn)
            doc_vecs = self.embedder.encode(
                tops["doc"].tolist(), batch_size=128,
                convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True
            ).astype(np.float32)
            qv = self.encode_query(query)
            order_idx = self.mmr(
                qv,
                doc_vecs,
                lambda_mult=mmr_lambda,
                top_k=min(self.cfg.k, topn)
            )
            out = tops.iloc[order_idx]
        return out

    def hybrid(
            self, query: str, k: int = 20, dense_k: int = 500, sparse_k: int = 500,
            rrf_k: int = 60, w_dense: float = 1.0, w_sparse: float = 1.0
    ) -> pd.DataFrame:
        """Hybrid retrieval: dense (FAISS) + sparse (BM25) fused with RRF."""
        self.ensure_sparse()
        dense = self.search(query, k=dense_k).sort_values("score", ascending=False).reset_index(drop=True)
        sparse = self._sparse.search(query, topn=sparse_k).sort_values("bm25", ascending=False).reset_index(drop=True)
        fused = rrf_fuse(dense, sparse, rrf_k=rrf_k, w_dense=w_dense, w_sparse=w_sparse, topn=max(dense_k, sparse_k))
        year_now = getattr(self.cfg, "year_now", YEAR_NOW)
        y = pd.to_numeric(fused.get("year"), errors="coerce").fillna(year_now)
        recency = 1.0 / (1.0 + np.maximum(0, year_now - y))
        pop = (pd.to_numeric(fused.get("vote_count"), errors="coerce").fillna(0)).clip(0, 20000) / 20000.0
        fused["hybrid"] = 0.8 * fused["rrf"].astype(float) + 0.12 * recency + 0.08 * pop
        return fused.sort_values("hybrid", ascending=False).head(k)


@lru_cache(maxsize=2)
def get_retriever(cfg_tuple: Tuple) -> TextRetriever:
    """LRU-cached retriever keyed by a tuple of config values (so we don't reload models repeatedly)."""
    cfg = coerce_cfg(cfg_tuple)
    return TextRetriever(cfg)

def retrieve(
        query: Union[str, Sequence[str]],
        k: int = 20,
        method: str = "blend",
        mmr_lambda: float = 0.3,
        hybrid_dense_k: int = 500,
        hybrid_sparse_k: int = 500,
        rrf_k: int = 60,
        w_dense: float = 1.0,
        w_sparse: float = 1.0,
        lang_policy: str = "auto-soft"
) -> pd.DataFrame:
    cfg = SearchConfig(k=k, method=method, mmr_lambda=mmr_lambda)
    tr = get_retriever(tuple(asdict(cfg).values()))
    # detect language intent from query using data-driven aliases
    lang_code, lang_name = detect_lang_intent(qtext(query), tr.lang_aliases)
    q_aug = query if not lang_name else [qtext(query), f"Language: {lang_name}"]

    if method == "hybrid":
        out = tr.hybrid(
            q_aug,
            k=k,
            dense_k=hybrid_dense_k,
            sparse_k=hybrid_sparse_k,
            rrf_k=rrf_k,
            w_dense=w_dense,
            w_sparse=w_sparse)
    else:
        res = tr.search(q_aug, k=max(k, 200 if method == "mmr" else k))
        out = tr.rerank(q_aug, res, method=method, mmr_lambda=mmr_lambda)
    strict = (lang_policy.lower() == "auto-hard")
    out = apply_lang_policy(out, lang_code, lang_name, k, strict=strict, boost=1.35)
    return out.head(k)

def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", type=str, help="free-text taste or title")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--method", type=str, default="blend", choices=["retrieval","blend","ce","mmr", "hybrid"])
    ap.add_argument("--mmr_lambda", type=float, default=0.3)
    # Hybrid-specific knobs
    ap.add_argument("--hybrid_dense_k", type=int, default=500)
    ap.add_argument("--hybrid_sparse_k", type=int, default=500)
    ap.add_argument("--rrf_k", type=int, default=60)
    ap.add_argument("--w_dense", type=float, default=1.0)
    ap.add_argument("--w_sparse", type=float, default=1.0)
    ap.add_argument("--lang_policy", type=str, default="auto-soft",
                    choices=["off", "auto-soft", "auto-hard"])
    args = ap.parse_args()

    out = retrieve(
        args.query,
        k=args.k,
        method=args.method,
        mmr_lambda=args.mmr_lambda,
        hybrid_dense_k=args.hybrid_dense_k,
        hybrid_sparse_k=args.hybrid_sparse_k,
        rrf_k=args.rrf_k,
        w_dense=args.w_dense,
        w_sparse=args.w_sparse,
    )
    cols = [c for c in ["title", "year", "score", "bm25", "rrf", "hybrid", "blend", "rerank", "poster_url"] if
            c in out.columns]
    print(out[cols].head(args.k).to_string(index=False))

if __name__ == "__main__":
    cli()