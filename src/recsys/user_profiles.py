from dataclasses import asdict
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.config import SEARCH_PAYLOAD
from src.recsys.search_and_rerank import get_retriever, SearchConfig
from src.utils.nlp_utils import safe_str, jlist

def load_payload() -> pd.DataFrame:
    """Load the search payload DataFrame from the configured path."""
    df = pd.read_parquet(SEARCH_PAYLOAD)
    must = ["movie_id", "title", "doc"]
    for m in must:
        if m not in df.columns:
            raise RuntimeError(f"SEARCH_PAYLOAD missing required column: {m}")
    return df

def resolve_seed_titles(titles: List[str], df: pd.DataFrame, limit_per_title: int = 1) -> pd.DataFrame:
    """Resolve a list of movie titles to their corresponding rows in the DataFrame."""
    pool = []
    titles = [t.strip() for t in titles if t and t.strip()]
    if not titles:
        return pd.DataFrame(columns=df.columns)
    low_titles = df["title"].astype(str)
    for t in titles:
        t_low = t.lower()
        hits = df[low_titles.str.lower().str.contains(t_low, na=False)].head(limit_per_title)
        if hits.empty:
            hits = df[low_titles.str.lower() == t_low].head(1)
        pool.append(hits)
    if not pool:
        return pd.DataFrame(columns=df.columns)
    out = pd.concat(pool, ignore_index=True).drop_duplicates(subset=["movie_id"])
    return out

def pseudo_query_from_rows(rows: pd.DataFrame, char_cap: int = 600) -> str:
    """Build a text that captures the user's taste for BM25 and CE."""
    parts = []
    titles = [ safe_str(t) for t in rows.get("title", []) if safe_str(t) ]
    if titles:
        parts.append("Seeds: " + ", ".join(titles[:5]))
    # gather genres/actors/directors from doc or struct cols
    def _take(names, k):
        return ", ".join([safe_str(x) for x in names if safe_str(x)][:k])
    genres = []
    if "genres" in rows.columns:
        for cell in rows["genres"]:
            g = jlist(cell)
            if g and isinstance(g[0], dict):
                genres.extend([d.get("name") for d in g if isinstance(d, dict) and d.get("name")])
            elif g:
                genres.extend([safe_str(x) for x in g])
    actors = []
    for key in ("actors", "actors_json"):
        if key in rows.columns:
            for cell in rows[key]:
                actors.extend([safe_str(x) for x in jlist(cell)])
            break
    directors = []
    for key in ("directors", "directors_json"):
        if key in rows.columns:
            for cell in rows[key]:
                directors.extend([safe_str(x) for x in jlist(cell)])
            break
    if genres:
        parts.append("Genres: " + _take(dict.fromkeys(genres).keys(), 8))
    if actors:
        parts.append("Cast: " + _take(dict.fromkeys(actors).keys(), 8))
    if directors:
        parts.append("Directors: " + _take(dict.fromkeys(directors).keys(), 5))
    text = ". ".join([p for p in parts if p])
    return text[:char_cap]

def prepare_user_profile_from_titles(seed_titles: List[str]) -> Tuple[np.ndarray, str, pd.DataFrame]:
    """Prepare a user profile vector and pseudo query from a list of seed titles."""
    df = load_payload()
    rows = resolve_seed_titles(seed_titles, df)
    if rows.empty:
        raise ValueError("Could not resolve any seeds from the provided titles.")
    # embed the seed docs and average (normalized → average → renorm)
    cfg = SearchConfig()  # default is fine
    tr = get_retriever(tuple(asdict(cfg).values()))
    doc_texts = rows["doc"].astype(str).tolist()
    mat = tr.embedder.encode(doc_texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    user_vec = np.mean(mat, axis=0, dtype=np.float32)
    # renormalize to unit length
    norm = np.linalg.norm(user_vec) + 1e-8
    user_vec = (user_vec / norm).astype(np.float32).reshape(1, -1)
    pseudo_query = pseudo_query_from_rows(rows)
    return user_vec, pseudo_query, rows