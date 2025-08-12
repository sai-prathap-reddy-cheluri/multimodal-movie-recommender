from pathlib import Path
import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from src.config import (
    MOVIES_PARQUET, SEARCH_PAYLOAD, TEXT_INDEX, TEXT_IDMAP,
    EMBED_MODEL_NAME, DEVICE, USE_FP16
)

def s(x) -> str:
    """Safe scalar → string: handles pd.NA / NaN / None."""
    try:
        return "" if pd.isna(x) else str(x)
    except Exception:
        return "" if x is None else str(x)

def jlist(x):
    """Robust JSON->list or pass-through list."""
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

def to_doc(row):
    # Accept both pre-parsed and *_json columns
    genres = jlist(row.get("genres")) or [str(g) for g in jlist(row.get("genre_ids_json", "[]"))]
    actors = jlist(row.get("actors")) or jlist(row.get("actors_json", "[]"))
    directors = jlist(row.get("directors")) or jlist(row.get("directors_json", "[]"))

    title = s(row.get("title"))
    overview = s(row.get("overview"))

    year = row.get("year")
    year_str = f"Year: {int(year)}" if pd.notna(year) else ""

    parts = [
        title,
        overview,
        f"Genres: {', '.join(genres)}" if genres else "",
        f"Cast: {', '.join(actors[:5])}" if actors else "",
        f"Directors: {', '.join(directors[:3])}" if directors else "",
        year_str
    ]
    # Filter empties and join with ". "
    return ". ".join([p for p in parts if p]).strip()

def main():
    assert isinstance(MOVIES_PARQUET, Path)
    assert MOVIES_PARQUET.exists(), f"Missing {MOVIES_PARQUET}"

    df = pd.read_parquet(MOVIES_PARQUET)
    if "doc" not in df.columns:
        df["doc"] = df.apply(to_doc, axis=1)

    # Minimal payload saved for search UI / rerank stage
    keep_cols = ["movie_id", "title", "year", "poster_url", "overview",
                 "actors", "actors_json", "directors", "directors_json",
                 "genres", "genre_ids_json", "vote_count", "popularity", "original_language", "doc"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    payload = df[keep_cols].copy()

    # Embedding model
    model = SentenceTransformer(str(EMBED_MODEL_NAME), device=DEVICE)
    # Encode with normalization for cosine via inner product
    emb = model.encode(
        payload["doc"].tolist(),
        batch_size=256,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Build FAISS index (cosine via IP on normalized vectors)
    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(emb.astype(np.float32))
    faiss.write_index(index, str(TEXT_INDEX))

    # ID map and payload cache
    idmap = pd.DataFrame({"rowid": np.arange(len(payload), dtype=np.int32),
                          "movie_id": payload["movie_id"].values if "movie_id" in payload.columns else np.arange(len(payload))})
    idmap.to_parquet(TEXT_IDMAP, index=False)
    payload.to_parquet(SEARCH_PAYLOAD, index=False)

    print(f"Built index for {index.ntotal} movies.")
    print(f"Saved: {TEXT_INDEX}, {TEXT_IDMAP}, {SEARCH_PAYLOAD}")

if __name__ == "__main__":
    main()