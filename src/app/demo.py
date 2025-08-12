import json
import sys
from pathlib import Path

import streamlit as st
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.recsys.search_and_rerank import retrieve, jlist

def s(x) -> str:
    """Safe scalar -> string."""
    try:
        return "" if pd.isna(x) else str(x)
    except Exception:
        return "" if x is None else str(x)

def truncate(text: str, n: int = 120) -> str:
    text = s(text)
    return text[:n] + ("…" if len(text) > n else "")

def first_score(row: pd.Series):
    for c in ("blend", "rerank", "score"):
        if c in row and row[c] is not None:
            try:
                if pd.notna(row[c]):
                    return float(row[c])
            except Exception:
                continue
    return None

def apply_min_votes(df: pd.DataFrame, k: int, min_votes: int) -> pd.DataFrame:
    """Keep items with vote_count>=min_votes (treat NaN as 0), then top-up to ensure k results."""
    if "vote_count" not in df.columns or min_votes <= 0:
        return df.head(k)
    base = df.copy()
    votes = pd.to_numeric(base["vote_count"], errors="coerce").fillna(0)
    keep = base[votes >= int(min_votes)]
    if len(keep) >= k:
        return keep.head(k)
    need = k - len(keep)
    rest = base[~base.index.isin(keep.index)].head(need)
    return pd.concat([keep, rest], ignore_index=True)

# -----UI-------
st.set_page_config(page_title="🎬 Next-Gen Movie Recommender", layout="wide")
st.title("🎬 Next-Gen Movie Recommender")
st.caption("Dense retrieval (embeddings) + light rerank (blend / CE)")

with st.sidebar:
    st.header("Search")
    q = st.text_input("Type a taste / title / query", "indian thriller")
    method = st.selectbox("Method", ["blend", "retrieval", "ce", "mmr"], index=0)
    k = st.slider("Top-K", 5, 40, 20)
    mmr_lambda = 0.3
    if method == "mmr":
        mmr_lambda = st.slider("MMR λ (diversity)", 0.1, 0.9, 0.3,
                               help="Lower = more variety; Higher = tighter to query")
    min_vote = st.number_input("Min vote_count", min_value=0, max_value=10000, value=0, step=1,
                               help="Use 0 to avoid filtering; raise to prefer mainstream picks")
    go = st.button("Search", use_container_width=True)

# ---- Results
if go:
    try:
        res = retrieve(q, k=k, method=method, mmr_lambda=mmr_lambda)
        res = apply_min_votes(res, k=k, min_votes=min_vote)

        if res.empty:
            st.warning("No results. Try lowering Min vote_count, switching method, or broadening the query.")
        else:
            cols = st.columns(5)
            for i, (_, row) in enumerate(res.iterrows()):
                with cols[i % 5]:
                    poster = row.get("poster_url")
                    if isinstance(poster, str) and poster:
                        st.image(poster, use_container_width=True)

                    title = s(row.get("title"))
                    y = row.get("year")
                    year_str = str(int(y)) if pd.notna(y) else "—"
                    st.markdown(f"**{title}**")

                    sc = first_score(row)
                    if sc is not None:
                        st.caption(f"{year_str} • score: {sc:.3f}")
                    else:
                        st.caption(year_str)

                    # Reason line: Cast/Director if available, else overview snippet
                    casts = jlist(row.get("actors")) or jlist(row.get("actors_json"))
                    dirs  = jlist(row.get("directors")) or jlist(row.get("directors_json"))
                    reason_parts = []
                    if casts:
                        reason_parts.append("Cast: " + ", ".join(map(str, casts[:2])))
                    if dirs:
                        reason_parts.append("Dir: " + ", ".join(map(str, dirs[:1])))
                    if not reason_parts:
                        reason_parts.append(truncate(row.get("overview"), 120))
                    st.write(" • ".join([p for p in reason_parts if p]))

                    if i >= k - 1:
                        break
    except Exception as e:
        st.error(f"Error while searching: {e}")
else:
    st.info("Enter a query, pick a method, and click **Search**. For more variety, try `method = mmr` with λ ≈ 0.2–0.3.")