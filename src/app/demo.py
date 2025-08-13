import sys, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json, ast
from typing import List, Sequence, Union
import pandas as pd
import streamlit as st
from src.recsys.search_and_rerank import retrieve

# Helpers
def s(x) -> str:
    try:
        return "" if pd.isna(x) else str(x)
    except Exception:
        return "" if x is None else str(x)

def jlist(x) -> List[str]:
    try:
        if x is None or (hasattr(pd, "isna") and pd.isna(x)):
            return []
    except Exception:
        pass
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return []
        try:
            y = json.loads(x)
            return y if isinstance(y, list) else []
        except Exception:
            try:
                y = ast.literal_eval(x)
                return y if isinstance(y, list) else []
            except Exception:
                return []
    return []

def truncate(text: str, n: int = 120) -> str:
    text = s(text)
    return text[:n] + ("…" if len(text) > n else "")

def first_score(row) -> Union[float, None]:
    # Works for dict or Series
    for c in ("hybrid", "blend", "rerank", "score"):
        try:
            v = row.get(c) if isinstance(row, dict) else (row[c] if c in row else None)
        except Exception:
            v = None
        if v is not None:
            try:
                if pd.notna(v):
                    return float(v)
            except Exception:
                continue
    return None

def apply_min_votes(df: pd.DataFrame, k: int, min_votes: int) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
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

def coerce_df(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, tuple) and obj and isinstance(obj[0], pd.DataFrame):
        return obj[0]
    if isinstance(obj, list):
        try:
            return pd.DataFrame(obj)
        except Exception:
            pass
    raise TypeError(f"Expected DataFrame from retrieve(), got {type(obj)}")

# UI
st.set_page_config(page_title="🎬 Next-Gen Movie Recommender", layout="wide")
st.title("🎬 Baseline ‘Next-Gen’ Recommender")
st.caption("Hybrid dense+sparse retrieval + light rerank (+optional MMR)")

with st.sidebar:
    st.header("Search")
    raw = st.text_input("Taste / title / query (comma → multi)", "indian thriller")
    method = st.selectbox("Method", ["hybrid", "blend", "retrieval", "ce", "mmr"], index=0)
    k = st.slider("Top-K", 5, 40, 20)
    mmr_lambda = 0.3
    if method == "mmr":
        mmr_lambda = st.slider("MMR λ (diversity)", 0.1, 0.9, 0.3)
    min_vote = st.number_input("Min vote_count", min_value=0, max_value=10000, value=0, step=1)
    go = st.button("Search", use_container_width=True)

if go:
    try:
        # allow multi-queries
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        q: Union[str, Sequence[str]] = parts if len(parts) > 1 else (parts[0] if parts else "")

        res = retrieve(q, k=k, method=method, mmr_lambda=mmr_lambda)
        res = coerce_df(res)
        res = apply_min_votes(res, k=k, min_votes=min_vote)

        if not isinstance(res, pd.DataFrame) or res.empty:
            st.warning("No results. Try lowering Min vote_count, switching method, or broadening the query.")
            st.stop()

        rows = res.to_dict(orient="records")
        cols = st.columns(5)

        for i, row in enumerate(rows):
            with cols[i % 5]:
                poster = row.get("poster_url")
                if isinstance(poster, str) and poster:
                    st.image(poster, use_container_width=True)

                title = s(row.get("title"))
                y = row.get("year")
                year_str = str(int(y)) if (isinstance(y, (int, float)) and pd.notna(y)) else s(y) or "—"
                st.markdown(f"**{title}**")

                sc = first_score(row)
                st.caption(f"{year_str} • score: {sc:.3f}" if sc is not None else year_str)

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
        st.exception(e)
        try:
            st.write("Debug peek:", type(res).__name__)
            if isinstance(res, pd.DataFrame):
                st.write(res.head(3).to_dict(orient="records"))
        except Exception:
            pass
else:
    st.info("Enter a query, choose a method, and click **Search**. Try `Method = hybrid` for best exact-name/franchise hits.")
