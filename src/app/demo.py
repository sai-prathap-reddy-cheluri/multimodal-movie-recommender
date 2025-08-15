import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Sequence, Union, Optional
import pandas as pd
import streamlit as st
from src.recsys.search_and_rerank import retrieve, retrieve_user_profile
from src.utils.nlp_utils import jlist, safe_str

try:
    from src.recsys.explanations import make_reasons_for_frame
    HAS_EXPL = True
except Exception:
    HAS_EXPL = False

def truncate(text: str, n: int = 120) -> str:
    text = safe_str(text)
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
    seeds_raw = st.text_input("Seed titles (comma → multi)", "Drishyam, Memories")
    mode = st.radio("Query mode", ["Query only", "Seeds only", "Both"], index=0)
    method = st.selectbox("Method", ["hybrid", "blend", "retrieval", "ce", "mmr"], index=0)
    k = st.slider("Top-K", 5, 40, 20)
    mmr_lambda = 0.3
    if method == "mmr":
        mmr_lambda = st.slider("MMR λ (diversity)", 0.1, 0.9, 0.3)
    min_vote = st.number_input("Min vote_count", min_value=0, max_value=10000, value=0, step=1)
    lang_policy_label = st.selectbox(
        "Language handling",
        ["Auto (soft boost)", "Auto (hard keep)", "Off"],
        index=0,
        help="Soft boost keeps diversity; Hard keep filters/top-ups strictly by language"
    )
    # Map to backend flags
    if lang_policy_label.startswith("Auto (soft"):
        lang_policy = "auto-soft"
    elif lang_policy_label.startswith("Auto (hard"):
        lang_policy = "auto-hard"
    else:
        lang_policy = "off"

go = st.button("Search", use_container_width=True)

if go:
    try:
        # allow multi-queries
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        q: Union[str, Sequence[str]] = parts if len(parts) > 1 else (parts[0] if parts else "")
        seed_titles = [t.strip() for t in seeds_raw.split(",") if t.strip()] if seeds_raw else []
        res = None
        pseudo_for_expl: Optional[str] = None

        if mode == "Query only" or not seed_titles:
            # regular retrieve
            try:
                res_obj = retrieve(q, k=k, method=method, mmr_lambda=mmr_lambda, lang_policy=lang_policy)
            except TypeError:
                res_obj = retrieve(q, k=k, method=method, mmr_lambda=mmr_lambda)
            res = coerce_df(res_obj)
            pseudo_for_expl = q if isinstance(q, str) else ", ".join(q)

        elif mode == "Seeds only":
            # build user profile from seeds and use personalized retrieval
            from src.recsys.user_profiles import prepare_user_profile_from_titles  # [STEP6]

            user_vec, pseudo_query, _rows = prepare_user_profile_from_titles(seed_titles)
            res = retrieve_user_profile(
                user_vec,
                pseudo_query=pseudo_query,
                k=k,
                method=method,
                mmr_lambda=mmr_lambda,
            )
            pseudo_for_expl = pseudo_query

        else:  # Both
            # combine: use seeds to steer, include text query in pseudo
            from src.recsys.user_profiles import prepare_user_profile_from_titles  # [STEP6]

            user_vec, pseudo_query, _rows = prepare_user_profile_from_titles(seed_titles)
            combo = pseudo_query
            if isinstance(q, str) and q:
                combo = f"{pseudo_query}. Query: {q}"
            elif isinstance(q, (list, tuple)) and q:
                combo = f"{pseudo_query}. Query: {'; '.join(q)}"
            res = retrieve_user_profile(
                user_vec,
                pseudo_query=combo,
                k=k,
                method=method,
                mmr_lambda=mmr_lambda,
            )
            pseudo_for_expl = combo

        res = apply_min_votes(res, k=k, min_votes=min_vote)

        if not isinstance(res, pd.DataFrame) or res.empty:
            st.warning("No results. Try lowering Min vote_count, switching method, or broadening the query/seeds.")
            st.stop()

        # explanations (use pseudo/text used in retrieval)
        if HAS_EXPL and pseudo_for_expl:
            try:
                reasons_series = make_reasons_for_frame(pseudo_for_expl, res, max_reasons=2)
                res = res.copy()
                res["__reasons"] = reasons_series
            except Exception:
                pass

        rows = res.to_dict(orient="records")
        cols = st.columns(5)
        for i, row in enumerate(rows):
            with cols[i % 5]:
                poster = row.get("poster_url")
                if isinstance(poster, str) and poster:
                    st.image(poster, use_container_width=True)

                title = safe_str(row.get("title"))
                y = row.get("year")
                year_str = str(int(y)) if (isinstance(y, (int, float)) and pd.notna(y)) else (safe_str(y) or "—")
                st.markdown(f"**{title}**")

                sc = first_score(row)
                st.caption(f"{year_str} • score: {sc:.3f}" if sc is not None else year_str)

                rlist = row.get("__reasons")
                if isinstance(rlist, list) and rlist:
                    st.write(" • ".join(rlist))
                else:
                    st.write(truncate(row.get("overview"), 120))

                if i >= k - 1:
                    break

    except Exception as e:
        st.error(f"Error while searching: {e}")
        st.exception(e)
    else:
        st.info(
            "Type a query and/or seed a few titles (e.g., **Drishyam, Memories**), pick a method (try **hybrid**) and hit Search.")