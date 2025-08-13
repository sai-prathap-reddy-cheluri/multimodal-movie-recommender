from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd


def rank_from_order(df: pd.DataFrame, id_col: str = "rowid" ) -> Dict[int, int]:
    """
    Map id -> 1-based rank using the current row order.
    Assumes df is already sorted by its own relevance for that source.
    """
    idx = df[id_col].tolist()
    return {int(i): r for r, i in enumerate(idx, start=1)}

def rrf_fuse(
        dense_df: pd.DataFrame,
        sparse_df: pd.DataFrame,
        rrf_k: int = 60,
        w_dense: float = 1.0,
        w_sparse: float = 1.0,
        topn: int = 500
) -> pd.DataFrame:
    """
    Reciprocal Rank Fusion of two ranked lists using 'rowid' as the join key.
    Returns a DataFrame ordered by fused 'rrf' score (desc), head(topn).
    """
    if "rowid" not in dense_df.columns or "rowid" not in sparse_df.columns:
        raise ValueError("Both inputs must contain a 'rowid' column.")

    rd = rank_from_order(dense_df)
    rs = rank_from_order(sparse_df)
    keys = set(rd) | set(rs)

    # compute fused scores
    fused: list[Tuple[int, float]] = []
    for rid in keys:
        s =0.0
        if rid in rd:
            s += w_dense * (1.0 / (rrf_k + rd[rid]))
        if rid in rs:
            s += w_sparse * (1.0 / (rrf_k + rs[rid]))
        fused.append((rid, s))

    fused = sorted(fused, key=lambda x: x[1], reverse=True)
    order = [rid for rid, _ in fused][:topn]
    scores = {rid: sc for rid, sc in fused}

    # union payload rows
    joined = pd.concat([dense_df, sparse_df], ignore_index=True).drop_duplicates(subset=["rowid"])
    out = joined.set_index("rowid").loc[order].reset_index()
    out["rrf"] = out["rowid"].map(scores).astype(float)
    return out