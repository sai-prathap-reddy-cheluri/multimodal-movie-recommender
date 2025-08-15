from __future__ import annotations

import re
from typing import List, Union, Sequence

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

def tokenize(text: str) -> List[str]:
    """Simple, fast tokenizer that works without NLTK"""
    return re.findall(r"[a-z0-9]+", (text or "").lower())

class SparseBM25:
    """
    Lightweight BM25 over the doc field for search payload.
    """

    def __init__(self, df_payload: pd.DataFrame):
        if "doc" not in df_payload.columns:
            raise ValueError("SparseBM25 requires a 'doc' column in payload.")
        self.df = df_payload.reset_index(drop=True)
        self.docs = [tokenize(t) for t in self.df["doc"].astype(str).tolist()]
        self.bm25 = BM25Okapi(self.docs)

    def search(self, query: Union[str, Sequence[str]], topn: int = 500) -> pd.DataFrame:
        """Search for query in the doc field, return topn results with scores."""
        if isinstance(query, (list, tuple)):
            tokens = []
            for q in query:
                tokens.extend(tokenize(str(q)))
        else:
            tokens = tokenize(str(query))

        scores = self.bm25.get_scores(tokens)
        idx = np.argsort(scores)[::-1][:topn]
        out = self.df.iloc[idx].copy()
        out["bm25"] = np.asarray(scores, dtype=np.float32)[idx]
        out["rowid"] = idx.astype(np.int32)
        return out
