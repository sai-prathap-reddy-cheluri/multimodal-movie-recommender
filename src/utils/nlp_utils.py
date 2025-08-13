import json

import pandas as pd


def safe_str(x) -> str:
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