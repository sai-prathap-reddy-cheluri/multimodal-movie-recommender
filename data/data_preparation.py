import hashlib
import json

import pandas as pd
import argparse, json, hashlib, pathlib as p

IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
SAMPLE_ROWS_DEFAULT = 10000

def strip_all_strings(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].str.strip()
            df.loc[df[c] == "", c] = pd.NA
    return df

def to_bool(x):
    if pd.isna(x): return pd.NA
    s =str(x).strip().lower()
    if s in {"1","true","t","yes","y"}: return True
    if s in {"0","false","f","no","n"}: return False
    return pd.NA

def parse_date_iso(x):
    # Accepts 'YYYY-MM-DD' and gracefully NA otherwise
    if pd.isna(x): return pd.NA
    s = str(x).strip()
    if not s: return pd.NA
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return s
    try:
        dt = pd.to_datetime(s, errors="coerce", utc=True)
        if pd.isna(dt): return pd.NA
        return dt.date().isoformat()
    except Exception:
        return pd.NA

def parse_list_cell(x, as_int = False) -> str:
    """
    Returns a JSON string list.
    Accepts: JSON like "[1, 2]" or delimited text "A, B, C" / "A|B" / "A;B" / "A / B".
    """
    if pd.isna(x): return json.dumps([])
    s = str(x).strip()
    # Try JSON list
    if (s.startswith("[") and s.endswith("]") or s.startswith("{") and s.endswith("}")):
        try:
            val = json.loads(s)
            if isinstance(val, list):
                if as_int:
                    out = []
                    for v in val:
                        try: out.append(int(v))
                        except Exception: pass
                    return json.dumps(out, ensure_ascii=False)
                return json.dumps([str(v).strip() for v in val if str(v).strip() != ""],  ensure_ascii=False)
        except Exception:
            pass
    # Fallback delimiters
    for sep in ["|", ",",";"," / "]:
        if sep in s:
            parts = [t.strip() for t in s.split(sep) if t.strip() != ""]
            if as_int:
                cleaned = []
                for t in parts:
                    try: cleaned.append(int(t))
                    except Exception: pass
                return json.dumps(cleaned, ensure_ascii=False)
            return json.dumps(parts, ensure_ascii=False)

    # single token
    try:
        return json.loads([int(s)] if as_int else json.dumps([s]))
    except Exception:
        return json.dumps([s])

def profile(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 ** 2, 2),
        "nulls": {c: int(df[c].isna().sum()) for c in df.columns},
        "uniques": {c: int(df[c].nunique(dropna=True)) for c in df.columns},
        "dtypes": {c: str(t) for c,t in df.dtypes.items()},
        "top_langs": df["original_language"].value_counts(dropna=True).head(10).to_dict()
            if "original_language" in df.columns else {},
        "period": {
            "min_release_date": str(pd.to_datetime(df["release_date"], errors="coerce").min().date())
                if "release_date" in df.columns else None,
            "max_release_date": str(pd.to_datetime(df["release_date"], errors="coerce").max().date())
                if "release_date" in df.columns else None,
        }
    }

def sha256(path: p.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def abs_url(path_val: str | None) -> str | None:
    if path_val is None or (isinstance(path_val, float) and pd.isna(path_val)):
        return None
    s = str(path_val).strip()
    if not s: return None
    if s.startswith("http://") or s.startswith("https://"): return s
    if s.startswith("/"): return f"{IMAGE_BASE}{s}"
    return f"{IMAGE_BASE}{s}"

def _ensure_json_string_series(s: pd.Series, as_int=False) -> pd.Series:
    """Coerce any value (list/str/NA/other) to a JSON string list."""
    def _conv(v):
        if v is None or (isinstance(v, float) and pd.isna(v)) or v is pd.NA:
            return json.dumps([])
        if isinstance(v, list):
            if as_int:
                out = []
                for x in v:
                    try:
                        out.append(int(x))
                    except Exception:
                        # fall back to skipping non-ints
                        pass
                return json.dumps(out, ensure_ascii=False)
            # list of strings
            return json.dumps(
                [str(x).strip() for x in v if str(x).strip() != ""],
                ensure_ascii=False
            )
        if isinstance(v, str):
            s = v.strip()
            # already JSON?
            if s.startswith("[") and s.endswith("]"):
                try:
                    json.loads(s)
                    return s  # keep as-is
                except Exception:
                    pass
            # not JSON → try delimiters / single token
            if as_int:
                # split on common delimiters
                for sep in ["|", ",", ";", " / "]:
                    if sep in s:
                        out = []
                        for t in [t.strip() for t in s.split(sep) if t.strip() != ""]:
                            try:
                                out.append(int(t))
                            except Exception:
                                pass
                        return json.dumps(out, ensure_ascii=False)
                # single token
                try:
                    return json.dumps([int(s)], ensure_ascii=False)
                except Exception:
                    return json.dumps([], ensure_ascii=False)
            else:
                for sep in ["|", ",", ";", " / "]:
                    if sep in s:
                        return json.dumps(
                            [t.strip() for t in s.split(sep) if t.strip() != ""],
                            ensure_ascii=False
                        )
                return json.dumps([s], ensure_ascii=False) if s else json.dumps([])
        # numbers or other objects → wrap as a single element
        try:
            return json.dumps([int(v)], ensure_ascii=False) if as_int else json.dumps([str(v)], ensure_ascii=False)
        except Exception:
            return json.dumps([str(v)], ensure_ascii=False)

    return s.map(_conv).astype("string")

def run(src_csv: str, out_dir: str, sample_rows: int):
    out = p.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    reports = p.Path("reports")
    reports.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        src_csv,
        dtype = str,
        keep_default_na = False,
        encoding = "utf-8",
        on_bad_lines = "skip",
        engine = "python"
    )

    # Normalize headers and strings
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df = strip_all_strings(df)

    # Expect id column and rename to 'movie_id'
    if "id" not in df.columns:
        raise SystemExit("Expected column 'id' not found. Please make sure your CSV has it.")
    df = df.rename(columns = {"id":"movie_id"})
    df["movie_id"] = df["movie_id"].astype("string")

    # Cast simple text columns if present
    for c in ["title","original_title","original_language","overview","poster_path","backdrop_path"]:
        if c in df.columns:
            df[c] = df[c].astype("string")

    # Booleans
    for c in ["adult", "video"]:
        if c in df.columns:
            df[c] = df[c].map(to_bool)

    # Numerics
    for c in ["popularity", "vote_average", "runtime"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    if "vote_count" in df.columns:
        df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce").astype("Int64")

    # Dates & year
    if "release_date" in df.columns:
        df["release_date"] = df["release_date"].map(parse_date_iso)
        year = pd.to_datetime(df["release_date"], errors="coerce").dt.year
        df["year"] = year.astype("Int64")

    # Lists -> JSON strings
    if "genre_ids" in df.columns:
        df["genre_ids_json"] = df["genre_ids"].map(lambda x: parse_list_cell(x, as_int=True))
    if "actors" in df.columns:
        df["actors_json"] = df["actors"].map(parse_list_cell)
    if "directors" in df.columns:
        df["directors_json"] = df["directors"].map(parse_list_cell)

    # Quality-of-life: absolute image URLs
    if "poster_path" in df.columns:
        df["poster_url"] = df["poster_path"].map(abs_url)
    if "backdrop_path" in df.columns:
        df["backdrop_url"] = df["backdrop_path"].map(abs_url)
    df["has_poster"] = df.get("poster_url", pd.Series([None]*len(df))).notna()

    # Basic hygiene
    before = len(df)
    df = df.drop_duplicates(subset=["movie_id"])
    for c in ["popularity", "vote_average", "vote_count", "runtime"]:
        if c in df.columns:
            df.loc[df[c].notna() & (df[c] < 0), c] = pd.NA
    dropped = before - len(df)

    # Stable sort
    sort_cols = [c for c in ["release_date", "movie_id"] if c in df.columns]
    if sort_cols: df = df.sort_values(sort_cols, na_position="last")

    for col, as_int in [("genre_ids_json", True), ("actors_json", False), ("directors_json", False)]:
        if col in df.columns:
            df[col] = _ensure_json_string_series(df[col], as_int=as_int)

    # Write outputs
    flat_path = out / "movies.parquet"
    df.to_parquet(flat_path, index=False)  # requires pyarrow
    if "year" in df.columns:
        df.to_parquet(out / "movies_parquet", index=False, partition_cols=["year"])

    sample_path = out / "movies_sample.parquet"
    df.sample(min(sample_rows, len(df)), random_state=42).to_parquet(sample_path, index=False)

    # Reports
    (reports / "data_profile.json").write_text(json.dumps(profile(df), indent=2))
    (reports / "checksums.txt").write_text(
        f"movies.parquet\t{sha256(flat_path)}\nmovies_sample.parquet\t{sha256(sample_path)}\n"
    )

    print(f"✅ Wrote {flat_path}  rows={len(df)}  dropped_dupes={dropped}")
    if "year" in df.columns: print("📁 Wrote partitioned Parquet -> data/processed/movies_parquet/")
    print("📄 reports/data_profile.json  🔐 reports/checksums.txt")


def cli():
    ap = argparse.ArgumentParser(description="Prepare DS-ready Parquet release for the movies dataset.")
    ap.add_argument("csv", help="Path to raw CSV (e.g., data/movies_2020-01-01_2025-08-08.csv)")
    ap.add_argument("--out", default="data/processed", help="Output directory (default: data/processed)")
    ap.add_argument("--sample", type=int, default=SAMPLE_ROWS_DEFAULT, help="Sample size (default: 10000)")
    args = ap.parse_args()
    run(args.csv, args.out, args.sample)


if __name__ == "__main__":
    cli()