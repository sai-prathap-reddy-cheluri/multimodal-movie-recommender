import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Sequence, Optional, Union

import pandas as pd
import pycountry
from torch.fx.experimental.symbolic_shapes import lru_cache

from src.config import SEARCH_PAYLOAD
from src.utils.nlp_utils import jlist, safe_str

def norm_tokens(text: str) -> List[str]:
    """Normalize text to a list of lowercase alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", (text or "").lower())

DOC_PATTERNS = {
     "genres": re.compile(r"\bGenres:\s*([^.]+)"),
    "cast": re.compile(r"\bCast:\s*([^.]+)"),
    "directors": re.compile(r"\bDirectors?:\s*([^.]+)"),
    "language": re.compile(r"\bLanguage:\s*([A-Za-z]+)"),
    "countries": re.compile(r"\bCountries:\s*([^.]+)")
}

def row_facets(row: pd.Series) -> dict:
    """Extract facets from a DataFrame row representing a movie."""
    out = {
        "title": safe_str(row.get("title")),
        "year": None,
        "genres": [],
        "actors": [],
        "directors": [],
        "language": "",
        "countries": [],
        "doc": safe_str(row.get("doc")),
        "original_language": safe_str(row.get("original_language")).lower()
    }
    # Year
    y = row.get("year")
    try:
        if pd.notna(y):
            out["year"] = y
    except Exception:
        pass

    # Genres
    for key in ("genres", "genres_ids_json"):
        if key in row:
            vals = jlist(row.get(key))
            if vals and isinstance(vals[0], dict):
                out["genres"] = [g.get("name") for g in vals if isinstance(g, dict) and g.get("name")]
                break
            elif vals and isinstance(vals[0], (str, int)):
                out["genres"] = [str(v) for v in vals]
                break

    # Actors
    for key in ("actors", "actors_json"):
        if key in row:
            out["actors"] = [str(v) for v in jlist(row.get(key))]
            if out["actors"]:
                break

    # Directors
    for key in ("directors", "directors_json"):
        if key in row:
            out["directors"] = [str(v) for v in jlist(row.get(key))]
            if out["directors"]:
                break

    # Language
    if not out["language"]:
        m = DOC_PATTERNS["language"].search(out["doc"])
        if m:
            out["language"] = m.group(1)

    # Countries
    for key in ("production_countries", "production_countries_json"):
        if key in row:
            vals = jlist(row.get(key))
            if vals and isinstance(vals[0], dict):
                out["countries"] = [c.get("name") for c in vals if isinstance(c, dict) and c.get("name")]
                break
            elif vals and isinstance(vals[0], str):
                out["countries"] = [v for v in vals if v]
                break
    if not out["countries"]:
        m = DOC_PATTERNS["countries"].search(out["doc"])
        if m:
            out["countries"] = [c.strip() for c in m.group(1).split(",") if c.strip()]

    # Parse genres, actors and directors from doc if still empty
    if not out["genres"]:
        m = DOC_PATTERNS["genres"].search(out["doc"])
        if m:
            out["genres"] = [g.strip() for g in m.group(1).split(",") if g.strip()]
    if not out["actors"]:
        m = DOC_PATTERNS["cast"].search(out["doc"])
        if m:
            out["actors"] = [a.strip() for a in m.group(1).split(",") if a.strip()]
    if not out["directors"]:
        m = DOC_PATTERNS["directors"].search(out["doc"])
        if m:
            out["directors"] = [d.strip() for d in m.group(1).split(",") if d.strip()]
    return out


@dataclass
class ReasonContext:
    genre_vocab: set
    lang_aliases: Dict[str, Tuple[str, str]] # alias_lower -> (iso_code, english_name)

@lru_cache(maxsize=1)
def load_payload() -> pd.DataFrame:
    """Load the search payload DataFrame with relevant columns."""
    df = pd.read_parquet(SEARCH_PAYLOAD)
    keep = [c for c in df.columns if c in {
        "title","year","overview","doc","original_language","genres","genre_ids_json",
        "actors","actors_json","directors","directors_json","production_countries","production_countries_json"
    }]
    if "doc" not in keep:
        keep.append("doc")
    return df[keep].copy()

def build_reason_context() -> ReasonContext:
    """Build a context for generating reasons based on the search payload."""
    df = load_payload()
    # Genre vocab
    vocab = set()
    if "genres" in df.columns:
        df["__g"] = df["genres"].apply(jlist())
        for lst in df["__g"]:
            for g in lst:
                vocab.add(str(g).lower())
    aliases: Dict[str, Tuple[str, str]] = {}
    if "original_language" in df.columns:
        codes = df["original_language"].dropna().astype(str).str.lower().unique()
        for code in codes:
            name = code
            try:
                rec = pycountry.languages.get(aplha_2=code)
                if rec and getattr(rec, "name", None):
                    name = rec.name
            except Exception:
                pass
            aliases[code] = (code, name)
            aliases[name.lower()] = (code, name)
    return ReasonContext(genre_vocab=vocab, lang_aliases=aliases)

def detect_query_facets(query_text: str, ctx: ReasonContext) -> dict:
    """Detect facets from the query text, such as genres, language, year, etc."""
    tokens = norm_tokens(query_text)
    token_set = set(tokens)
    # Genres
    q_genres = {g for g in ctx.genre_vocab if g in token_set}
    # Language
    lc = ln = None
    for alias, (code, name) in ctx.lang_aliases.items():
        if alias and alias in query_text.lower():
            lc, ln = code, name
            break
    # Year
    years = [int(t) for t in tokens if len(t) == 4 and t.isdigit() and 1900 <= int(t) <= 2100]
    decade = None
    m = re.search(r"(\d{3})0s", query_text.lower())
    if m:
        decade = int(m.group(1)) * 10
    after = None
    m2 = re.search(r"(after|since)\s+(\d{4})", query_text.lower())
    if m2:
        try:
            after = int(m2.group(2))
        except:
            pass
    before = None
    m3 = re.search(r"(before|pre)\s+(\d{4})", query_text.lower())
    if m3:
        try:
            before = int(m3.group(2))
        except:
            pass

    return {"genres": q_genres, "lang_code": lc, "lang_name": ln, "years": years, "decade": decade, "after": after,
            "before": before}

# Reasons
def fmt_join(xs: Sequence[str], k: int) -> str:
    """Format a list of strings into a comma-separated string, limiting to k unique items."""
    xs = [x for x in xs if x]
    if not xs: return ""
    xs = list(dict.fromkeys(xs))
    return ", ".join(xs[:k])

def title_keyword_reason(query_text: str, title: str) -> Optional[str]:
    """Check if the title contains a keyword from the query text."""
    qtoks = set(norm_tokens(query_text)) - {"movie","film","the","a","an","of","in"}
    ttoks = set(norm_tokens(title))
    hit = [t for t in qtoks if t in ttoks and len(t) >= 4]
    if hit:
        return f"Title contains: {hit[0].title()}"
    return None

def genre_reason(q_genres: set, genres: List[str]) -> Optional[str]:
    """Check if the query genres overlap with the item's genres."""
    if not q_genres or not genres: return None
    g = [g for g in genres if g and g.lower() in q_genres]
    if g:
        return f"Shares genres: {fmt_join(g, 2)}"
    return None

def lang_reason(lang_intent_name: Optional[str], row_lang_name: str, row_lang_code: str) -> Optional[str]:
    """Check if the query language matches the item's language."""
    if not lang_intent_name: return None
    if row_lang_name and lang_intent_name.lower() == row_lang_name.lower():
        return f"Matches language: {row_lang_name}"
    if row_lang_code and lang_intent_name and lang_intent_name[:2].lower() == row_lang_code.lower():
        return f"Matches language code: {row_lang_code.upper()}"
    return None

def year_reason(facets: dict, year: Optional[int]) -> Optional[str]:
    """Check if the item's year aligns with the query facets."""
    if not year: return None
    if facets.get("years"):
        # if user named a year exactly and we're close
        yq = facets["years"][0]
        if abs(year - yq) <= 2:
            return f"Close to year {yq}"
    if facets.get("decade"):
        d = facets["decade"]
        if d <= year < d+10:
            return f"In the {d}s"
    if facets.get("after") and year >= facets["after"]:
        return f"After {facets['after']}"
    if facets.get("before") and year <= facets["before"]:
        return f"Before {facets['before']}"
    return None

def make_reasons_for_row(query: Union[str, Sequence[str]], row: pd.Series, ctx: Optional[ReasonContext] = None,
                         max_reasons: int = 2) -> List[str]:
    """Generate reasons for why a movie row matches the query."""
    text = " ".join(query) if isinstance(query, (list, tuple)) else str(query)
    ctx = ctx or build_reason_context()
    facets = detect_query_facets(text, ctx)
    rf = row_facets(row)

    reasons: List[str] = []

    # 1) Language match (highest priority if the query asked for it)
    lr = lang_reason(facets.get("lang_name"), rf.get("language"), rf.get("original_language"))
    if lr: reasons.append(lr)

    # 2) Genre overlap (query genres vs item genres)
    gr = genre_reason(facets.get("genres"), rf.get("genres", []))
    if gr: reasons.append(gr)

    # 3) Title keyword hit
    tr = title_keyword_reason(text, rf.get("title"))
    if tr: reasons.append(tr)

    # 4) Year alignment if query asked for it
    yr = year_reason(facets, rf.get("year"))
    if yr: reasons.append(yr)

    # 5) Fallbacks: highlight salient people
    if len(reasons) < max_reasons and rf.get("actors"):
        reasons.append("Cast: " + fmt_join(rf["actors"], 2))
    if len(reasons) < max_reasons and rf.get("directors"):
        reasons.append("Dir: " + fmt_join(rf["directors"], 1))

    # final trim / de-dup
    dedup = []
    seen = set()
    for r in reasons:
        if r and r not in seen:
            dedup.append(r)
            seen.add(r)
        if len(dedup) >= max_reasons:
            break
    return dedup

def make_reasons_for_frame(query: Union[str, Sequence[str]], df: pd.DataFrame, max_reasons: int = 2) -> pd.Series:
    """Generate reasons for each row in a DataFrame based on the query."""
    ctx = build_reason_context()
    if df is None or df.empty:
        return pd.Series([], dtype=object)
    out = []
    for _, row in df.iterrows():
        out.append(make_reasons_for_row(query, row, ctx=ctx, max_reasons=max_reasons))
    return pd.Series(out, index=df.index, dtype=object)