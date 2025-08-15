import asyncio
import os.path
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict

import aiohttp
import pandas as pd
import requests
from src.config import TMDB_API_KEY, DATA_DIR

def parse_date(s):
    """Parse a date string or datetime object into a datetime object."""
    if isinstance(s, datetime):
        return s
    elif isinstance(s, str):
        return datetime.strptime(s, "%Y-%m-%d")
    raise ValueError(f"Unrecognized date: {s}")

def daterange(start_date, end_date, step_years=1):
    """Generates date windows from start_date to end_date with a step in years."""
    s = parse_date(start_date)
    e = parse_date(end_date)
    while s < e:
        window_start = s
        window_end = min(window_start.replace(year=window_start.year + step_years), e)
        yield window_start.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d")
        s = window_end

def fetch_window(from_date, to_date, include_adult, region: Optional[str],
                 max_pages = 500) -> Tuple[List[dict], int]:
    """Fetches up to 10,000 movies for this window, returns (movies, total_pages)"""
    all_movies = []
    url = "https://api.themoviedb.org/3/discover/movie"
    for page in range(1, max_pages + 1):
        params = {
            "api_key": TMDB_API_KEY,
            "language": "en-US",  # response language
            "sort_by": "popularity.desc",
            "include_adult": "true" if include_adult else "false",
            "include_video": "false",
            "primary_release_date.gte": from_date,
            "primary_release_date.lte": to_date,
            "page": page,
        }
        if region:
            params["region"] = region
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Error fetching page {page}: {response.status_code}")
            break
        data = response.json()
        if "results" in data and data["results"]:
            all_movies.extend(data["results"])
        total_pages = data.get("total_pages", 1)
        if page == 1:
            print(f"Pages in this window: {total_pages}")
        if page >= total_pages:
            break
    return all_movies, total_pages

def recursive_fetch(from_date, to_date, include_adult = False, region=None, max_pages = 500,
                    day_threshold=1) -> List[dict]:
    """Recursively splits the date windows to avoid API 10k limit."""
    print(f"Fetching window: {from_date} to {to_date}")
    movies, total_pages = fetch_window(from_date, to_date, include_adult,region, max_pages)
    if total_pages < max_pages or (parse_date(to_date) - parse_date(from_date)).days <= day_threshold:
        return movies
    else:
        # Window too dense, split in half and recurse
        start = parse_date(from_date)
        end = parse_date(to_date)
        mid = start + (end - start) / 2
        first_half = recursive_fetch(
            from_date,
            mid.strftime("%Y-%m-%d"),
            include_adult,
            region,
            max_pages,
            day_threshold
        )
        second_half = recursive_fetch(
            (mid + timedelta(days=1)).strftime("%Y-%m-%d"),
            to_date,
            include_adult,
            region,
            max_pages,
            day_threshold
        )
        return first_half + second_half

async def fetch_movie_details(session: aiohttp.ClientSession, movie_id: int) -> Dict[str, object]:
    """
    Asynchronously fetch runtime, full cast, and directors for a movie by ID.
    Returns a dict with 'runtime', 'actors', 'directors'.
    """
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {"api_key": TMDB_API_KEY, "append_to_response": "credits"}
    try:
        async with session.get(url, params=params) as response:
            if response.status != 200:
                return {"runtime": None, "actors": [], "directors": []}
            data = await response.json()
            runtime = data.get("runtime")
            credits = data.get("credits", {})
            cast = credits.get("cast") or []
            actors = [c.get("name") for c in sorted(cast, key=lambda x: x.get("order", 999))]
            crew = credits.get("crew") or []
            directors = [member["name"] for member in crew if member.get("job") == "Director"]
            return {"runtime": runtime, "actors": actors, "directors": directors}
    except Exception as e:
        print(f"Error fetching {movie_id}: {e}")
        return {"runtime": None, "actors": [], "directors": []}

async def fetch_details_batch(movie_ids: List[int], concurrency: int = 20) -> Dict[int, Dict[str, object]]:
    """
    Launch asynchronous tasks to fetch details for the movies based on movie_id
    """
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_movie_details(session, mid) for mid in movie_ids]
        results = await asyncio.gather(*tasks)
    return {mid: res for mid, res in zip(movie_ids, results)}


def download_movie_data(from_date: str,
    to_date: str,
    filename: str,
    include_adult: bool = False,
    region: Optional[str]=None,
    max_pages: int = 500,
    day_threshold: int = 1,
    detail_concurrency: int = 20,
) -> Optional[str]:
    """
    Fetch movies for a date range, then asynchronously fetch full cast and directors.
    Returns the file path if data is written, otherwise None.
    """
    print(f"\n=== Begin download: {from_date} to {to_date} ===")
    movies = recursive_fetch(
        from_date, to_date, include_adult, region, max_pages, day_threshold
    )
    if not movies:
        print("No movies found.")
        return None

    df = pd.DataFrame(movies).drop_duplicates(subset="id")
    # Gather movie IDs
    ids = df["id"].dropna().astype(int).tolist()
    print(f"Fetching details for {len(ids)} movies...")

    # Asynchronously fetch credits (actors & directors) and runtime
    detail_map = asyncio.run(
        fetch_details_batch(ids, concurrency=detail_concurrency)
    )

    # Assign new columns
    df["runtime"] = df["id"].apply(lambda mid: detail_map.get(mid, {}).get("runtime"))
    df["actors"] = df["id"].apply(lambda mid: ", ".join(detail_map.get(mid, {}).get("actors", [])))
    df["directors"] = df["id"].apply(lambda mid: ", ".join(detail_map.get(mid, {}).get("directors", [])))

    # Save to CSV
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, filename)
    df.to_csv(file_path, index=False)
    print(f"Wrote {len(df)} movies (plus actors, directors, runtime) to {file_path}")
    return file_path