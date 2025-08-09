import os
import math
import argparse
import asyncio
import random
from typing import Dict, List

import pandas as pd
import aiohttp

try:
    from src.config import TMDB_API_KEY
except ImportError:
    TMDB_API_KEY = os.getenv("TMDB_API_KEY")


def _is_empty(s: pd.Series) -> pd.Series:
    return s.isna() | (s.astype(str).str.strip() == "")


def _mask_both_empty(df: pd.DataFrame) -> pd.Series:
    actors_empty = _is_empty(df["actors"])
    directors_empty = _is_empty(df["directors"])
    return actors_empty & directors_empty


async def _fetch_movie_detail(session: aiohttp.ClientSession, movie_id: int) -> Dict:
    """Fetch movie details + credits in one request."""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {"api_key": TMDB_API_KEY, "append_to_response": "credits"}
    async with session.get(url, params=params, timeout=30) as resp:
        if resp.status == 404:
            return {"runtime": None, "actors": [], "directors": []}
        data = await resp.json()
        runtime = data.get("runtime")
        credits = data.get("credits", {}) or {}
        cast = credits.get("cast", [])
        actors = [c.get("name") for c in sorted(cast, key=lambda x: x.get("order", 999))]
        crew = credits.get("crew", [])
        directors = [c.get("name") for c in crew if c.get("job") == "Director"]
        return {"runtime": runtime, "actors": actors, "directors": directors}


async def fetch_with_retry(session, movie_id, retries=8, base_backoff=1.0) -> Dict:
    """Retry wrapper with exponential backoff + jitter."""
    for attempt in range(retries):
        try:
            return await _fetch_movie_detail(session, movie_id)
        except aiohttp.ClientResponseError as e:
            if e.status == 429:
                wait_time = float(e.headers.get("Retry-After", "2"))
                await asyncio.sleep(wait_time)
            else:
                await asyncio.sleep(base_backoff * (2 ** attempt) + random.uniform(0, 0.5))
        except (aiohttp.ClientError, asyncio.TimeoutError):
            await asyncio.sleep(base_backoff * (2 ** attempt) + random.uniform(0, 0.5))
    return {"runtime": None, "actors": [], "directors": []}


async def process_batch(session, ids):
    results = {}
    tasks = [fetch_with_retry(session, mid) for mid in ids]
    details = await asyncio.gather(*tasks)
    for mid, detail in zip(ids, details):
        results[mid] = detail
    return results


def save_progress(df, csv_path):
    df.to_csv(csv_path, index=False)
    print(f"Progress saved to {csv_path}")


async def backfill(csv_path, concurrency=10, batch_size=500):
    df = pd.read_csv(csv_path)

    if "actors" not in df.columns or "directors" not in df.columns:
        raise ValueError("CSV must already have 'actors' and 'directors' columns")

    mask = _mask_both_empty(df)
    todo_ids = df.loc[mask, "id"].dropna().astype(int).tolist()
    print(f"Total rows needing backfill: {len(todo_ids)}")

    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        total_batches = math.ceil(len(todo_ids) / batch_size)
        for i in range(total_batches):
            batch_ids = todo_ids[i * batch_size:(i + 1) * batch_size]
            print(f"Processing batch {i + 1}/{total_batches} ({len(batch_ids)} movies)")

            results = await process_batch(session, batch_ids)

            for mid in batch_ids:
                detail = results.get(mid, {})
                if detail:
                    df.loc[df["id"] == mid, "runtime"] = detail.get("runtime")
                    df.loc[df["id"] == mid, "actors"] = ", ".join(detail.get("actors", []))
                    df.loc[df["id"] == mid, "directors"] = ", ".join(detail.get("directors", []))

            save_progress(df, csv_path)

    print("Backfill complete!!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill missing actors/directors in CSV")
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrent requests")
    parser.add_argument("--batch-size", type=int, default=500, help="Movies per batch")
    args = parser.parse_args()

    if not TMDB_API_KEY:
        raise RuntimeError("TMDB_API_KEY not set in src/config.py or environment variables.")

    asyncio.run(backfill(args.csv, concurrency=args.concurrency, batch_size=args.batch_size))