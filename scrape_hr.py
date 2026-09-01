"""
Hockey-Reference NHL historical game scraper.
Scrapes regular-season + playoff results from 1917-18 through end_year (inclusive).
end_year=2025 → full 2025-26 season.
Caches results for fast subsequent runs.
"""

import time
from pathlib import Path
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; NHL-Elo-Bot/1.1; "
        "+https://github.com/yourusername/nhl-elo)"
    )
}

def season_to_hr_year(season_start: int) -> int:
    """HR uses the ending calendar year on most season pages."""
    return season_start + 1

def scrape_season_games(season_start: int) -> pd.DataFrame:
    year = season_to_hr_year(season_start)
    url = f"https://www.hockey-reference.com/leagues/NHL_{year}_games.html"

    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"Failed {url}: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(r.content, "lxml")
    table = soup.find("table", id="games")
    if table is None:
        tables = soup.find_all("table")
        table = tables[0] if tables else None
    if table is None:
        return pd.DataFrame()

    try:
        df = pd.read_html(str(table))[0]
    except Exception:
        return pd.DataFrame()

    df.columns = [str(c).strip() for c in df.columns]
    rename = {
        "Date": "date",
        "Visitor": "away_team",
        "G": "away_goals",
        "Home": "home_team",
        "G.1": "home_goals",
        "Unnamed: 5": "ot",
        "Notes": "notes",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    keep = [c for c in ["date", "away_team", "away_goals", "home_team", "home_goals", "ot", "notes"] if c in df.columns]
    df = df[keep].copy()

    df = df[df["date"].notna()]
    df = df[~df["date"].astype(str).str.contains("Date", na=False)]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    for col in ["away_goals", "home_goals"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["away_goals", "home_goals"])

    df["season"] = f"{season_start}-{str(season_start + 1)[-2:]}"
    df["season_start"] = season_start

    # Playoff flag (April+ or Notes keywords)
    df["playoff"] = df["date"].dt.month >= 4
    if "notes" in df.columns:
        keywords = r"Final|Conference|Division|Quarter|Semi|Stanley|Playoff"
        df["playoff"] = df["playoff"] | df["notes"].astype(str).str.contains(keywords, case=False, na=False)

    df["away_team"] = df["away_team"].str.strip()
    df["home_team"] = df["home_team"].str.strip()
    if "ot" in df.columns:
        df["ot"] = df["ot"].fillna("").astype(str).str.upper()
    else:
        df["ot"] = ""

    return df.reset_index(drop=True)


def scrape_all_seasons(start_year: int = 1917, end_year: int = 2025, force_refresh: bool = False) -> pd.DataFrame:
    cache = DATA_DIR / f"nhl_games_{start_year}_{end_year}.csv"
    if cache.exists() and not force_refresh:
        print(f"Loading cached data → {cache}")
        return pd.read_csv(cache, parse_dates=["date"])

    frames = []
    for y in tqdm(range(start_year, end_year + 1), desc="Scraping Hockey-Reference"):
        df = scrape_season_games(y)
        if not df.empty:
            frames.append(df)
        time.sleep(1.6)  # polite rate limit

    if not frames:
        raise RuntimeError("No games were scraped")

    full = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    full.to_csv(cache, index=False)
    print(f"Saved {len(full):,} games → {cache}")
    return full


if __name__ == "__main__":
    games = scrape_all_seasons(1917, 2025, force_refresh=False)
    print(games.tail(10))
    print(f"\nTotal games: {len(games):,}")
    print(f"Range: {games['date'].min().date()} → {games['date'].max().date()}")
