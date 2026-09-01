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

    # Hockey-Reference hides many tables inside HTML comments
    html = r.text.replace("<!--", "").replace("-->", "")

    try:
        tables = pd.read_html(html)
    except Exception as e:
        print(f"Could not parse tables for {season_start}-{season_start + 1}: {e}")
        return pd.DataFrame()

    df = None
    for t in tables:
        cols = [str(c).strip() for c in t.columns]
        if "Date" in cols and ("Visitor" in cols or "Home" in cols):
            df = t
            break

    if df is None:
        print(f"No games table found for {
