"""
NHL.com API game loader.
History starts with 2005-06 and runs through 2025-26.
Saves progress after each season so Streamlit can resume if it times out.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
CACHE_PATH = DATA_DIR / "nhl_games.csv"

API = "https://api-web.nhle.com/v1"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; NHL-Elo/1.0)",
    "Accept": "application/json",
}

START_SEASON = 2005   # 2005-06
END_SEASON = 2025     # 2025-26 completed

CURRENT_TEAMS = [
    "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL",
    "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NJD",
    "NSH", "NYI", "NYR", "OTT", "PHI", "PIT", "SEA", "SJS",
    "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WPG", "WSH",
]

EXTRA_TEAMS = ["ARI", "PHX", "ATL"]

FRANCHISE_MAP = {
    "ARI": "UTA",
    "PHX": "UTA",
    "ATL": "WPG",
}

TEAM_NAMES = {
    "ANA": "Anaheim Ducks",
    "BOS": "Boston Bruins",
    "BUF": "Buffalo Sabres",
    "CAR": "Carolina Hurricanes",
    "CBJ": "Columbus Blue Jackets",
    "CGY": "Calgary Flames",
    "CHI": "Chicago Blackhawks",
    "COL": "Colorado Avalanche",
    "DAL": "Dallas Stars",
    "DET": "Detroit Red Wings",
    "EDM": "Edmonton Oilers",
    "FLA": "Florida Panthers",
    "LAK": "Los Angeles Kings",
    "MIN": "Minnesota Wild",
    "MTL": "Montreal Canadiens",
    "NJD": "New Jersey Devils",
    "NSH": "Nashville Predators",
    "NYI": "New York Islanders",
    "NYR": "New York Rangers",
    "OTT": "Ottawa Senators",
    "PHI": "Philadelphia Flyers",
    "PIT": "Pittsburgh Penguins",
    "SEA": "Seattle Kraken",
    "SJS": "San Jose Sharks",
    "STL": "St. Louis Blues",
    "TBL": "Tampa Bay Lightning",
    "TOR": "Toronto Maple Leafs",
    "UTA": "Utah Mammoth",
    "VAN": "Vancouver Canucks",
    "VGK": "Vegas Golden Knights",
    "WPG": "Winnipeg Jets",
    "WSH": "Washington Capitals",
}


def season_id(start_year: int) -> str:
    return f"{start_year}{start_year + 1}"


def map_team(abbrev: str) -> str:
    abbrev = str(abbrev).upper().strip()
    return FRANCHISE_MAP.get(abbrev, abbrev)


def _parse_games(payload: dict) -> list[dict]:
    rows = []
    for g in payload.get("games", []):
        game_type = g.get("gameType")
        if game_type not in (2, 3):
            continue

        state = str(g.get("gameState", "")).upper()
        if state not in {"OFF", "FINAL"}:
            continue

        away = g.get("awayTeam") or {}
        home = g.get("homeTeam") or {}
        away_g = away.get("score")
        home_g = home.get("score")
        if away_g is None or home_g is None:
            continue

        period = (g.get("periodDescriptor") or {}).get("periodType", "")
        season = g.get("season")
        rows.append(
            {
                "game_id": g.get("id"),
                "date": g.get("gameDate"),
                "season_id": season,
                "season_start": int(str(season)[:4]) if season else None,
                "playoff": game_type == 3,
                "away_team": map_team(away.get("abbrev", "")),
                "home_team": map_team(home.get("abbrev", "")),
                "away_goals": int(away_g),
                "home_goals": int(home_g),
                "ot": str(period).upper(),
                "neutral": bool(g.get("neutralSite", False)),
            }
        )
    return rows


def fetch_team_season(team: str, start_year: int) -> pd.DataFrame:
    url = f"{API}/club-schedule-season/{team}/{season_id(start_year)}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            return pd.DataFrame()
        return pd.DataFrame(_parse_games(r.json()))
    except Exception:
        return pd.DataFrame()


def clean_games(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.dropna(subset=["game_id", "away_goals", "home_goals", "date"]).copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    out = out.drop_duplicates(subset=["game_id"]).sort_values("date").reset_index(drop=True)
    return out


def save_games(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    clean_games(df).to_csv(CACHE_PATH, index=False)


def load_cached_games() -> pd.DataFrame:
    if CACHE_PATH.exists():
        try:
            return clean_games(pd.read_csv(CACHE_PATH, parse_dates=["date"]))
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def merge_games(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if old is None or old.empty:
        return clean_games(new)
    if new is None or new.empty:
        return clean_games(old)
    return clean_games(pd.concat([old, new], ignore_index=True))


def fetch_one_season(start_year: int, teams: Iterable[str]) -> pd.DataFrame:
    frames = []
    for team in teams:
        df = fetch_team_season(team, start_year)
        if not df.empty:
            frames.append(df)
        time.sleep(0.12)
    if not frames:
        return pd.DataFrame()
    return clean_games(pd.concat(frames, ignore_index=True))


def seasons_present(df: pd.DataFrame) -> set[int]:
    if df is None or df.empty or "season_start" not in df.columns:
        return set()
    return set(int(x) for x in df["season_start"].dropna().unique())


def load_games(force_refresh: bool = False, progress=None) -> pd.DataFrame:
    cached = pd.DataFrame() if force_refresh else load_cached_games()
    teams = CURRENT_TEAMS + EXTRA_TEAMS
    have = seasons_present(cached)

    needed = [y for y in range(START_SEASON, END_SEASON + 1) if y not in have]
    total = max(len(needed), 1)

    for i, year in enumerate(needed):
        if progress:
            progress.progress(
                min(0.9, (i + 1) / (total + 1)),
                text=f"Downloading {year}-{str(year + 1)[-2:]} from NHL.com…",
            )
        season_df = fetch_one_season(year, teams)
        if not season_df.empty:
            cached = merge_games(cached, season_df)
            save_games(cached)

    if progress:
        progress.progress(0.95, text="Refreshing 2025-26 and 2026-27…")

    latest = merge_games(
        fetch_one_season(END_SEASON, teams),
        fetch_one_season(END_SEASON + 1, CURRENT_TEAMS),
    )
    if not latest.empty:
        cached = merge_games(cached, latest)
        save_games(cached)

    if cached.empty:
        raise RuntimeError("NHL.com returned no games. Click Refresh and try again.")

    if progress:
        progress.progress(1.0, text="Done")
    return cached


if __name__ == "__main__":
    games = load_games()
    print(games.tail(10))
    print(f"Games: {len(games):,}")
    print(f"Range: {games['date'].min().date()} → {games['date'].max().date()}")
