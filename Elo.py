"""
NHL Elo engine – faithful implementation of the 2021 FiveThirtyEight / Neil Paine methodology.
"""

from collections import defaultdict
from typing import Dict, Tuple
import numpy as np
import pandas as pd

# Parameters straight from the article
K_FACTOR = 6.0
HOME_ADVANTAGE = 50.0
REVERSION = 0.30
MEAN_ELO = 1505.0
EXPANSION_ELO_CAP = 1490.0      # post-2005
EXPANSION_ELO_OLD = 1380.0
PLAYOFF_MULT = 1.25

# Hockey-Reference full names → stable abbreviations
TEAM_MAP = {
    "Anaheim Ducks": "ANA", "Arizona Coyotes": "ARI", "Utah Hockey Club": "UTA",
    "Utah Mammoth": "UTA", "Boston Bruins": "BOS", "Buffalo Sabres": "BUF",
    "Calgary Flames": "CGY", "Carolina Hurricanes": "CAR", "Chicago Blackhawks": "CHI",
    "Colorado Avalanche": "COL", "Columbus Blue Jackets": "CBJ", "Dallas Stars": "DAL",
    "Detroit Red Wings": "DET", "Edmonton Oilers": "EDM", "Florida Panthers": "FLA",
    "Los Angeles Kings": "LAK", "Minnesota Wild": "MIN", "Montreal Canadiens": "MTL",
    "Nashville Predators": "NSH", "New Jersey Devils": "NJD", "New York Islanders": "NYI",
    "New York Rangers": "NYR", "Ottawa Senators": "OTT", "Philadelphia Flyers": "PHI",
    "Pittsburgh Penguins": "PIT", "San Jose Sharks": "SJS", "Seattle Kraken": "SEA",
    "St. Louis Blues": "STL", "Tampa Bay Lightning": "TBL", "Toronto Maple Leafs": "TOR",
    "Vancouver Canucks": "VAN", "Vegas Golden Knights": "VGK", "Washington Capitals": "WSH",
    "Winnipeg Jets": "WPG",
    # Common historical
    "Quebec Nordiques": "QUE", "Hartford Whalers": "HFD", "Minnesota North Stars": "MNS",
    "Atlanta Thrashers": "ATL", "Phoenix Coyotes": "PHX", "Winnipeg Jets (1979)": "WIN",
    "California Golden Seals": "CGS", "Oakland Seals": "OAK", "Kansas City Scouts": "KCS",
    "Colorado Rockies": "CLR", "Atlanta Flames": "AFM", "Cleveland Barons": "CLE",
}


def standardize_team(name: str) -> str:
    return TEAM_MAP.get(str(name).strip(), str(name).strip())


def expected_win_prob(elo_a: float, elo_b: float, home: bool = False, playoff: bool = False) -> float:
    diff = elo_a - elo_b
    if home:
        diff += HOME_ADVANTAGE
    if playoff:
        diff *= PLAYOFF_MULT
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def mov_multiplier(margin: float) -> float:
    if margin <= 0:
        return 1.0
    return 0.6686 * np.log(margin) + 0.8048


def autocorrelation_adjustment(winner_elo_diff: float) -> float:
    return 2.05 / (winner_elo_diff * 0.001 + 2.05)


def update_elo(
    winner_elo: float,
    loser_elo: float,
    margin: float,
    winner_was_home: bool,
    playoff: bool = False,
) -> Tuple[float, float]:
    exp = expected_win_prob(winner_elo, loser_elo, home=winner_was_home, playoff=playoff)
    mov = mov_multiplier(margin)
    winner_diff = winner_elo - loser_elo + (HOME_ADVANTAGE if winner_was_home else -HOME_ADVANTAGE)
    auto = autocorrelation_adjustment(winner_diff)
    shift = K_FACTOR * mov * auto * (1.0 - exp)
    return winner_elo + shift, loser_elo - shift


def run_elo(games: pd.DataFrame, initial_elo: float = MEAN_ELO) -> Tuple[pd.DataFrame, Dict[str, float]]:
    games = games.copy()
    games["away_team"] = games["away_team"].apply(standardize_team)
    games["home_team"] = games["home_team"].apply(standardize_team)

    ratings: Dict[str, float] = defaultdict(lambda: initial_elo)
    history = []
    current_season = None

    for _, row in games.iterrows():
        season = int(row["season_start"])
        if current_season is not None and season != current_season:
            # Between-season 30 % reversion
            for t in list(ratings):
                ratings[t] = (1 - REVERSION) * ratings[t] + REVERSION * MEAN_ELO
        current_season = season

        away, home = row["away_team"], row["home_team"]
        away_g, home_g = int(row["away_goals"]), int(row["home_goals"])
        playoff = bool(row.get("playoff", False))

        # Expansion starting values
        if away not in ratings:
            ratings[away] = EXPANSION_ELO_CAP if season >= 2005 else EXPANSION_ELO_OLD
        if home not in ratings:
            ratings[home] = EXPANSION_ELO_CAP if season >= 2005 else EXPANSION_ELO_OLD

        away_pre, home_pre = ratings[away], ratings[home]

        if home_g > away_g:
            margin = home_g - away_g
            new_h, new_a = update_elo(home_pre, away_pre, margin, winner_was_home=True, playoff=playoff)
            ratings[home], ratings[away] = new_h, new_a
            winner = home
        else:
            margin = away_g - home_g
            new_a, new_h = update_elo(away_pre, home_pre, margin, winner_was_home=False, playoff=playoff)
            ratings[away], ratings[home] = new_a, new_h
            winner = away

        history.append({
            "date": row["date"],
            "season": row["season"],
            "season_start": season,
            "away_team": away,
            "home_team": home,
            "away_goals": away_g,
            "home_goals": home_g,
            "playoff": playoff,
            "away_elo_pre": away_pre,
            "home_elo_pre": home_pre,
            "away_elo_post": ratings[away],
            "home_elo_post": ratings[home],
            "winner": winner,
        })

    return pd.DataFrame(history), dict(ratings)


def get_preseason_ratings(final_ratings: Dict[str, float]) -> Dict[str, float]:
    """Apply the 30 % reversion for the start of the next season (2026-27)."""
    return {t: (1 - REVERSION) * e + REVERSION * MEAN_ELO for t, e in final_ratings.items()}
