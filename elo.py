"""Neil Paine / FiveThirtyEight NHL Elo methodology."""

from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import pandas as pd

K_FACTOR = 6.0
HOME_ADVANTAGE = 50.0
REVERSION = 0.30
MEAN_ELO = 1505.0
EXPANSION_ELO = 1490.0
PLAYOFF_MULT = 1.25

EXPANSION_START = {
    "VGK": 2017,
    "SEA": 2021,
    "UTA": 2024,
}


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


def run_elo(games: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    games = games.copy().sort_values(["date", "game_id"] if "game_id" in games.columns else ["date"])
    ratings: Dict[str, float] = {}
    history = []
    current_season = None

    for _, row in games.iterrows():
        season = int(row["season_start"])
        if current_season is not None and season != current_season:
            for team in list(ratings):
                ratings[team] = (1 - REVERSION) * ratings[team] + REVERSION * MEAN_ELO
        current_season = season

        away = str(row["away_team"])
        home = str(row["home_team"])
        away_g = int(row["away_goals"])
        home_g = int(row["home_goals"])
        playoff = bool(row.get("playoff", False))
        neutral = bool(row.get("neutral", False))

        if away not in ratings:
            ratings[away] = EXPANSION_ELO if EXPANSION_START.get(away, 9999) >= season - 1 else MEAN_ELO
        if home not in ratings:
            ratings[home] = EXPANSION_ELO if EXPANSION_START.get(home, 9999) >= season - 1 else MEAN_ELO

        away_pre = ratings[away]
        home_pre = ratings[home]

        if home_g == away_g:
            continue

        if home_g > away_g:
            margin = home_g - away_g
            new_h, new_a = update_elo(
                home_pre, away_pre, margin, winner_was_home=not neutral, playoff=playoff
            )
            ratings[home], ratings[away] = new_h, new_a
            winner = home
        else:
            margin = away_g - home_g
            new_a, new_h = update_elo(
                away_pre, home_pre, margin, winner_was_home=False, playoff=playoff
            )
            ratings[away], ratings[home] = new_a, new_h
            winner = away

        history.append(
            {
                "date": row["date"],
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
            }
        )

    return pd.DataFrame(history), dict(ratings)


def get_preseason_ratings(final_ratings: Dict[str, float]) -> Dict[str, float]:
    return {t: (1 - REVERSION) * e + REVERSION * MEAN_ELO for t, e in final_ratings.items()}
