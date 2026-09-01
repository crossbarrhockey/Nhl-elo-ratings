import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from scrape_nhl import load_games, TEAM_NAMES, CURRENT_TEAMS
from elo import run_elo, get_preseason_ratings, expected_win_prob, MEAN_ELO

st.set_page_config(page_title="NHL Elo Ratings", layout="wide", page_icon="🏒")


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def compute(force_refresh: bool = False):
    games = load_games(force_refresh=force_refresh)
    elo_df, final = run_elo(games)
    preseason = get_preseason_ratings(final)
    return games, elo_df, final, preseason


st.title("🏒 NHL Elo Ratings")
st.caption(
    "Neil Paine / FiveThirtyEight methodology. "
    "Data from NHL.com, 2005-06 through 2025-26. "
    "Shown ratings are 2026-27 preseason values after 30% regression to 1505."
)

force = st.sidebar.button("Refresh scores from NHL.com")
if force:
    st.cache_data.clear()

progress = st.progress(0, text="Loading games…")
try:
    games, elo_df, final_ratings, preseason = compute(force_refresh=force)
    progress.empty()
except Exception as e:
