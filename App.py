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
    progress.empty()
    st.error(f"Could not load NHL data: {e}")
    st.info("If this is the first run, reboot the app. It saves each season and will continue next time.")
    st.stop()

view = st.sidebar.radio(
    "View",
    ["2026-27 Preseason Rankings", "Historical Trajectories", "Win Probability", "About"],
)

if view == "2026-27 Preseason Rankings":
    rows = []
    for team, elo in preseason.items():
        if team in CURRENT_TEAMS:
            rows.append({"Team": TEAM_NAMES.get(team, team), "Abbrev": team, "Elo": elo})
    rank_df = pd.DataFrame(rows).sort_values("Elo", ascending=False).reset_index(drop=True)
    rank_df.insert(0, "Rank", range(1, len(rank_df) + 1))

    st.dataframe(
        rank_df.style.format({"Elo": "{:.1f}"}).background_gradient(subset=["Elo"], cmap="RdYlGn"),
        use_container_width=True,
        height=780,
    )
    st.download_button(
        "Download rankings CSV",
        rank_df.to_csv(index=False),
        "nhl_elo_2026_27_preseason.csv",
        "text/csv",
    )
    st.caption(
        f"Games used: {len(games):,} | "
        f"{pd.to_datetime(games['date']).min().date()} → {pd.to_datetime(games['date']).max().date()}"
    )

elif view == "Historical Trajectories":
    teams = sorted([t for t in elo_df["home_team"].unique() if t in CURRENT_TEAMS])
    labels = {t: TEAM_NAMES.get(t, t) for t in teams}
    default = [t for t in ["CAR", "COL", "EDM"] if t in teams] or teams[:3]
    selected = st.multiselect("Teams", teams, default=default, format_func=lambda t: labels[t])
    if selected:
        pieces = []
        for t in selected:
            mask = (elo_df["home_team"] == t) | (elo_df["away_team"] == t)
            sub = elo_df.loc[mask].copy()
            sub["elo"] = np.where(sub["home_team"] == t, sub["home_elo_post"], sub["away_elo_post"])
            sub["team"] = labels[t]
            pieces.append(sub[["date", "team", "elo"]])
        hist = pd.concat(pieces)
        fig = px.line(hist, x="date", y="elo", color="team", title="Elo over time")
        fig.add_hline(y=MEAN_ELO, line_dash="dash", annotation_text="League average 1505")
        st.plotly_chart(fig, use_container_width=True)

elif view == "Win Probability":
    teams = [t for t in CURRENT_TEAMS if t in preseason]
    c1, c2, c3 = st.columns(3)
    with c1:
        home = st.selectbox("Home", teams, format_func=lambda t: TEAM_NAMES.get(t, t))
    with c2:
        away = st.selectbox("Away", teams, index=min(1, len(teams) - 1), format_func=lambda t: TEAM_NAMES.get(t, t))
    with c3:
        playoff = st.checkbox("Playoff game", value=False)
    if home and away and home != away:
        p = expected_win_prob(preseason[home], preseason[away], home=True, playoff=playoff)
        st.metric(f"{TEAM_NAMES[home]} win probability", f"{p * 100:.1f}%")
        st.metric(f"{TEAM_NAMES[away]} win probability", f"{(1 - p) * 100:.1f}%")
        st.write(f"Home Elo {preseason[home]:.1f}  |  Away Elo {preseason[away]:.1f}")

else:
    st.markdown(
        """
        Same formulas as Neil Paine's 2021 NHL Elo article.

        - K = 6
        - Home ice = +50
        - MOV multiplier = `0.6686 * ln(margin) + 0.8048`
        - Autocorrelation = `2.05 / (WinnerEloDiff * 0.001 + 2.05)`
        - Playoff EloDiff × 1.25
        - 30% reversion to 1505 after each season
        - Expansion / relocated clubs start near 1490

        Data: NHL.com API only, 2005-06 through 2025-26.
        Atlanta maps to Winnipeg. Phoenix/Arizona map to Utah.
        """
    )
