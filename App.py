import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scrape_hr import scrape_all_seasons
from elo import run_elo, get_preseason_ratings, expected_win_prob, MEAN_ELO

st.set_page_config(page_title="NHL Elo Ratings", layout="wide", page_icon="🏒")

@st.cache_data(ttl=60 * 60 * 24)
def load_everything(end_year: int = 2025):
    games = scrape_all_seasons(1917, end_year, force_refresh=False)
    elo_df, final = run_elo(games)
    preseason = get_preseason_ratings(final)
    return games, elo_df, final, preseason


st.title("🏒 NHL Elo Rating System")
st.markdown(
    "Faithful implementation of the **2021 FiveThirtyEight / Neil Paine** methodology.  \n"
    "Data scraped exclusively from [Hockey-Reference](https://www.hockey-reference.com) "
    "through the completed **2025-26** season.  \n"
    "Preseason ratings are ready for the **2026-27** campaign."
)

with st.spinner("Loading data & computing Elo (first run scrapes & caches)…"):
    games, elo_df, final_ratings, preseason = load_everything(2025)

view = st.sidebar.radio(
    "View",
    ["2026-27 Preseason Rankings", "Historical Trajectories", "Win Probability Calculator", "About / Methodology"],
)

# ---------- Rankings ----------
if view == "2026-27 Preseason Rankings":
    st.header("2026-27 Preseason Elo Rankings")
    st.caption("30 % reversion toward 1505 applied after the final 2025-26 game.")

    rank_df = (
        pd.DataFrame.from_dict(preseason, orient="index", columns=["Elo"])
        .sort_values("Elo", ascending=False)
        .reset_index()
        .rename(columns={"index": "Team"})
    )
    rank_df.insert(0, "Rank", range(1, len(rank_df) + 1))

    st.dataframe(
        rank_df.style.format({"Elo": "{:.1f}"}).background_gradient(subset=["Elo"], cmap="RdYlGn"),
        use_container_width=True,
        height=780,
    )
    st.download_button(
        "Download CSV",
        rank_df.to_csv(index=False),
        "nhl_elo_2026_27_preseason.csv",
        "text/csv",
    )

# ---------- History ----------
elif view == "Historical Trajectories":
    st.header("Historical Elo Trajectories")
    teams = sorted(elo_df["home_team"].unique())
    selected = st.multiselect("Teams", teams, default=["CAR", "COL", "EDM", "FLA"][:3])

    if selected:
        pieces = []
        for t in selected:
            mask = (elo_df["home_team"] == t) | (elo_df["away_team"] == t)
            sub = elo_df.loc[mask].copy()
            sub["elo"] = np.where(sub["home_team"] == t, sub["home_elo_post"], sub["away_elo_post"])
            sub["team"] = t
            pieces.append(sub[["date", "team", "elo"]])
        hist = pd.concat(pieces)
        fig = px.line(hist, x="date", y="elo", color="team", title="Elo over time")
        fig.add_hline(y=MEAN_ELO, line_dash="dash", annotation_text="League avg 1505")
        st.plotly_chart(fig, use_container_width=True)

# ---------- Calculator ----------
elif view == "Win Probability Calculator":
    st.header("Game Win Probability")
    teams = sorted(preseason.keys())
    c1, c2, c3 = st.columns(3)
    with c1:
        home = st.selectbox("Home", teams, index=teams.index("CAR") if "CAR" in teams else 0)
    with c2:
        away = st.selectbox("Away", teams, index=teams.index("VGK") if "VGK" in teams else 1)
    with c3:
        playoff = st.checkbox("Playoff game", value=False)

    if home and away:
        p = expected_win_prob(preseason[home], preseason[away], home=True, playoff=playoff)
        st.metric(f"{home} win probability", f"{p*100:.1f}%")
        st.metric(f"{away} win probability", f"{(1-p)*100:.1f}%")
        st.write(f"Home Elo {preseason[home]:.1f}  |  Away Elo {preseason[away]:.1f}")

# ---------- About ----------
else:
    st.header("Methodology (from the 2021 article)")
    st.markdown(
        """
        - **K-factor** = 6  
        - **Home-ice advantage** = +50 Elo  
        - **Margin-of-victory multiplier** = `0.6686 * ln(margin) + 0.8048`  
        - **Autocorrelation adjustment** = `2.05 / (WinnerEloDiff * 0.001 + 2.05)`  
        - **Playoff adjustment** = EloDiff × 1.25  
        - **Between-season reversion** = 30 % toward 1505  
        - **Expansion teams** (post-2005) start at 1490  

        Data source: every game on Hockey-Reference from 1917-18 through the end of the 2025-26 season.  
        Ratings are updated after every game exactly as described in the original post.
        """
    )

st.sidebar.markdown("---")
st.sidebar.info(
    "Scrapes Hockey-Reference only. "
    "First run builds a local cache; subsequent runs are instant."
)
