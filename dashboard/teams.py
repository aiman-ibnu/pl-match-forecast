import plotly.express as px
import streamlit as st

from dashboard import common as c


def render():
    st.header("Team strength (Elo) and form")
    st.write("Elo is a rating that goes up after a win and down after a loss. Beating a strong team moves it more than beating a weak one.")
    h = c.elo_history()
    teams = sorted(h["team"].unique())
    default = [t for t in ["Arsenal", "Liverpool", "Manchester City", "Chelsea"] if t in teams]
    pick = st.multiselect("Clubs", teams, default=default)
    seasons = sorted(h["season"].unique())
    lo, hi = st.select_slider("Seasons", options=seasons, value=(seasons[0], seasons[-1]))
    d = h[h["team"].isin(pick) & (h["season"] >= lo) & (h["season"] <= hi)]
    if d.empty:
        st.info("Pick at least one club.")
    else:
        st.plotly_chart(px.line(d, x="date", y="elo", color="team").update_layout(yaxis_title="Elo rating", xaxis_title=None, legend_title=None), width="stretch")

    st.subheader("Current form (last 5 matches)")
    stt = c.state()
    import pandas as pd
    cur = pd.DataFrame(stt).T.reset_index(names="club")
    latest = cur["last_season"].max()
    cur = cur[cur["last_season"] == latest].sort_values("elo", ascending=False)
    cur = cur.rename(columns={"elo": "Elo", "ppg5": "Points/game", "gf5": "Goals for", "ga5": "Goals against", "last_date": "Last match"})
    st.dataframe(cur[["club", "Elo", "Points/game", "Goals for", "Goals against", "Last match"]].round(2), hide_index=True, width="stretch")
    st.caption(f"Clubs in season {latest}. Values are after the last recorded match.")
