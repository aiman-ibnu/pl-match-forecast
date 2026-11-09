import plotly.express as px
import streamlit as st

from dashboard import common as c
from src import data_processing as dp


def render():
    st.header("Overview: Premier League 2014-15 to 2025-26")
    m = c.matches()
    played = m[m["played"]]
    hw = (played["result"] == "H").mean(); dr = (played["result"] == "D").mean(); aw = (played["result"] == "A").mean()
    k = st.columns(4)
    k[0].metric("Matches played", f"{len(played):,}")
    k[1].metric("Home wins", f"{hw:.1%}")
    k[2].metric("Draws", f"{dr:.1%}")
    k[3].metric("Away wins", f"{aw:.1%}")
    st.caption("Source: openfootball/football.json (CC0). 2025-26 is only partly played, so it is a live partial season.")

    by = played.groupby("season")["result"].value_counts(normalize=True).unstack().reset_index()
    long = by.melt("season", var_name="result", value_name="share")
    long["result"] = long["result"].map(c.LABELS)
    fig = px.line(long, x="season", y="share", color="result", markers=True,
                  color_discrete_map={c.LABELS[k]: c.COLORS[k] for k in "HDA"})
    fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Share of matches", xaxis_title=None, legend_title=None)
    st.subheader("Home advantage over the seasons")
    st.plotly_chart(fig, width="stretch")
    st.caption("Home wins are the most common result in every season; 2020-21 (matches without fans) is the visible dip.")

    goals = played.groupby("season").apply(lambda g: (g["home_goals"] + g["away_goals"]).mean(), include_groups=False).reset_index(name="goals")
    st.subheader("Goals per match")
    st.plotly_chart(px.bar(goals, x="season", y="goals").update_layout(xaxis_title=None, yaxis_title="Goals per match"), width="stretch")

    st.subheader("League table")
    seasons = sorted(played["season"].unique(), reverse=True)
    s = st.selectbox("Season", seasons)
    st.dataframe(dp.season_table(m, s), hide_index=True, width="stretch")
