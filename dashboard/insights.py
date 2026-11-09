import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard import common as c
from src import evaluation as ev


def render():
    st.header("Insights")
    pr = c.predictions(); test = pr[pr["set"] == "test"]
    bs = c.bootstrap()

    st.subheader("1. Elo already carries almost all the signal")
    ab = c.ablation()
    st.dataframe(ab.round(3), hide_index=True, width="stretch")
    st.write(f"Adding form and rest days to Elo changes log loss by {bs['vs_elo_only']['log_loss']['diff']:+.4f} "
             f"(CI {bs['vs_elo_only']['log_loss']['lo']:+.4f} to {bs['vs_elo_only']['log_loss']['hi']:+.4f}). "
             "That interval includes zero, so the extra features did not help on these test seasons.")

    st.subheader("2. Which inputs the model relies on")
    imp = c.importance()
    xcol, ycol = "feature", "logloss_increase"
    st.plotly_chart(px.bar(imp.sort_values(ycol), x=ycol, y=xcol, orientation="h").update_layout(yaxis_title=None, xaxis_title="Log-loss increase when shuffled"), width="stretch")

    st.subheader("3. Are the probabilities honest? (home win)")
    cal = ev.calibration_table(test["result"], test["pH"])
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line=dict(dash="dash", color=c.COLORS["base"]))
    fig.add_scatter(x=cal["pred"], y=cal["actual"], mode="lines+markers", name="Model", line=dict(color=c.COLORS["main"]),
                    text=cal["n"], hovertemplate="Predicted %{x:.2f}<br>Actual %{y:.2f}<br>n=%{text}")
    fig.update_layout(xaxis_title="Predicted P(home win)", yaxis_title="Actual home-win rate")
    st.plotly_chart(fig, width="stretch")
    st.caption("Bins with few matches are noisy (see n on hover).")

    st.subheader("4. The model never predicts a draw")
    cm = test.groupby(["result", "pred"]).size().unstack(fill_value=0)
    st.dataframe(cm, width="stretch")
    st.write("Draws happen in about 23% of matches, and the model's draw probability never exceeds about 32% on the test seasons, so a home or away win is always more likely. "
             "Draw probability is still reported in the Prediction page.")

    st.subheader("Limitations")
    st.markdown("- Only final scores are used (no shots, xG, injuries, lineups, bookmaker odds).\n"
                "- Two test seasons (754 matches); a different split could shift the numbers.\n"
                "- 2025-26 data is incomplete, so it is shown as a partial check only.\n"
                "- Not for betting; it is a learning project.")
