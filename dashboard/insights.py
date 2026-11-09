import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import common as c
from dashboard import theme as t
from src import evaluation as ev

FEATURE_NAMES = {
    "elo_diff": "Elo difference", "elo_home": "Home Elo", "elo_away": "Away Elo",
    "ppg5_home": "Home points/game (5)", "ppg5_away": "Away points/game (5)", "gf5_home": "Home scored/game (5)",
    "gf5_away": "Away scored/game (5)", "ga5_home": "Home conceded/game (5)", "ga5_away": "Away conceded/game (5)",
    "rest_home": "Home rest days", "rest_away": "Away rest days",
}


def render():
    t.hero("Insights", "What the model learned",
           "Five findings from the held-out test seasons (2023-24 and 2024-25), plus the limits of this project.")
    pr = c.predictions(); test = pr[pr["set"] == "test"].copy()
    bs = c.bootstrap()
    test["correct"] = test["pred"] == test["result"]

    t.kpi_row([
        t.kpi(f"{test['correct'].mean() * 100:.1f}", "Test accuracy", "%", note=f"{len(test)} matches"),
        t.kpi(f"{(test['result'] == 'D').mean() * 100:.1f}", "Draws that happened", "%", note="Share of test matches"),
        t.kpi(f"{(test['pred'] == 'D').sum()}", "Draws predicted", "", note=f"Max P(draw) = {test['pD'].max():.0%}"),
        t.kpi(f"{bs['vs_elo_only']['log_loss']['diff']:+.4f}", "Log-loss vs Elo only", "",
              note=f"CI {bs['vs_elo_only']['log_loss']['lo']:+.4f} to {bs['vs_elo_only']['log_loss']['hi']:+.4f}"),
    ])

    # 1. ablation
    a, b = st.columns(2, gap="large")
    with a:
        t.section("1. Elo already carries almost all the signal",
                  "Log loss of the model trained on different feature sets (lower is better). Form alone is clearly worse; "
                  "everything with Elo lands in the same place.")
        ab = c.ablation().sort_values("log_loss", ascending=False)
        best = ab["log_loss"].min()
        fig = go.Figure()
        fig.add_scatter(x=ab["log_loss"], y=ab["features"], mode="markers+text",
                        marker=dict(size=13, color=[t.COPPER_HI if v == best else t.COPPER for v in ab["log_loss"]], line=dict(color=t.BG, width=2)),
                        text=[f"{v:.3f}" for v in ab["log_loss"]], textposition="middle right", textfont=dict(color=t.INK_2),
                        customdata=np.stack([ab["accuracy"], ab["n"]], axis=1),
                        hovertemplate="<b>%{y}</b><br>Log loss %{x:.4f}<br>Accuracy %{customdata[0]:.1%}<br>%{customdata[1]} features<extra></extra>")
        fig.update_layout(xaxis=dict(title="Log loss (test)", range=[0.95, 1.03]), yaxis_title=None, margin=dict(l=10, t=10))
        t.chart(fig, 300, key="abl")
        st.caption(f"Adding form and rest days to Elo changes log loss by {bs['vs_elo_only']['log_loss']['diff']:+.4f}; "
                   "the confidence interval includes zero, so the extra features did not help on these seasons.")

    # 2. importance
    with b:
        t.section("2. Which inputs the model relies on",
                  "How much worse log loss gets when one input is shuffled. Longer bar = the model leans on it more. Whiskers: ±1 std.")
        imp = c.importance().sort_values("logloss_increase")
        fig = go.Figure(go.Bar(
            x=imp["logloss_increase"], y=imp["feature"].map(FEATURE_NAMES), orientation="h",
            marker_color=[t.COPPER if f.startswith("elo") else t.COPPER_LO for f in imp["feature"]],
            error_x=dict(type="data", array=imp["std"], color=t.MUTED, thickness=1, width=3),
            hovertemplate="%{y}<br>+%{x:.4f} log loss<extra></extra>"))
        fig.update_layout(xaxis_title="Log-loss increase when shuffled", yaxis_title=None, margin=dict(l=10, t=10))
        t.chart(fig, 300, key="imp")

    # 3. calibration + 4. accuracy by Elo gap
    a, b = st.columns(2, gap="large")
    with a:
        t.section("3. Are the probabilities honest? (home win)",
                  "When the model says 60%, does the home side win about 60% of the time? Dots near the dashed line = honest. Dot size = number of matches.")
        cal = ev.calibration_table(test["result"], test["pH"])
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line=dict(dash="dash", color=t.MUTED, width=1))
        fig.add_scatter(x=cal["pred"], y=cal["actual"], mode="lines+markers", name="Model", line=dict(color=t.COPPER, width=2),
                        marker=dict(size=np.clip(np.sqrt(cal["n"]) * 2.2, 8, 30), color=t.COPPER, line=dict(color=t.BG, width=2)),
                        customdata=cal["n"], hovertemplate="Predicted %{x:.0%}<br>Actual %{y:.0%}<br>n = %{customdata}<extra></extra>")
        fig.update_layout(xaxis=dict(title="Predicted P(home win)", tickformat=".0%", range=[-0.03, 1.03]),
                          yaxis=dict(title="Actual home-win rate", tickformat=".0%", range=[-0.03, 1.05]))
        t.chart(fig, 360, key="cal")
    with b:
        t.section("4. Bigger Elo gap, easier call",
                  "Accuracy by how far apart the two clubs' ratings were before kick-off. Dotted line: random guessing (33%).")
        gap = test["elo_diff"].abs()
        bins = [0, 50, 100, 150, 200, 300, np.inf]
        labs = ["0-50", "50-100", "100-150", "150-200", "200-300", "300+"]
        test["gap"] = pd.cut(gap, bins, labels=labs, right=False)
        g = test.groupby("gap", observed=False).agg(acc=("correct", "mean"), n=("correct", "size")).reset_index()
        fig = go.Figure(go.Bar(x=g["gap"].astype(str), y=g["acc"], marker_color=t.COPPER, customdata=g["n"],
                               text=[f"{v:.0%}" for v in g["acc"]], textposition="outside",
                               hovertemplate="Elo gap %{x}<br>Accuracy %{y:.1%}<br>n = %{customdata}<extra></extra>"))
        fig.add_hline(y=1 / 3, line=dict(color=t.INK_2, dash="dot", width=1), layer="above")
        fig.update_layout(xaxis_title="Absolute Elo difference", yaxis=dict(title="Accuracy", tickformat=".0%", range=[0, 1]))
        t.chart(fig, 360, key="gap")

    # 5. confusion
    t.section("5. The model never predicts a draw",
              "Rows: what happened. Columns: what the model picked. Draws happen in about a quarter of matches, but the model's "
              f"draw probability never goes above {test['pD'].max():.0%}, so a home or away win always looks more likely.")
    order = ["H", "D", "A"]
    cm = pd.crosstab(test["result"], test["pred"]).reindex(index=order, columns=order, fill_value=0)
    a, b = st.columns([3, 2], gap="large")
    with a:
        names = [c.LABELS[k] for k in order]
        fig = go.Figure(go.Heatmap(z=cm.values, x=[f"Picked: {n}" for n in names], y=[f"Actual: {n}" for n in names],
                                   colorscale=t.SEQ, xgap=3, ygap=3, showscale=False, text=cm.values, texttemplate="%{text}",
                                   textfont=dict(size=16), hovertemplate="%{y}<br>%{x}<br>%{z} matches<extra></extra>"))
        fig.update_layout(yaxis_autorange="reversed", xaxis_side="top", xaxis_showgrid=False, yaxis_showgrid=False, margin=dict(t=40))
        t.chart(fig, 300, key="cm")
    with b:
        st.html(f"""<div class="panel"><div style="color:{t.INK};font-size:15px;margin-bottom:8px">Limitations</div>
          <ul style="color:{t.INK_2};font-size:13px;line-height:1.6;padding-left:18px;margin:0">
            <li>Only final scores are used (no shots, xG, injuries, lineups or bookmaker odds).</li>
            <li>Two test seasons (754 matches); a different split could shift the numbers.</li>
            <li>2025-26 is incomplete, so it is only a partial check.</li>
            <li>Not for betting; it is a learning project.</li></ul></div>""")
