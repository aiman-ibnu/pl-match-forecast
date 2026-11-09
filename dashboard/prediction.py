import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import common as c


def render():
    st.header("Match prediction")
    meta = c.meta(); stt = c.state(); mdl = c.model()
    latest = max(v["last_season"] for v in stt.values())
    clubs = sorted(t for t, v in stt.items() if v["last_season"] == latest)
    a, b = st.columns(2)
    home = a.selectbox("Home club", clubs, index=0)
    away = b.selectbox("Away club", [x for x in clubs if x != home], index=0)
    rest = st.slider("Rest days before the match (both clubs)", 3, 14, 7)

    h, w = stt[home], stt[away]
    row = {"elo_diff": h["elo"] - w["elo"], "elo_home": h["elo"], "elo_away": w["elo"],
           "ppg5_home": h["ppg5"], "ppg5_away": w["ppg5"], "gf5_home": h["gf5"], "gf5_away": w["gf5"],
           "ga5_home": h["ga5"], "ga5_away": w["ga5"], "rest_home": rest, "rest_away": rest}
    X = pd.DataFrame([row])[meta["features"]]
    p = dict(zip(mdl.classes_, mdl.predict_proba(X)[0]))
    fig = go.Figure(go.Bar(x=[c.LABELS[k] for k in "HDA"], y=[p[k] for k in "HDA"],
                           marker_color=[c.COLORS[k] for k in "HDA"], text=[f"{p[k]:.0%}" for k in "HDA"], textposition="outside"))
    fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1], title=f"{home} vs {away}")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Model: {meta['model']}. Uses each club's latest recorded Elo and form. This is a probability, not a certainty: "
               "even a clear favourite loses or draws often.")

    st.subheader("How good are these predictions? (test seasons only)")
    pr = c.predictions(); test = pr[pr["set"] == "test"]
    mt = c.metrics_table(); mt = mt[mt["set"] == "test"][["model", "accuracy", "log_loss", "RPS", "Brier"]]
    st.dataframe(mt.round(3), hide_index=True, width="stretch")
    bs = c.bootstrap()
    st.write(f"Versus always predicting a home win, accuracy is higher by {bs['vs_always_home']['accuracy']['diff']:.3f} "
             f"(95% CI {bs['vs_always_home']['accuracy']['lo']:.3f} to {bs['vs_always_home']['accuracy']['hi']:.3f}). "
             f"Versus the Elo-only baseline the difference is {bs['vs_elo_only']['accuracy']['diff']:+.3f} "
             f"(CI {bs['vs_elo_only']['accuracy']['lo']:+.3f} to {bs['vs_elo_only']['accuracy']['hi']:+.3f}), i.e. no real difference.")

    st.subheader("Browse test-set predictions")
    seasons = sorted(test["season"].unique())
    s = st.selectbox("Season", seasons)
    t = test[test["season"] == s].copy()
    t["match"] = t["home"] + " " + t["home_goals"].astype(int).astype(str) + "-" + t["away_goals"].astype(int).astype(str) + " " + t["away"]
    t["correct"] = t["pred"] == t["result"]
    st.dataframe(t.assign(date=t["date"].dt.date).round({"pH": 2, "pD": 2, "pA": 2})[["date", "match", "result", "pred", "pH", "pD", "pA", "correct"]], hide_index=True, width="stretch")
