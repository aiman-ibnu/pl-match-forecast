import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import common as c
from dashboard import logos
from dashboard import theme as t


def _side(team: str, tag: str, info, n_clubs: int) -> str:
    return f"""
    <div class="side">
      <div class="tag">{tag}</div>
      {t.crest(team, 96)}
      <div class="name">{t.esc(team)}</div>
      <div class="stats"><div><b>{info['elo']:.0f}</b>Elo</div><div><b>#{int(info['rank'])}</b>of {n_clubs}</div>
        <div><b>{info['ppg5']:.1f}</b>Pts/game</div></div>
      <div>{t.form_pills(c.last_form(team, 5))}</div>
    </div>"""


def render():
    t.hero("Prediction", "Match forecast",
           "Pick a home and an away club. The model uses each club's latest Elo rating and form to give the chance of a home win, "
           "a draw and an away win.")
    meta = c.meta(); stt = c.state(); mdl = c.model()
    cur = c.current_clubs()
    clubs = sorted(cur["club"])
    a, b, r = st.columns([2, 2, 2])
    home = a.selectbox("Home club", clubs, index=0)
    away = b.selectbox("Away club", [x for x in clubs if x != home], index=0)
    rest = r.slider("Rest days before the match (both clubs)", 3, 14, 7)

    h, w = stt[home], stt[away]
    row = {"elo_diff": h["elo"] - w["elo"], "elo_home": h["elo"], "elo_away": w["elo"],
           "ppg5_home": h["ppg5"], "ppg5_away": w["ppg5"], "gf5_home": h["gf5"], "gf5_away": w["gf5"],
           "ga5_home": h["ga5"], "ga5_away": w["ga5"], "rest_home": rest, "rest_away": rest}
    X = pd.DataFrame([row])[meta["features"]]
    p = dict(zip(mdl.classes_, mdl.predict_proba(X)[0]))

    info = cur.set_index("club")
    best = max(p, key=p.get)
    verdict = {"H": f"{home} win", "D": "a draw", "A": f"{away} win"}[best]
    seg = "".join(f'<div style="width:{p[k] * 100:.2f}%;background:{t.OUTCOME[k]}" title="{c.LABELS[k]}: {p[k]:.1%}">'
                  f'{p[k]:.0%}</div>' for k in "HDA")
    big = "".join(f'<div style="border-color:{t.OUTCOME[k]}"><b>{p[k] * 100:.0f}<span style="font-size:15px">%</span></b>'
                  f'<span>{lab}</span></div>'
                  for k, lab in [("H", f"{logos.short(home)} win"), ("D", "Draw"), ("A", f"{logos.short(away)} win")])
    st.html(f"""
    <div class="panel" style="margin-top:8px">
      <div class="vs">{_side(home, "Home", info.loc[home], len(cur))}<div class="mid">VS</div>{_side(away, "Away", info.loc[away], len(cur))}</div>
      <div style="margin-top:22px;font-size:13px;color:{t.INK_2}">Most likely: <b style="color:{t.INK}">{t.esc(verdict)}</b>
        · Elo gap {row['elo_diff']:+.0f} (home side)</div>
      <div class="prob">{seg}</div>
      {t.legend_html([(c.LABELS[k], t.OUTCOME[k]) for k in "HDA"])}
      <div class="big3">{big}</div>
    </div>""")
    st.caption(f"Model: {meta['model']}. A probability is not a certainty: even a clear favourite often draws or loses. "
               "The model never picks a draw as the single most likely result (see Insights).")

    # ---------------- head to head ----------------
    t.section("Head to head", f"All league meetings between {home} and {away} since 2014-15, at either ground.")
    m = c.matches()
    hh = m[m["played"] & (((m["home"] == home) & (m["away"] == away)) | ((m["home"] == away) & (m["away"] == home)))].sort_values("date")
    if hh.empty:
        st.info("These clubs have not met in the Premier League in this dataset.")
    else:
        winner = [r.home if r.result == "H" else (r.away if r.result == "A" else "Draw") for r in hh.itertuples()]
        cnt = {home: winner.count(home), "Draw": winner.count("Draw"), away: winner.count(away)}
        l, rr = st.columns([2, 3], gap="large")
        with l:
            total = len(hh)
            bar = "".join(f'<div style="width:{v / total * 100:.2f}%;background:{col}" title="{t.esc(k)}: {v}">{v if v else ""}</div>'
                          for (k, v), col in zip(cnt.items(), [t.OUTCOME["H"], t.OUTCOME["D"], t.OUTCOME["A"]]))
            st.html(f"""<div class="panel">
              <div style="display:flex;justify-content:space-between;align-items:center">
                {t.crest(home, 44)}<div style="text-align:center"><div style="font-size:32px;font-weight:300;color:{t.INK}">
                {cnt[home]} · {cnt['Draw']} · {cnt[away]}</div><div class="note">{total} meetings</div></div>{t.crest(away, 44)}</div>
              <div class="prob" style="height:30px">{bar}</div>
              {t.legend_html([(f"{logos.short(home)} wins", t.OUTCOME["H"]), ("Draws", t.OUTCOME["D"]), (f"{logos.short(away)} wins", t.OUTCOME["A"])])}
            </div>""")
        with rr:
            rows = "".join(
                f'<tr><td>{x.date:%d %b %Y}</td><td class="r">{t.esc(logos.short(x.home))} {t.crest(x.home, 20)}</td>'
                f'<td class="sc">{int(x.home_goals)} - {int(x.away_goals)}</td><td>{t.crest(x.away, 20)} {t.esc(logos.short(x.away))}</td></tr>'
                for x in hh.tail(8).iloc[::-1].itertuples())
            st.html(f'<div class="panel"><table class="h2h">{rows}</table><div class="note">Latest 8 meetings, newest first.</div></div>')

    # ---------------- model quality ----------------
    t.section("How good are these predictions?", "Scored on two held-out seasons (2023-24 and 2024-25, 754 matches) that the model never saw in training.")
    mt = c.metrics_table()
    mt = mt[mt["set"] == "test"][["model", "accuracy", "log_loss", "RPS", "Brier"]].reset_index(drop=True)
    bs = c.bootstrap()
    sel = mt[mt["model"] == meta["model"]].iloc[0]
    elo = mt[mt["model"].str.startswith("Baseline: Elo")].iloc[0]
    home_b = mt[mt["model"].str.startswith("Baseline: always")].iloc[0]
    t.kpi_row([
        t.kpi(f"{sel['accuracy'] * 100:.1f}", "Selected model accuracy", "%", note=meta["model"]),
        t.kpi(f"{elo['accuracy'] * 100:.1f}", "Elo-only baseline", "%", f"{(sel['accuracy'] - elo['accuracy']) * 100:+.1f} pp",
              note="Difference is within noise"),
        t.kpi(f"{home_b['accuracy'] * 100:.1f}", "Always pick home win", "%", f"{(sel['accuracy'] - home_b['accuracy']) * 100:+.1f} pp",
              note=f"95% CI +{bs['vs_always_home']['accuracy']['lo'] * 100:.1f} to +{bs['vs_always_home']['accuracy']['hi'] * 100:.1f} pp"),
        t.kpi(f"{sel['log_loss']:.3f}", "Log loss (lower is better)", "", note=f"Always home: {home_b['log_loss']:.3f}"),
    ])
    l, rr = st.columns([1, 1], gap="large")
    with l:
        d = mt.sort_values("accuracy")
        cols = [t.COPPER_HI if x == meta["model"] else (t.COPPER_LO if x.startswith("Baseline") else "#8E6A52") for x in d["model"]]
        fig = go.Figure(go.Bar(x=d["accuracy"], y=d["model"], orientation="h", marker_color=cols,
                               text=[f"{v:.1%}" for v in d["accuracy"]], textposition="outside",
                               hovertemplate="%{y}<br>Accuracy %{x:.1%}<extra></extra>"))
        fig.update_layout(xaxis=dict(tickformat=".0%", range=[0, 0.7], title="Accuracy on test seasons"), yaxis_title=None, margin=dict(l=10, t=10))
        t.chart(fig, 260, key="acc")
    with rr:
        st.dataframe(mt.round(3), hide_index=True, width="stretch",
                     column_config={"model": "Model", "accuracy": st.column_config.NumberColumn("Accuracy", format="%.3f"),
                                    "log_loss": st.column_config.NumberColumn("Log loss", format="%.3f")})

    # ---------------- browse ----------------
    t.section("Browse test-set predictions", "Each held-out match with the model's probabilities. Filter by season or club; sort by any column.")
    pr = c.predictions(); test = pr[pr["set"] == "test"]
    f1, f2, f3 = st.columns([1, 2, 1])
    s = f1.selectbox("Season", sorted(test["season"].unique()))
    tt = test[test["season"] == s].copy()
    club = f2.multiselect("Clubs", sorted(set(tt["home"]) | set(tt["away"])), format_func=logos.short)
    only = f3.radio("Show", ["All", "Correct", "Wrong"], horizontal=True)
    if club:
        tt = tt[tt["home"].isin(club) | tt["away"].isin(club)]
    tt["correct"] = tt["pred"] == tt["result"]
    if only != "All":
        tt = tt[tt["correct"] == (only == "Correct")]
    tt["hc"] = tt["home"].map(logos.uri); tt["ac"] = tt["away"].map(logos.uri)
    tt["score"] = tt["home_goals"].astype(int).astype(str) + " - " + tt["away_goals"].astype(int).astype(str)
    tt["date"] = tt["date"].dt.date
    st.html(f'<div class="note">{len(tt)} matches · {tt["correct"].mean():.1%} predicted correctly</div>' if len(tt) else "")
    st.dataframe(
        tt[["date", "hc", "home", "score", "away", "ac", "pH", "pD", "pA", "pred", "result", "correct"]],
        hide_index=True, width="stretch", height=420,
        column_config={
            "date": "Date", "hc": st.column_config.ImageColumn("", width=40), "ac": st.column_config.ImageColumn("", width=40),
            "home": "Home", "away": "Away", "score": "Score",
            "pH": st.column_config.ProgressColumn("P(home)", format="%.2f", min_value=0, max_value=1),
            "pD": st.column_config.ProgressColumn("P(draw)", format="%.2f", min_value=0, max_value=1),
            "pA": st.column_config.ProgressColumn("P(away)", format="%.2f", min_value=0, max_value=1),
            "pred": "Predicted", "result": "Actual", "correct": "Correct",
        })
