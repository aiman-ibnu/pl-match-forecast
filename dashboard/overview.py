import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import common as c
from dashboard import logos
from dashboard import theme as t


def _season_stats(played: pd.DataFrame) -> pd.DataFrame:
    g = played.groupby("season")
    s = pd.DataFrame({
        "matches": g.size(),
        "H": g["result"].apply(lambda r: (r == "H").mean()),
        "D": g["result"].apply(lambda r: (r == "D").mean()),
        "A": g["result"].apply(lambda r: (r == "A").mean()),
        "goals": g.apply(lambda x: (x["home_goals"] + x["away_goals"]).mean(), include_groups=False),
    })
    return s.reset_index()


def render():
    m = c.matches()
    played = m[m["played"]]
    seasons = sorted(played["season"].unique(), reverse=True)
    latest = seasons[0]
    partial = played.groupby("season").size().lt(380)

    t.hero("Overview · 2014-15 to 2025-26", "Premier League at a glance",
           "Twelve seasons of results. Pick a season to update the cards, the points race, the table and the score-line map. "
           "Hover any bar, line or cell for exact values.")

    f1, f2 = st.columns([1, 3])
    season = f1.selectbox("Season", seasons, help="2025-26 is still being played (partial season).")
    is_partial = bool(partial.get(season, False))
    f2.html(f'<div class="note" style="margin-top:34px">{"⏳ Partial season: " + str(int(played[played.season == season].shape[0])) + " of 380 matches played." if is_partial else "✔ Complete season (380 matches)."}</div>')

    # ---------------- KPI cards ----------------
    st_ = _season_stats(played)
    order = st_["season"].tolist()
    hi = order.index(season)
    row = st_.iloc[hi]
    avg = st_.mean(numeric_only=True)
    tables = c.season_tables()
    champs = [(s, tables[s].iloc[0]) for s in order]
    top = tables[season].iloc[0]

    def pp(v):  # percentage-point delta vs 12-season average
        return f"{v * 100:+.1f} pp"

    cards = [
        t.kpi(int(top["pts"]), "Leader" if is_partial else "Champion", "pts", note=top["team"],
              lead=t.crest(top["team"], 40),
              spark=t.spark_bars([ch["pts"] for _, ch in champs], [f"{s} · {ch['team']}" for s, ch in champs], hi, "{:.0f} pts")),
        t.kpi(f"{row['H'] * 100:.1f}", "Home wins", "%", pp(row["H"] - avg["H"]), note="vs 12-season avg",
              spark=t.spark_bars(st_["H"] * 100, order, hi, "{:.1f}%")),
        t.kpi(f"{row['D'] * 100:.1f}", "Draws", "%", pp(row["D"] - avg["D"]), note="vs 12-season avg",
              spark=t.spark_bars(st_["D"] * 100, order, hi, "{:.1f}%")),
        t.kpi(f"{row['A'] * 100:.1f}", "Away wins", "%", pp(row["A"] - avg["A"]), note="vs 12-season avg",
              spark=t.spark_bars(st_["A"] * 100, order, hi, "{:.1f}%")),
        t.kpi(f"{row['goals']:.2f}", "Goals per match", "", f"{row['goals'] - avg['goals']:+.2f}", note="vs 12-season avg",
              spark=t.spark_bars(st_["goals"], order, hi, "{:.2f}")),
        t.kpi(f"{len(played):,}", "Matches in dataset", "", note=f"{len(order)} seasons",
              spark=t.spark_bars(st_["matches"], order, hi, "{:.0f} matches")),
    ]
    t.kpi_row(cards)

    # ---------------- points race + table ----------------
    tr = c.team_results()
    tr = tr[tr["season"] == season].sort_values(["team", "date"]).copy()
    tr["game"] = tr.groupby("team").cumcount() + 1
    tr["cum"] = tr.groupby("team")["pts"].cumsum()
    table = tables[season]
    clubs_by_pos = table["team"].tolist()

    t.section("Points race", "Cumulative points after each game. Highlighted clubs get a colour and their crest at the end of the line; the rest stay grey for context.")
    pick = st.multiselect("Highlight clubs (max 8)", clubs_by_pos, default=clubs_by_pos[:6], max_selections=8,
                          key=f"race_{season}", format_func=logos.short)
    fig = go.Figure()
    for team in clubs_by_pos:
        if team in pick:
            continue
        d = tr[tr["team"] == team]
        fig.add_scatter(x=d["game"], y=d["cum"], mode="lines", line=dict(color="rgba(184,172,160,.18)", width=1),
                        name=team, showlegend=False, hovertemplate=f"{team}<br>Game %{{x}}: %{{y}} pts<extra></extra>")
    ends = []
    for i, team in enumerate(pick):
        d = tr[tr["team"] == team]
        fig.add_scatter(x=d["game"], y=d["cum"], mode="lines", line=dict(color=t.SERIES[i], width=2.2), name=logos.short(team),
                        customdata=np.stack([d["opp"], d["gf"], d["ga"], d["res"]], axis=1),
                        hovertemplate=f"<b>{team}</b><br>Game %{{x}}: %{{y}} pts<br>%{{customdata[3]}} %{{customdata[1]:.0f}}-%{{customdata[2]:.0f}} vs %{{customdata[0]}}<extra></extra>")
        ends.append((team, d["game"].iloc[-1] + 1.6, d["cum"].iloc[-1]))
    ymax = tr["cum"].max()
    t.add_crests(fig, ends, 2.6, ymax * 0.075)
    fig.update_layout(xaxis_title="Games played", yaxis_title="Points", xaxis_range=[0, tr["game"].max() + 3],
                      hovermode="closest", legend=dict(y=1.0, yanchor="bottom"))
    t.chart(fig, 470, key="race")

    right, b = st.columns([6, 5], gap="large")
    with right:
        t.section("League table", "Sorted by points, goal difference, goals scored. Click a column header to re-sort.")
        form = tr.sort_values("date").groupby("team")["res"].apply(lambda r: "".join(r.tail(5)))
        tb = table.copy()
        tb.insert(1, "crest", tb["team"].map(logos.uri))
        tb["form"] = tb["team"].map(form)
        st.dataframe(
            tb[["pos", "crest", "team", "played", "w", "d", "l", "gd", "pts", "form"]],
            hide_index=True, width="stretch", height=36 * len(tb) + 40,
            column_config={
                "pos": st.column_config.NumberColumn("#", width=40),
                "crest": st.column_config.ImageColumn("", width=40),
                "team": st.column_config.TextColumn("Club", width=170),
                "played": st.column_config.NumberColumn("P", width=40),
                "w": st.column_config.NumberColumn("W", width=40), "d": st.column_config.NumberColumn("D", width=40),
                "l": st.column_config.NumberColumn("L", width=40), "gd": st.column_config.NumberColumn("GD", format="%+d", width=50),
                "pts": st.column_config.ProgressColumn("Pts", format="%d", min_value=0, max_value=int(tb["pts"].max()), width="small"),
                "form": st.column_config.TextColumn("Last 5", help="Oldest to newest. W win, D draw, L loss."),
            })

    a, g = st.columns(2, gap="large")
    with a:
        t.section("Home advantage over the seasons", "Share of home wins, draws and away wins. The 2020-21 dip in home wins is the season played without fans.")
        fig = go.Figure()
        op = [1.0 if s == season else 0.5 for s in order]
        for k in "HDA":
            txt = [f"{v:.0%}" if s == season else "" for s, v in zip(order, st_[k])]
            fig.add_bar(x=order, y=st_[k], name=c.LABELS[k], marker=dict(color=t.OUTCOME[k], opacity=op, line=dict(color=t.BG, width=1.5)),
                        text=txt, textposition="inside", insidetextfont=dict(color="#0E0C0B", size=12),
                        hovertemplate=f"%{{x}}<br>{c.LABELS[k]}: %{{y:.1%}}<extra></extra>")
        fig.update_layout(barmode="stack", yaxis_tickformat=".0%", yaxis_range=[0, 1], bargap=0.18, hovermode="x unified",
                          legend_traceorder="normal")
        t.chart(fig, 380, key="mix")

    with b:
        t.section("Score-line map", "How often each final score happens. Rows: home goals, columns: away goals. Draws sit on the diagonal.")
        all_s = st.toggle("All seasons", value=False, key="score_all")
        d = played if all_s else played[played["season"] == season]
        hg = d["home_goals"].clip(upper=5).astype(int)
        ag = d["away_goals"].clip(upper=5).astype(int)
        grid = pd.crosstab(hg, ag).reindex(index=range(6), columns=range(6), fill_value=0)
        share = grid / grid.values.sum()
        labels = [str(i) for i in range(5)] + ["5+"]
        fig = go.Figure(go.Heatmap(
            z=share.values, x=labels, y=labels, colorscale=t.SEQ, xgap=2, ygap=2, showscale=False,
            text=grid.values, texttemplate="%{text}", textfont=dict(size=12),
            hovertemplate="Home %{y} - %{x} Away<br>%{text} matches (%{z:.1%})<extra></extra>"))
        fig.update_layout(xaxis=dict(title="Away goals", side="top", type="category"), yaxis=dict(title="Home goals", type="category", autorange="reversed"),
                          xaxis_showgrid=False, yaxis_showgrid=False, margin=dict(t=40))
        t.chart(fig, 420, key="scores")
        most = share.stack().idxmax()
        st.html(f'<div class="note">Most common score: <b style="color:{t.INK}">{labels[most[0]]}-{labels[most[1]]}</b> '
                f'({share.values.max():.1%} of {"all" if all_s else season} matches).</div>')

    with g:
        t.section("Goals per match", "Average total goals per match each season (selected season highlighted).")
        fig = go.Figure(go.Bar(x=order, y=st_["goals"], marker=dict(color=[t.COPPER_HI if s == season else t.COPPER_LO for s in order]),
                               text=[f"{v:.2f}" if s == season else "" for s, v in zip(order, st_["goals"])], textposition="outside",
                               hovertemplate="%{x}: %{y:.2f} goals per match<extra></extra>"))
        fig.update_layout(yaxis_title="Goals per match", yaxis_range=[0, st_["goals"].max() * 1.15])
        t.chart(fig, 380, key="goals")
    st.caption("Source: openfootball/football.json (CC0). Club crests are trademarks of their clubs, used only to identify teams.")
