import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import common as c
from dashboard import logos
from dashboard import theme as t


def render():
    t.hero("Team strength", "Elo rating & recent form",
           "Elo goes up after a win and down after a loss; beating a strong team moves it more than beating a weak one. "
           "1500 is an average Premier League club.")

    h = c.elo_history()
    cur = c.current_clubs()
    teams = sorted(h["team"].unique())
    seasons = sorted(h["season"].unique())

    # ---------------- Elo over time ----------------
    f1, f2 = st.columns([3, 2])
    default = [x for x in ["Arsenal", "Liverpool", "Manchester City", "Chelsea"] if x in teams]
    pick = f1.multiselect("Clubs (max 8)", teams, default=default, max_selections=8)
    lo, hi = f2.select_slider("Seasons", options=seasons, value=(seasons[0], seasons[-1]))

    t.section("Elo rating over time", "Each line is one club; its crest marks where the line ends. Hover for the exact rating on a date.")
    d = h[h["team"].isin(pick) & (h["season"] >= lo) & (h["season"] <= hi)]
    if d.empty:
        st.info("Pick at least one club.")
    else:
        fig = go.Figure()
        fig.add_hline(y=1500, line=dict(color=t.MUTED, width=1, dash="dot"),
                      annotation_text="1500 = average", annotation_font_color=t.MUTED, annotation_position="top left",
                      annotation_bgcolor="rgba(14,12,11,.8)")
        ends = []
        for i, team in enumerate(pick):
            g = d[d["team"] == team]
            if g.empty:
                continue
            fig.add_scatter(x=g["date"], y=g["elo"], mode="lines", name=logos.short(team), line=dict(color=t.SERIES[i], width=2),
                            hovertemplate=f"<b>{team}</b><br>%{{x|%d %b %Y}}<br>Elo %{{y:.0f}}<extra></extra>")
            ends.append((team, g["date"].iloc[-1] + pd.Timedelta(days=45), g["elo"].iloc[-1]))
        span = (d["date"].max() - d["date"].min()).days
        t.add_crests(fig, ends, pd.Timedelta(days=max(30, span * 0.035)).total_seconds() * 1000, 32)
        fig.update_layout(yaxis_title="Elo rating", xaxis_range=[d["date"].min(), d["date"].max() + pd.Timedelta(days=max(60, span * 0.05))],
                          hovermode="x unified")
        t.chart(fig, 440, key="elo")

    # ---------------- Power ranking with crests ----------------
    left, right = st.columns([5, 6], gap="large")
    with left:
        t.section(f"Power ranking · {cur['last_season'].iloc[0]}",
                  "Latest Elo for every club in the current season. The crest is the data point; the stem shows the gap to an average club (1500).")
        r = cur.iloc[::-1].reset_index(drop=True)
        y = list(range(len(r)))
        fig = go.Figure()
        for yi, (_, row) in zip(y, r.iterrows()):
            col = t.COPPER if row["elo"] >= 1500 else t.MUTED
            fig.add_scatter(x=[1500, row["elo"]], y=[yi, yi], mode="lines", line=dict(color=col, width=2), hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=r["elo"], y=y, mode="markers", marker=dict(size=24, color="rgba(0,0,0,0)"), showlegend=False,
                        customdata=r[["club", "rank", "ppg5"]].values,
                        hovertemplate="<b>#%{customdata[1]} %{customdata[0]}</b><br>Elo %{x:.0f}<br>Points/game (last 5): %{customdata[2]:.1f}<extra></extra>")
        t.add_crests(fig, [(row["club"], row["elo"], yi) for yi, (_, row) in zip(y, r.iterrows())], 40, 0.9)
        fig.add_vline(x=1500, line=dict(color=t.MUTED, width=1, dash="dot"))
        pad = 40
        fig.update_layout(xaxis_title="Elo rating", xaxis_range=[min(r["elo"].min(), 1500) - pad, r["elo"].max() + pad],
                          yaxis=dict(tickvals=y, ticktext=[f"{logos.short(x)}  " for x in r["club"]], showgrid=False, range=[-0.8, len(r) - 0.2]),
                          margin=dict(l=10, r=20, t=10, b=10))
        t.chart(fig, 30 * len(r) + 60, key="rank")

    # ---------------- Club spotlight ----------------
    with right:
        t.section("Club spotlight", "Pick a club to see its current shape and how it finished each season.")
        clubs = cur["club"].tolist()
        team = st.selectbox("Club", clubs, index=0, key="spot", format_func=logos.short)
        row = cur.set_index("club").loc[team]
        form = c.last_form(team, 10)
        st.html(f"""
        <div class="panel" style="display:flex;gap:18px;align-items:center;flex-wrap:wrap">
          {t.crest(team, 84)}
          <div style="flex:1;min-width:220px">
            <div class="kicker" style="color:{t.COPPER_HI};font-size:11px;letter-spacing:.16em;text-transform:uppercase">Elo rank #{int(row['rank'])} of {len(cur)}</div>
            <div style="font-size:24px;color:{t.INK};font-weight:300;margin:2px 0 8px">{t.esc(team)}</div>
            <div class="stats" style="justify-content:flex-start">
              <div><b>{row['elo']:.0f}</b>Elo</div><div><b>{row['ppg5']:.1f}</b>Pts/game (5)</div>
              <div><b>{row['gf5']:.1f}</b>Scored/game</div><div><b>{row['ga5']:.1f}</b>Conceded/game</div>
            </div>
            <div style="margin-top:10px;font-size:12px;color:{t.MUTED}">Last {len(form)} (oldest → newest)</div>
            <div style="margin-top:4px">{t.form_pills(form)}</div>
          </div>
        </div>""")

        tables = c.season_tables()
        fin = [(s, int(tb.loc[tb["team"] == team, "pos"].iloc[0]), int(tb.loc[tb["team"] == team, "pts"].iloc[0]))
               for s, tb in tables.items() if team in set(tb["team"])]
        if fin:
            fd = pd.DataFrame(fin, columns=["season", "pos", "pts"])
            fig = go.Figure()
            fig.add_hrect(y0=0.5, y1=4.5, fillcolor=t.COPPER, opacity=0.08, line_width=0,
                          annotation_text="Top 4", annotation_font_color=t.MUTED, annotation_position="top left")
            fig.add_hrect(y0=17.5, y1=20.5, fillcolor=t.LOSS, opacity=0.07, line_width=0,
                          annotation_text="Relegation", annotation_font_color=t.MUTED, annotation_position="bottom left")
            fig.add_scatter(x=fd["season"], y=fd["pos"], mode="lines+markers+text", line=dict(color=t.COPPER, width=2),
                            marker=dict(size=9, color=t.COPPER, line=dict(color=t.BG, width=2)),
                            text=fd["pos"], textposition="top center", textfont=dict(color=t.INK_2, size=11), customdata=fd["pts"],
                            hovertemplate="%{x}: finished %{y} (%{customdata} pts)<extra></extra>", showlegend=False)
            fig.update_layout(yaxis=dict(autorange=False, range=[20.5, 0.5], title="League position", tickvals=[1, 4, 8, 12, 17, 20]),
                              xaxis=dict(categoryorder="array", categoryarray=sorted(tables)), margin=dict(t=20))
            t.chart(fig, 300, key="finish")
            st.caption("The latest season is still in progress, so its position is provisional.")

    # ---------------- Form table ----------------
    t.section("Current form (last 5 matches)", f"Clubs in season {cur['last_season'].iloc[0]}, values after their last recorded match.")
    tb = cur.copy()
    tb["crest"] = tb["club"].map(logos.uri)
    tb["form"] = tb["club"].map(lambda x: "".join(c.last_form(x, 5)))
    st.dataframe(
        tb[["rank", "crest", "club", "elo", "ppg5", "gf5", "ga5", "form", "last_date"]],
        hide_index=True, width="stretch", height=36 * len(tb) + 40,
        column_config={
            "rank": st.column_config.NumberColumn("#", width=40),
            "crest": st.column_config.ImageColumn("", width=40),
            "club": "Club",
            "elo": st.column_config.ProgressColumn("Elo", format="%.0f", min_value=1300, max_value=float(max(1800, tb["elo"].max()))),
            "ppg5": st.column_config.NumberColumn("Points/game", format="%.1f"),
            "gf5": st.column_config.NumberColumn("Scored/game", format="%.1f"),
            "ga5": st.column_config.NumberColumn("Conceded/game", format="%.1f"),
            "form": st.column_config.TextColumn("Last 5", help="Oldest to newest. W win, D draw, L loss."),
            "last_date": "Last match",
        })
