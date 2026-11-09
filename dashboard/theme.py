"""Visual theme: dark surface with copper accents, a Plotly template, and small HTML building blocks."""
import html

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from dashboard import logos

# --- tokens ---------------------------------------------------------------
BG, PANEL, PANEL_2, LINE = "#0E0C0B", "#15110F", "#1D1814", "#2C241E"
INK, INK_2, MUTED = "#EDE6DF", "#B8ACA0", "#7F7469"
COPPER, COPPER_HI, COPPER_LO = "#D2733A", "#EBA673", "#6B4A36"
WIN, DRAW, LOSS = "#199E70", "#8C8580", "#E66767"   # form pills, always shown with a W/D/L letter

# Match outcome: validated against the dark surface; always shown together with text labels
OUTCOME = {"H": "#D2733A", "D": "#9085E9", "A": "#199E70"}
# Club lines: fixed order, never cycled (selection is capped at 8)
SERIES = ["#3987E5", "#D2733A", "#199E70", "#C98500", "#D55181", "#008300", "#9085E9", "#E66767"]
# Sequential copper ramp for magnitude: near-surface (low) -> bright (high)
SEQ = [[0, "#1B1411"], [0.15, "#3A2419"], [0.35, "#6E3B1F"], [0.6, "#B05E2C"], [0.8, "#E08A4E"], [1, "#F6D3AF"]]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
html, body, [data-testid="stAppViewContainer"] {{ font-family: 'Inter', system-ui, sans-serif; }}
[data-testid="stAppViewContainer"] {{
  background: radial-gradient(1100px 480px at 75% -12%, rgba(210,115,58,.17), transparent 62%), {BG};
}}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #130F0D, #0B0908); border-right: 1px solid {LINE}; }}
.block-container {{ padding-top: 2.4rem; max-width: 1400px; }}

.hero {{ border-bottom:1px solid {LINE}; padding-bottom:14px; margin-bottom:4px; }}
.hero .kicker {{ color:{COPPER_HI}; font-size:12px; letter-spacing:.18em; text-transform:uppercase; }}
.hero .title {{ font-size:34px; margin-top:4px; font-weight:300; color:{INK}; letter-spacing:-.01em; }}
.hero p {{ color:{INK_2}; margin:6px 0 0 0; max-width:780px; font-size:14px; }}

.sec {{ margin:22px 0 4px 0; }}
.sec .t {{ color:{INK}; font-size:17px; font-weight:500; }}
.sec .s {{ color:{MUTED}; font-size:13px; margin-top:3px; max-width:900px; }}
.sec::after {{ content:""; display:block; height:1px; margin-top:10px;
  background: linear-gradient(90deg, {COPPER} 0, {COPPER} 48px, {LINE} 48px, {LINE} 100%); }}

.kpis {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap:12px; }}
.kpi {{ background: linear-gradient(180deg, {PANEL_2}, {PANEL}); border:1px solid {LINE}; border-radius:6px;
  padding:14px 16px 12px 16px; min-height:138px; }}
.kpi .top {{ display:flex; justify-content:space-between; align-items:flex-start; gap:8px; }}
.kpi .v {{ font-size:34px; font-weight:300; color:{INK}; line-height:1; font-variant-numeric: tabular-nums; }}
.kpi .u {{ font-size:14px; color:{INK_2}; margin-left:3px; }}
.kpi .d {{ font-size:12px; color:{INK_2}; border-left:2px solid {COPPER}; padding-left:6px; white-space:nowrap; }}
.kpi .l {{ font-size:13px; color:{INK_2}; margin-top:8px; }}
.kpi .n {{ font-size:11.5px; color:{MUTED}; margin-top:2px; }}
.spark {{ display:flex; align-items:stretch; gap:3px; height:34px; margin-top:12px; }}
.spark .b {{ flex:1; display:flex; align-items:flex-end; cursor:default; }}
.spark .b i {{ display:block; width:100%; border-radius:1px 1px 0 0; }}
.spark .b:hover i {{ filter:brightness(1.5); }}
.kpi .lead {{ display:flex; align-items:center; gap:10px; }}

.panel {{ background: linear-gradient(180deg, {PANEL_2}, {PANEL}); border:1px solid {LINE}; border-radius:6px; padding:16px 18px; }}
.crest {{ display:inline-block; vertical-align:middle; }}
.pill {{ display:inline-flex; width:20px; height:20px; border-radius:3px; align-items:center; justify-content:center;
  font-size:11px; font-weight:600; color:#0E0C0B; margin-right:3px; }}

.vs {{ display:grid; grid-template-columns: 1fr auto 1fr; gap:14px; align-items:center; }}
.vs .side {{ display:flex; flex-direction:column; align-items:center; text-align:center; gap:6px; }}
.vs .name {{ font-size:20px; color:{INK}; }}
.vs .tag {{ font-size:11px; letter-spacing:.16em; color:{COPPER_HI}; text-transform:uppercase; }}
.vs .mid {{ font-size:13px; color:{MUTED}; letter-spacing:.2em; }}
.stats {{ display:flex; gap:18px; justify-content:center; font-size:12px; color:{MUTED}; }}
.stats b {{ display:block; color:{INK}; font-size:17px; font-weight:400; font-variant-numeric: tabular-nums; }}

.prob {{ display:flex; height:48px; border-radius:4px; overflow:hidden; gap:2px; margin-top:10px; }}
.prob div {{ display:flex; align-items:center; justify-content:center; font-size:14px; color:#0E0C0B; font-weight:600;
  min-width:0; white-space:nowrap; overflow:hidden; }}
.legend {{ display:flex; gap:16px; flex-wrap:wrap; font-size:12px; color:{INK_2}; margin-top:8px; }}
.legend i {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; vertical-align:-1px; }}
.big3 {{ display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-top:14px; }}
.big3 div {{ border-left:2px solid; padding-left:10px; }}
.big3 b {{ display:block; font-size:30px; font-weight:300; color:{INK}; font-variant-numeric: tabular-nums; }}
.big3 span {{ font-size:12px; color:{INK_2}; }}

.h2h {{ width:100%; border-collapse:collapse; font-size:13px; }}
.h2h td {{ padding:7px 6px; border-bottom:1px solid {LINE}; color:{INK_2}; }}
.h2h td.sc {{ text-align:center; color:{INK}; font-variant-numeric: tabular-nums; white-space:nowrap; }}
.h2h td.r {{ text-align:right; }}

.brand {{ display:flex; align-items:center; gap:10px; margin:2px 0 10px 0; }}
.brand .b1 {{ font-size:16px; color:{INK}; line-height:1.2; }}
.brand .b2 {{ font-size:10.5px; color:{COPPER_HI}; letter-spacing:.16em; text-transform:uppercase; }}
.strip {{ display:flex; flex-wrap:wrap; gap:4px; }}
div[data-testid="stDataFrame"] {{ border:1px solid {LINE}; border-radius:6px; }}
.note {{ color:{MUTED}; font-size:12.5px; margin-top:6px; }}
</style>
"""


def setup():
    st.html(CSS)


# --- plotly ---------------------------------------------------------------
pio.templates["copper"] = go.layout.Template(layout=dict(
    font=dict(family="Inter, system-ui, sans-serif", color=INK_2, size=12),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=SERIES, margin=dict(l=10, r=10, t=36, b=10),
    xaxis=dict(gridcolor="rgba(255,255,255,.05)", linecolor=LINE, zeroline=False, tickcolor=LINE, title_font_color=MUTED),
    yaxis=dict(gridcolor="rgba(255,255,255,.06)", linecolor="rgba(0,0,0,0)", zeroline=False, title_font_color=MUTED),
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=INK_2)),
    hoverlabel=dict(bgcolor=PANEL_2, bordercolor=COPPER, font=dict(color=INK, family="Inter")),
    hovermode="closest", bargap=0.25,
))
pio.templates.default = "copper"
PLOT_CFG = {"displayModeBar": False}


def chart(fig, height: int = 360, key: str | None = None):
    fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", config=PLOT_CFG, key=key)


def add_crests(fig, points, size_x, size_y, xref="x", yref="y"):
    """Place a crest image at each (team, x, y)."""
    for team, x, y in points:
        fig.add_layout_image(source=logos.uri(team), x=x, y=y, xref=xref, yref=yref, sizex=size_x, sizey=size_y,
                             xanchor="center", yanchor="middle", sizing="contain", layer="above")


# --- html blocks ----------------------------------------------------------
def esc(s) -> str:
    return html.escape(str(s))


def hero(kicker: str, title: str, sub: str = ""):
    p = f"<p>{esc(sub)}</p>" if sub else ""
    st.html(f'<div class="hero"><div class="kicker">{esc(kicker)}</div><div class="title">{esc(title)}</div>{p}</div>')


def section(title: str, sub: str = ""):
    s = f'<div class="s">{esc(sub)}</div>' if sub else ""
    st.html(f'<div class="sec"><div class="t">{esc(title)}</div>{s}</div>')


def crest(team: str, size: int = 28) -> str:
    return (f'<img class="crest" src="{logos.uri(team)}" width="{size}" height="{size}" '
            f'alt="{esc(team)}" title="{esc(team)}">')


def spark_bars(values, labels=None, highlight: int | None = None, fmt: str = "{:.2f}") -> str:
    """Thin-bar sparkline (SVG). Hover a bar to see its value; the highlighted bar is bright copper.

    The baseline is lifted below the minimum so small season-to-season changes stay visible;
    exact values are in the tooltip and in the charts below.
    """
    vals = [float(v) for v in values]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    base = lo - span * 0.6
    # Built from divs (Streamlit's HTML sanitizer strips inline <svg>); each bar keeps a hover title.
    out = []
    for i, v in enumerate(vals):
        pct = max(6.0, (v - base) / (hi - base) * 100)
        col = COPPER_HI if i == highlight else COPPER_LO
        lab = f"{labels[i]}: " if labels is not None else ""
        out.append(f'<div class="b" title="{esc(lab)}{fmt.format(v)}"><i style="height:{pct:.1f}%;background:{col}"></i></div>')
    return f'<div class="spark">{"".join(out)}</div>'


def kpi(value, label: str, unit: str = "", delta: str = "", note: str = "", spark: str = "", lead: str = "") -> str:
    d = f'<span class="d">{esc(delta)}</span>' if delta else ""
    n = f'<div class="n">{esc(note)}</div>' if note else ""
    return (f'<div class="kpi"><div class="top"><div class="lead">{lead}<div><span class="v">{esc(value)}</span>'
            f'<span class="u">{esc(unit)}</span></div></div>{d}</div><div class="l">{esc(label)}</div>{n}{spark}</div>')


def kpi_row(cards: list[str]):
    st.html(f'<div class="kpis">{"".join(cards)}</div>')


FORM_NAME = {"W": "Win", "D": "Draw", "L": "Loss"}


def form_pills(results: list[str]) -> str:
    col = {"W": WIN, "D": DRAW, "L": LOSS}
    return "".join(f'<span class="pill" style="background:{col[r]}" title="{FORM_NAME[r]}">{r}</span>' for r in results)


def legend_html(items: list[tuple[str, str]]) -> str:
    return '<div class="legend">' + "".join(f'<span><i style="background:{c}"></i>{esc(t)}</span>' for t, c in items) + "</div>"
