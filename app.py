import streamlit as st

from dashboard import insights, overview, prediction, teams, theme
from dashboard import common as c

st.set_page_config(page_title="Premier League Match Forecast", page_icon="⚽", layout="wide")
theme.setup()

PAGES = {"Overview": overview, "Team Elo & Form": teams, "Prediction": prediction, "Insights": insights}

with st.sidebar:
    st.html('<div class="brand"><div style="font-size:28px">⚽</div><div><div class="b2">Premier League</div>'
            '<div class="b1">Match Forecast</div></div></div>')
    names = list(PAGES)
    start = st.query_params.get("page", names[0])   # deep link, e.g. ?page=Prediction
    page = st.radio("Page", names, index=names.index(start) if start in names else 0)
    st.query_params["page"] = page
    clubs = c.current_clubs()["club"].tolist()
    st.html('<div class="note" style="margin-top:14px">Clubs in the latest season, by Elo</div>'
            f'<div class="strip" style="margin-top:6px">{"".join(theme.crest(t, 26) for t in clubs)}</div>')
    st.caption("Predicts home / draw / away probabilities from Elo and recent form. "
               "Data: openfootball (CC0). Learning project, not betting advice.")

PAGES[page].render()
