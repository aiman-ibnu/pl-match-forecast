import streamlit as st

from dashboard import insights, overview, prediction, teams

st.set_page_config(page_title="Premier League Match Forecast", layout="wide")
PAGES = {"Overview": overview, "Team Elo & Form": teams, "Prediction": prediction, "Insights": insights}

st.sidebar.title("Premier League Match Forecast")
page = st.sidebar.radio("Page", list(PAGES))
st.sidebar.caption("Predicts home / draw / away probabilities from Elo and recent form. Learning project, not betting advice.")
PAGES[page].render()
