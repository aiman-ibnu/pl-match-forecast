"""Shared loaders and colours for the dashboard (all read from files produced by the pipeline)."""
import json

import joblib
import pandas as pd
import streamlit as st

import config
from src import data_processing as dp
from src.features import current_state

from dashboard import theme

COLORS = {**theme.OUTCOME, "main": theme.COPPER, "base": theme.MUTED}
LABELS = {"H": "Home win", "D": "Draw", "A": "Away win"}


@st.cache_data
def matches() -> pd.DataFrame:
    return dp.load()


@st.cache_data
def elo_history() -> pd.DataFrame:
    return pd.read_csv(config.RATINGS_FILE, parse_dates=["date"])


@st.cache_data
def predictions() -> pd.DataFrame:
    return pd.read_csv(config.PREDICTIONS_FILE, parse_dates=["date"])


@st.cache_data
def metrics_table() -> pd.DataFrame:
    return pd.read_csv(config.METRICS_FILE)


@st.cache_data
def bootstrap() -> dict:
    return json.loads((config.REPORTS_DIR / "bootstrap_best_vs_baselines.json").read_text())


@st.cache_data
def importance() -> pd.DataFrame:
    return pd.read_csv(config.REPORTS_DIR / "permutation_importance_test.csv")


@st.cache_data
def ablation() -> pd.DataFrame:
    return pd.read_csv(config.REPORTS_DIR / "ablation_features.csv")


@st.cache_data
def state() -> dict:
    return current_state(matches())


@st.cache_resource
def model():
    return joblib.load(config.MODEL_FILE)


@st.cache_data
def meta() -> dict:
    return json.loads(config.MODEL_META_FILE.read_text())


@st.cache_data
def season_tables() -> dict:
    m = matches()
    return {s: dp.season_table(m, s) for s in sorted(m.loc[m["played"], "season"].unique())}


@st.cache_data
def team_results() -> pd.DataFrame:
    """One row per club per played match: date, season, opponent, venue, goals for/against, W/D/L."""
    p = matches()
    p = p[p["played"]]
    h = pd.DataFrame({"date": p["date"], "season": p["season"], "matchday": p["matchday"], "team": p["home"], "opp": p["away"],
                      "venue": "H", "gf": p["home_goals"], "ga": p["away_goals"]})
    a = pd.DataFrame({"date": p["date"], "season": p["season"], "matchday": p["matchday"], "team": p["away"], "opp": p["home"],
                      "venue": "A", "gf": p["away_goals"], "ga": p["home_goals"]})
    r = pd.concat([h, a], ignore_index=True).sort_values(["date", "team"])
    r["res"] = ["W" if f > g else ("D" if f == g else "L") for f, g in zip(r["gf"], r["ga"])]
    r["pts"] = r["res"].map({"W": 3, "D": 1, "L": 0})
    return r.reset_index(drop=True)


def last_form(team: str, n: int = 5) -> list[str]:
    r = team_results()
    return r.loc[r["team"] == team, "res"].tail(n).tolist()


def current_clubs() -> pd.DataFrame:
    """Clubs of the latest season with Elo, form and Elo rank (1 = strongest)."""
    cur = pd.DataFrame(state()).T.reset_index(names="club")
    cur = cur[cur["last_season"] == cur["last_season"].max()].copy()
    for col in ["elo", "ppg5", "gf5", "ga5"]:
        cur[col] = cur[col].astype(float)
    cur = cur.sort_values("elo", ascending=False).reset_index(drop=True)
    cur["rank"] = range(1, len(cur) + 1)
    return cur
