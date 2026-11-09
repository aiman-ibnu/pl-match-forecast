"""Shared loaders and colours for the dashboard (all read from files produced by the pipeline)."""
import json

import joblib
import pandas as pd
import streamlit as st

import config
from src import data_processing as dp
from src.features import current_state

COLORS = {"H": "#2563eb", "D": "#9ca3af", "A": "#dc2626", "main": "#2563eb", "base": "#9ca3af"}
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
