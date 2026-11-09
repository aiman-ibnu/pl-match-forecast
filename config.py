"""Central configuration for the Premier League match-result forecasting project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# --- Data source: openfootball/football.json (CC0 / public domain) -------------
RAW_BASE_URL = "https://raw.githubusercontent.com/openfootball/football.json/master"
LEAGUE_FILE = "en.1.json"  # English Premier League
SEASONS = ["2014-15", "2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
           "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
MATCHES_PER_SEASON = 380

MATCHES_FILE = PROCESSED_DIR / "matches.csv"
FEATURES_FILE = PROCESSED_DIR / "features.csv"
RATINGS_FILE = PROCESSED_DIR / "elo_history.csv"
PREDICTIONS_FILE = PROCESSED_DIR / "test_predictions.csv"
METRICS_FILE = REPORTS_DIR / "model_metrics.csv"
MODEL_FILE = MODELS_DIR / "best_model.joblib"
MODEL_META_FILE = MODELS_DIR / "best_model_meta.json"

# --- Modelling -------------------------------------------------------------
RANDOM_STATE = 42
N_TEST_SEASONS = 2      # last complete seasons are held out
FORM_WINDOW = 5         # matches used for "recent form"
ELO_START = 1500.0
ELO_K = 20.0
ELO_HOME_ADV = 65.0     # Elo points added to the home side when computing expectations
ELO_SEASON_REGRESSION = 0.25  # share of the gap to 1500 recovered between seasons
CLASSES = ["A", "D", "H"]     # away win, draw, home win (alphabetical, used for probabilities)

ELO_NEWCOMER = 1420.0   # rating of a club seen for the first time after the first season (promoted clubs)
WARMUP_SEASONS = 1      # first season only warms up the Elo ratings and form windows; it is not used for training
