# Premier League Match Forecast

Predict home win / draw / away win for English Premier League matches from Elo ratings and recent form, compare against honest baselines, and explore everything in an interactive Streamlit dashboard. UTS project for Metode Visualisasi Data.

**Headline result (test = 2023-24 and 2024-25, 754 matches):**

| Model | Accuracy | Log loss |
|---|---|---|
| Always home win | 0.434 | 1.068 |
| Elo only | 0.557 | 0.962 |
| Logistic regression (selected) | 0.558 | 0.963 |

The model clearly beats "always home", but is **not better than Elo alone** (difference within noise), and it never predicts draws. See `reports/project_description.md`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
The repository already contains the data snapshot (`data/raw`, `data/processed`) and the trained model, so the dashboard runs immediately.

## Re-run the whole pipeline

```bash
python -m src.data_collection     # optional: re-download (2025-26 will have more matches, numbers will change)
python -m src.data_processing     # clean + validate -> data/processed/matches.csv
python -m src.features            # leakage-free features + Elo history
python -m src.model_training      # CV, models, metrics, bootstrap -> reports/, models/ (~30 s)
python scripts/build_notebooks.py # regenerate and execute notebooks
python -m pytest tests -q         # 12 tests: integrity, leakage, metrics, dashboard smoke tests
```

## Structure
```
app.py, dashboard/      Streamlit app (Overview, Team Elo & Form, Prediction, Insights); theme.py = dark/copper style
assets/logos/           club crests (python -m scripts.fetch_logos; initials badge if missing)
src/                    data_collection, data_processing, features, evaluation, model_training
notebooks/              01 data, 02 EDA, 03 modelling (executed)
reports/                project_description.md, metrics CSVs, figures/
tests/                  pipeline + app tests
data/                   raw (CC0 snapshot), processed, README.md
```

## How it avoids leakage
Features of a match use only earlier matches (state is updated after writing each row; a test changes one result and checks that its features and all earlier ones are unchanged). Train/test split is by season in time order; hyper-parameters are chosen by expanding-window CV on training seasons only.

## Limitations
Scores only (no shots, xG, lineups, odds); 2 test seasons; 2025-26 incomplete; no draw predictions; educational, not betting advice.

## Dashboard
Dark theme with copper accents. Every chart is interactive (hover for values; season, club and filter controls update the page). Club crests appear in tables, charts and the match card. Open a page directly with `?page=Prediction` etc.

## Data and license
Data: openfootball/football.json, CC0. See `data/README.md`. Club crests are trademarks of their clubs and are used only to identify teams. Add a `LICENSE` file for the code before publishing (e.g. MIT).
