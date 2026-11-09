# Data

- **Source:** https://github.com/openfootball/football.json, file `en.1.json` of each season folder (2014-15 to 2025-26).
- **License:** CC0 1.0 (public domain dedication), so the raw files may be redistributed here.
- `raw/` one JSON per season exactly as downloaded (snapshot taken 2026-11-09).
- `processed/matches.csv` cleaned matches (club names harmonised, placeholder scores marked as not played).
- `processed/features.csv`, `elo_history.csv`, `test_predictions.csv` outputs of the pipeline.
- **2025-26** is incomplete in this snapshot (353 of 380 matches); it is used only as a partial check and for the "current Elo" in the dashboard.
- Club-name cleaning: leading "AFC " and trailing " FC"/" AFC" removed (e.g. "Sunderland AFC" becomes "Sunderland").
- Not included: shots, xG, lineups, injuries, odds.
