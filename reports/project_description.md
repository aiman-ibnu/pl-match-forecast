# Project Description

## 1. Project Title
Premier League Match Result Forecast: Elo, Recent Form and an Honest Baseline Comparison (UTS Metode Visualisasi Data)

## 2. Abstract
This project predicts the result of English Premier League matches (home win, draw, away win) from information available before kickoff: an Elo team rating, form over the last 5 matches, and rest days. Data are 12 seasons (2014-15 to 2025-26) of the CC0 openfootball dataset, 4,533 played matches. Models are evaluated out-of-time on the 2023-24 and 2024-25 seasons (754 matches). The selected logistic regression reaches 55.8% accuracy and log loss 0.963 against 43.4% and 1.068 for always predicting a home win. It is not better than a one-feature Elo-only baseline (55.7%, 0.962; the difference is within noise). The main finding is therefore that Elo already contains nearly all usable signal, and that draws are not predictable from scores alone. An interactive Streamlit dashboard presents the data, ratings, predictions and findings.

## 3. Background
Football is low-scoring, so results contain a lot of luck. Rating systems such as Elo are a standard, easy-to-explain way to express team strength. Bookmakers and analysts also talk about "form" and "home advantage". This project checks, with data, how much of that is actually useful for prediction.

## 4. Problem Statement
Given only past results, how well can we predict the outcome of a Premier League match, and does anything beyond a simple team-strength rating (recent form, rest days, a more flexible model) improve prediction?

## 5. Objectives
1. Build a clean, reproducible match dataset from an openly licensed source.
2. Build leakage-free pre-match features.
3. Compare simple baselines with three machine-learning models using a time-based split.
4. Report uncertainty (bootstrap intervals) and limitations honestly.
5. Deliver an interactive visual dashboard that explains the data and the result.

## 6. Dataset Description
Source files: `openfootball/football.json`, `en.1.json` for each season. One row per match: season, matchday, date, home and away club, full-time and half-time score. 4,560 fixtures, of which 4,533 have a real final score (2025-26 is incomplete: 353 of 380 matches recorded in the stored snapshot). 35 distinct clubs. Only scores are available: there are no shots, expected goals, lineups, injuries or odds.

## 7. Data Sources and Licenses
openfootball/football.json is released under CC0 (public domain dedication), so the raw files are included in `data/raw/`. No data were scraped from commercial or restricted sites; football-data.co.uk, FBref and similar sources were deliberately not used. The code is the student's own work; add a LICENSE of your choice (e.g. MIT) when publishing.

## 8. Data Preprocessing
- Club names harmonised (e.g. "Arsenal FC" and "Arsenal" become "Arsenal"; "AFC Bournemouth" becomes "Bournemouth").
- Fixtures with a placeholder score (0-0 without a real result) are marked as not played and excluded.
- Validation checks: 20 clubs per season, no duplicate fixture, 380 matches per complete season, 38 per club.
- Result label from full-time score: H, D, A.

## 9. Exploratory Data Analysis
See `notebooks/02_eda.ipynb` and the Overview page. Home wins are the most common result in all 12 seasons (37.9% to 49.2% by season). 2020-21 (matches mostly without fans) has the lowest home-win share (37.9%) and the highest away-win share (40.3%). Goals per match range from 2.57 (2014-15) to 3.28 (2023-24). Home-win share rises from 17% to 74% across eight equal-size Elo-difference groups, while draws only vary between about 17% and 27%.

## 10. Feature Engineering
All features for a match use only matches played strictly before it (state is updated after the features are written).
- `elo_home`, `elo_away`, `elo_diff`: Elo with K=20, home advantage 65 points in the expected-score formula, 25% regression to the mean between seasons, 1420 for newly promoted clubs, 1500 initial. 2014-15 is used only as Elo warm-up.
- `ppg5_*`, `gf5_*`, `ga5_*`: points per game, goals for and against over the last 5 matches (at least 3 required).
- `rest_home`, `rest_away`: days since the club's previous match.
Matches without enough history (75) are dropped from modelling.

## 11. Methodology
Out-of-time split by season: train 2015-16 to 2022-23 (3,001 matches), test 2023-24 and 2024-25 (754 matches), and a partial check on 2025-26 (353 matches). Hyper-parameters are selected by expanding-window cross-validation over the last four training seasons (train on earlier seasons, validate on the next); the model with the lowest CV log loss is selected. The test seasons are used once for reporting. Elo ratings are updated chronologically through the test period (they are features, not fitted parameters); the classifier itself is fitted on training seasons only.

## 12. Machine Learning Models
Baselines: (a) always predict a home win, with class frequencies from training as probabilities; (b) logistic regression on the Elo difference alone. Models: logistic regression (C grid), random forest (shallow, large leaves), histogram gradient boosting (shallow, regularised). Selected by CV: logistic regression, C = 0.01.

## 13. Model Evaluation
Metrics: accuracy, log loss (quality of probabilities), ranked probability score (RPS, respects the order away < draw < home), Brier score. Match-level bootstrap (2,000 resamples) for differences between models. Also a calibration table, a feature ablation and permutation importance on the test seasons.

| Model (test, 754 matches) | Accuracy | Log loss | RPS |
|---|---|---|---|
| Always home win | 0.434 | 1.068 | 0.235 |
| Elo only (logistic) | 0.557 | 0.962 | 0.198 |
| Logistic regression (selected) | 0.558 | 0.963 | 0.197 |
| Random forest | 0.554 | 0.977 | 0.202 |
| Gradient boosting | 0.556 | 0.978 | 0.199 |

## 14. Results and Findings
1. Selected model vs always home: accuracy +0.125 (95% CI 0.088 to 0.161), log loss -0.105 (CI -0.131 to -0.078). Clear improvement.
2. Selected model vs Elo-only: accuracy +0.001 (CI -0.011 to +0.013), log loss +0.001 (CI -0.005 to +0.007). No detectable difference.
3. Form alone is weaker (accuracy 0.512, log loss 1.014); adding form and rest days to Elo does not improve results.
4. Permutation importance is dominated by `elo_diff` and `elo_home`.
5. The model never predicts a draw: the draw probability never exceeds about 32% on the test seasons.
6. On the incomplete 2025-26 season (353 matches) the selected model gets 52.1% accuracy (log loss 0.992) versus 45.9% for always home; the same pattern holds, with lower accuracy than on the test seasons.

## 15. Data Visualization
Streamlit + Plotly dashboard, four pages: Overview (KPIs, result shares and goals by season, league tables), Team Elo and Form (rating trajectories, current form), Prediction (home/draw/away probabilities for any pair of current clubs, model comparison, browse test predictions), Insights (ablation, importance, calibration, confusion matrix, limitations). Static figures are in `reports/figures/`.

## 16. Practical Implications
A simple, transparent rating gives most of the achievable accuracy from results alone, so a more complex model is not justified without richer data (shots, xG, lineups). The dashboard shows probabilities rather than single guesses, which is the honest way to present football forecasts.

## 17. Limitations
- Only final scores; no shots, xG, injuries, lineups or bookmaker odds, and no comparison with bookmaker probabilities.
- Two test seasons (754 matches); results could differ in other periods.
- 2025-26 is incomplete in the stored snapshot, so it is a partial check only.
- Draws are not predicted as a class.
- Elo hyper-parameters (K, home advantage, regression) were fixed a priori, not tuned.
- Educational project, not betting advice.

## 18. Conclusion
Elo plus a regularised logistic regression predicts Premier League results substantially better than always picking the home team, but not better than Elo alone. The honest conclusion is that, from scores only, team strength explains almost everything that can be predicted, and richer data would be needed to go further.

## 19. References
- openfootball/football.json, https://github.com/openfootball/football.json (CC0).
- Elo, A. (1978). The Rating of Chessplayers, Past and Present.
- Constantinou, A. and Fenton, N. (2012). Solving the problem of inadequate scoring rules for assessing probabilistic football forecast models. Journal of Quantitative Analysis in Sports (RPS).
- scikit-learn, Streamlit, Plotly documentation.
