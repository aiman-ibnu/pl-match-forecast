"""Pre-match features: Elo ratings, recent form and rest days.

Every feature of a match is computed ONLY from matches played strictly before it
(matches are processed in chronological order and a match is added to the history after
its features were written). No bookmaker odds, no match statistics of the match itself.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque

import numpy as np
import pandas as pd

import config

FEATURES = ["elo_diff", "elo_home", "elo_away", "ppg5_home", "ppg5_away", "gf5_home", "gf5_away",
            "ga5_home", "ga5_away", "rest_home", "rest_away"]


def elo_expected(diff_with_home_adv: float) -> float:
    """Expected score (win = 1, draw = 0.5) of the home side."""
    return 1.0 / (1.0 + 10 ** (-diff_with_home_adv / 400.0))


def build_features(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (features table for played matches, Elo history per club after each match)."""
    elo: dict[str, float] = {}
    last_season: dict[str, str] = {}
    hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=config.FORM_WINDOW))  # (points, gf, ga)
    last_date: dict[str, pd.Timestamp] = {}
    rows, elo_rows = [], []
    first_season = matches["season"].min()
    current_season = None

    for r in matches[matches["played"]].itertuples(index=False):
        if r.season != current_season:                       # new season: pull ratings towards the mean
            for t in list(elo):
                elo[t] -= config.ELO_SEASON_REGRESSION * (elo[t] - config.ELO_START)
            current_season = r.season
        for t in (r.home, r.away):
            if t not in elo:
                elo[t] = config.ELO_START if r.season == first_season else config.ELO_NEWCOMER

        def form(t):
            h = hist[t]
            if len(h) < 3:
                return (np.nan, np.nan, np.nan)
            a = np.array(h, dtype=float)
            return tuple(a.mean(axis=0))

        fh, fa = form(r.home), form(r.away)
        rest = lambda t: min((r.date - last_date[t]).days, 21) if t in last_date else np.nan
        eh, ea = elo[r.home], elo[r.away]
        rows.append({
            "match_id": r.match_id, "season": r.season, "date": r.date, "matchday": r.matchday, "home": r.home, "away": r.away,
            "elo_home": eh, "elo_away": ea, "elo_diff": eh - ea,
            "ppg5_home": fh[0], "gf5_home": fh[1], "ga5_home": fh[2],
            "ppg5_away": fa[0], "gf5_away": fa[1], "ga5_away": fa[2],
            "rest_home": rest(r.home), "rest_away": rest(r.away),
            "home_goals": r.home_goals, "away_goals": r.away_goals, "result": r.result,
        })
        # ---- only now update the state with the result of this match ----
        s_home = {"H": 1.0, "D": 0.5, "A": 0.0}[r.result]
        delta = config.ELO_K * (s_home - elo_expected(eh - ea + config.ELO_HOME_ADV))
        elo[r.home] += delta
        elo[r.away] -= delta
        pts_h = {"H": 3, "D": 1, "A": 0}[r.result]
        pts_a = {"A": 3, "D": 1, "H": 0}[r.result]
        hist[r.home].append((pts_h, r.home_goals, r.away_goals))
        hist[r.away].append((pts_a, r.away_goals, r.home_goals))
        last_date[r.home] = last_date[r.away] = r.date
        elo_rows.append({"date": r.date, "season": r.season, "team": r.home, "elo": elo[r.home]})
        elo_rows.append({"date": r.date, "season": r.season, "team": r.away, "elo": elo[r.away]})

    feats = pd.DataFrame(rows)
    played_per_season = matches[matches["played"]].groupby("season").size()
    feats["season_complete"] = feats["season"].map(played_per_season >= config.MATCHES_PER_SEASON)
    return feats, pd.DataFrame(elo_rows)


def current_state(matches: pd.DataFrame) -> dict:
    """Latest Elo, form and last match date of every club (used by the dashboard what-if)."""
    elo: dict[str, float] = {}
    feats, hist = build_features(matches)
    last = hist.sort_values("date").groupby("team").tail(1).set_index("team")
    out = {}
    played = matches[matches["played"]]
    for t in last.index:
        g = played[(played["home"] == t) | (played["away"] == t)].tail(config.FORM_WINDOW)
        pts, gf, ga = [], [], []
        for r in g.itertuples():
            home = r.home == t
            gfor, gag = (r.home_goals, r.away_goals) if home else (r.away_goals, r.home_goals)
            pts.append(3 if gfor > gag else (1 if gfor == gag else 0)); gf.append(gfor); ga.append(gag)
        out[t] = {"elo": float(last.loc[t, "elo"]), "ppg5": float(np.mean(pts)), "gf5": float(np.mean(gf)), "ga5": float(np.mean(ga)),
                  "last_date": str(last.loc[t, "date"].date()), "last_season": last.loc[t, "season"]}
    return out


def save(feats: pd.DataFrame, elo_hist: pd.DataFrame) -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    feats.to_csv(config.FEATURES_FILE, index=False)
    elo_hist.to_csv(config.RATINGS_FILE, index=False)


if __name__ == "__main__":
    from src import data_processing as dp
    m = dp.load()
    f, e = build_features(m)
    save(f, e)
    print(f.shape, "missing per feature:", f[FEATURES].isna().sum()[lambda s: s > 0].to_dict())
    print(e.sort_values("date").groupby("team").tail(1).sort_values("elo", ascending=False).head(6).to_string(index=False))
