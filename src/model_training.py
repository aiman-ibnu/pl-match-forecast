"""Train and compare match-result models with out-of-time validation.

Split: seasons 2015-16..2022-23 train (2014-15 only warms up Elo/form), 2023-24 and 2024-25 test,
unfinished 2025-26 is a second, separate out-of-time check. Hyper-parameters are tuned by an
expanding-window CV over the *training* seasons only; the "best" model is the lowest CV log loss,
never chosen on test results.

Run: python -m src.model_training
"""
from __future__ import annotations

import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from src import evaluation as ev
from src.features import FEATURES

warnings.filterwarnings("ignore")
ELO_ONLY = ["elo_diff"]
FORM_ONLY = ["ppg5_home", "ppg5_away", "gf5_home", "gf5_away", "ga5_home", "ga5_away"]
ELO_SET = ["elo_diff", "elo_home", "elo_away"]


def make(name: str, params: dict | None = None):
    p = params or {}
    rs = config.RANDOM_STATE
    if name in ("Logistic Regression", "Baseline: Elo only (logistic)"):
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=rs, **p))
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=300, random_state=rs, n_jobs=2, **p)
    if name == "Gradient Boosting":
        return HistGradientBoostingClassifier(random_state=rs, **p)
    raise ValueError(name)


GRIDS = {
    "Logistic Regression": {"C": [0.01, 0.1, 1.0]},
    "Random Forest": {"max_depth": [3, 5, 8], "min_samples_leaf": [10, 30, 60]},
    "Gradient Boosting": {"max_depth": [2, 3], "learning_rate": [0.03, 0.1], "max_iter": [100, 200], "l2_regularization": [1.0]},
}


def proba(model, X) -> np.ndarray:
    p = model.predict_proba(X)
    order = [list(model.classes_).index(c) for c in ev.CLASSES]
    return p[:, order]


def cv_logloss(name, params, train: pd.DataFrame, cols, n_val: int = 4) -> float:
    seasons = sorted(train["season"].unique())
    losses, ys, ps = [], [], []
    for s in seasons[-n_val:]:
        tr, va = train[train["season"] < s], train[train["season"] == s]
        m = make(name, params).fit(tr[cols], tr["result"])
        ys.append(va["result"].to_numpy()); ps.append(proba(m, va[cols]))
    return ev.metrics(np.concatenate(ys), np.vstack(ps))["log_loss"]


def run() -> dict:
    t0 = time.time()
    feats = pd.read_csv(config.FEATURES_FILE, parse_dates=["date"])
    n_all = len(feats)
    feats = feats.dropna(subset=FEATURES).reset_index(drop=True)
    train, test, partial = ev.split_by_season(feats)
    print(f"matches with complete features: {len(feats)}/{n_all} | train {len(train)} ({train.season.min()}..{train.season.max()}) "
          f"| test {len(test)} | partial {len(partial)}")
    rows, probs = [], {}
    sets = {"test": test, "partial_2025_26": partial}

    # ---- baselines ----
    freq = train["result"].value_counts(normalize=True).reindex(ev.CLASSES).to_numpy()
    base_home = {k: np.tile(np.array([0.0, 0.0, 1.0]) * 0.0 + freq, (len(d), 1)) for k, d in sets.items()}
    for k, d in sets.items():
        m = ev.metrics(d["result"], base_home[k], pred=np.array(["H"] * len(d)))
        rows.append({"model": "Baseline: always home win", "type": "baseline", "set": k, "cv_log_loss": np.nan, "params": "{}", **m})
        probs[("Baseline: always home win", k)] = base_home[k]
    name = "Baseline: Elo only (logistic)"
    cv = cv_logloss(name, {}, train, ELO_ONLY)
    elo_model = make(name).fit(train[ELO_ONLY], train["result"])
    for k, d in sets.items():
        p = proba(elo_model, d[ELO_ONLY]); probs[(name, k)] = p
        rows.append({"model": name, "type": "baseline", "set": k, "cv_log_loss": cv, "params": "{}", **ev.metrics(d["result"], p)})

    # ---- ML models ----
    fitted, best_params = {}, {}
    for name, grid in GRIDS.items():
        t = time.time()
        scores = [(cv_logloss(name, p, train, FEATURES), p) for p in ParameterGrid(grid)]
        cvs, bp = min(scores, key=lambda x: x[0])
        model = make(name, bp).fit(train[FEATURES], train["result"])
        fitted[name], best_params[name] = model, bp
        for k, d in sets.items():
            p = proba(model, d[FEATURES]); probs[(name, k)] = p
            rows.append({"model": name, "type": "ml", "set": k, "cv_log_loss": cvs, "params": json.dumps(bp), **ev.metrics(d["result"], p)})
        print(f"[{name}] cv_logloss={cvs:.4f} ({time.time() - t:.0f}s) params={bp}", flush=True)

    res = pd.DataFrame(rows)
    ml = res[(res["type"] == "ml") & (res["set"] == "test")].sort_values("cv_log_loss")
    best = ml.iloc[0]["model"]
    res["selected_by_cv"] = res["model"] == best
    config.REPORTS_DIR.mkdir(exist_ok=True)
    res.to_csv(config.METRICS_FILE, index=False)

    # ---- uncertainty: best vs baselines on the test seasons ----
    boot = {"best_model": best,
            "vs_always_home": ev.bootstrap_diff(test["result"], probs[(best, "test")], probs[("Baseline: always home win", "test")]),
            "vs_elo_only": ev.bootstrap_diff(test["result"], probs[(best, "test")], probs[("Baseline: Elo only (logistic)", "test")])}
    (config.REPORTS_DIR / "bootstrap_best_vs_baselines.json").write_text(json.dumps(boot, indent=2))

    # ---- feature-set ablation (best model type, tuned params) ----
    abl = []
    for label, cols in [("Elo difference only", ELO_ONLY), ("Elo (home, away, diff)", ELO_SET), ("Form only (last 5)", FORM_ONLY),
                        ("Elo + form", ELO_SET + FORM_ONLY), ("All features (+ rest days)", FEATURES)]:
        m = make(best, best_params[best]).fit(train[cols], train["result"])
        abl.append({"features": label, "n": len(cols), **ev.metrics(test["result"], proba(m, test[cols]))})
    pd.DataFrame(abl).to_csv(config.REPORTS_DIR / "ablation_features.csv", index=False)

    # ---- permutation importance (test, log loss) ----
    pi = permutation_importance(fitted[best], test[FEATURES], test["result"], scoring="neg_log_loss", n_repeats=10,
                                random_state=config.RANDOM_STATE, n_jobs=1)
    pd.DataFrame({"feature": FEATURES, "logloss_increase": pi.importances_mean, "std": pi.importances_std}).sort_values(
        "logloss_increase", ascending=False).to_csv(config.REPORTS_DIR / "permutation_importance_test.csv", index=False)

    # ---- predictions of the best model (test + partial) for the dashboard ----
    out = []
    for k, d in sets.items():
        p = probs[(best, k)]
        o = d[["match_id", "season", "date", "matchday", "home", "away", "home_goals", "away_goals", "result", "elo_diff"]].copy()
        o[["pA", "pD", "pH"]] = p
        o["pred"] = np.array(ev.CLASSES)[p.argmax(1)]
        o["set"] = k
        o["model_name"] = best
        out.append(o)
    pd.concat(out).to_csv(config.PREDICTIONS_FILE, index=False)

    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(fitted[best], config.MODEL_FILE)
    config.MODEL_META_FILE.write_text(json.dumps({
        "model": best, "params": best_params[best], "features": FEATURES, "classes": ev.CLASSES,
        "train_seasons": sorted(train["season"].unique()), "test_seasons": sorted(test["season"].unique()),
        "partial_season": sorted(partial["season"].unique()), "n_train": len(train), "n_test": len(test), "n_partial": len(partial),
        "class_share_train": dict(zip(ev.CLASSES, map(float, freq)))}, indent=2))
    print(f"\nDone in {time.time() - t0:.0f}s. Selected by CV: {best}")
    show = res[res["set"] == "test"].drop(columns=["params", "set"])
    print(show.round(4).to_string(index=False))
    print(res[res["set"] == "partial_2025_26"].drop(columns=["params", "set", "cv_log_loss", "selected_by_cv"]).round(4).to_string(index=False))
    return {"best": best}


if __name__ == "__main__":
    run()
