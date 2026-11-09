"""Metrics for three-way match-result probabilities (classes ordered A < D < H)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

import config

CLASSES = config.CLASSES


def onehot(y) -> np.ndarray:
    y = np.asarray(y)
    return np.stack([(y == c).astype(float) for c in CLASSES], axis=1)


def rps(y, proba) -> float:
    """Ranked probability score (lower is better); respects the A < D < H ordering."""
    p, o = np.cumsum(np.asarray(proba), axis=1), np.cumsum(onehot(y), axis=1)
    return float(np.mean(0.5 * ((p[:, :2] - o[:, :2]) ** 2).sum(axis=1)))


def brier(y, proba) -> float:
    return float(np.mean(((np.asarray(proba) - onehot(y)) ** 2).sum(axis=1)))


def metrics(y, proba, pred=None) -> dict:
    proba = np.clip(np.asarray(proba, float), 1e-6, 1)
    proba = proba / proba.sum(axis=1, keepdims=True)
    pred = np.array(CLASSES)[proba.argmax(axis=1)] if pred is None else np.asarray(pred)
    return {"accuracy": accuracy_score(y, pred), "log_loss": log_loss(y, proba, labels=CLASSES),
            "RPS": rps(y, proba), "Brier": brier(y, proba)}


def split_by_season(feats: pd.DataFrame, n_test: int = config.N_TEST_SEASONS):
    """Out-of-time split. Returns train, test (last ``n_test`` complete seasons), partial (unfinished season)."""
    seasons = sorted(feats["season"].unique())
    full = [s for s in seasons if bool(feats.loc[feats["season"] == s, "season_complete"].iloc[0])]
    partial = [s for s in seasons if s not in full]
    test_s = full[-n_test:]
    train_s = [s for s in full if s not in test_s][config.WARMUP_SEASONS:]   # drop the Elo warm-up season
    assert max(train_s) < min(test_s)
    return (feats[feats["season"].isin(train_s)].copy(), feats[feats["season"].isin(test_s)].copy(),
            feats[feats["season"].isin(partial)].copy())


def bootstrap_diff(y, proba_a, proba_b, n_boot: int = 2000, seed: int = config.RANDOM_STATE) -> dict:
    """95% CI of (A - B) for accuracy and log loss by resampling matches."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); pa, pb = np.asarray(proba_a), np.asarray(proba_b)
    cls = np.array(CLASSES)
    ca, cb = (cls[pa.argmax(1)] == y).astype(float), (cls[pb.argmax(1)] == y).astype(float)
    idx_y = np.array([CLASSES.index(v) for v in y])
    la = -np.log(np.clip(pa[np.arange(len(y)), idx_y], 1e-6, 1)); lb = -np.log(np.clip(pb[np.arange(len(y)), idx_y], 1e-6, 1))
    n = len(y); acc, ll = [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        acc.append(ca[i].mean() - cb[i].mean()); ll.append(la[i].mean() - lb[i].mean())
    f = lambda a, d: {"diff": float(d), "lo": float(np.percentile(a, 2.5)), "hi": float(np.percentile(a, 97.5))}
    return {"accuracy": f(acc, ca.mean() - cb.mean()), "log_loss": f(ll, la.mean() - lb.mean())}


def calibration_table(y, p_home, bins: int = 10) -> pd.DataFrame:
    d = pd.DataFrame({"p": np.asarray(p_home), "hit": (np.asarray(y) == "H").astype(float)})
    d["bin"] = pd.cut(d["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    g = d.groupby("bin", observed=True).agg(pred=("p", "mean"), actual=("hit", "mean"), n=("p", "size")).reset_index(drop=True)
    return g
