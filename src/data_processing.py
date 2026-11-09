"""Build a tidy, validated match table from the raw openfootball JSON files.

Handles two quirks of the raw data:
* club names are inconsistent between seasons ('Arsenal FC' vs 'Arsenal', 'AFC Bournemouth');
* in the current, unfinished season, unplayed fixtures carry a placeholder score ``[0, 0]``
  instead of a ``{"ft": ...}`` object. Only matches with a real ``ft`` score count as played.
"""
from __future__ import annotations

import json
import re

import pandas as pd

import config


def clean_name(name: str) -> str:
    n = re.sub(r"^AFC\s+", "", name.strip())
    n = re.sub(r"\s+(FC|AFC)$", "", n)
    return n


def load_season(path) -> pd.DataFrame:
    season = path.stem.split("_")[-1]
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for m in payload["matches"]:
        sc = m.get("score")
        played = isinstance(sc, dict) and "ft" in sc
        rows.append({
            "season": season, "matchday": int(re.search(r"\d+", m["round"]).group()),
            "date": m["date"], "time": m.get("time"), "home": clean_name(m["team1"]), "away": clean_name(m["team2"]),
            "home_goals": sc["ft"][0] if played else None, "away_goals": sc["ft"][1] if played else None,
            "ht_home": sc["ht"][0] if played and "ht" in sc else None, "ht_away": sc["ht"][1] if played and "ht" in sc else None,
            "played": played,
        })
    return pd.DataFrame(rows)


def build_matches() -> pd.DataFrame:
    files = sorted(config.RAW_DIR.glob("en.1_*.json"))
    if not files:
        raise FileNotFoundError("No raw data. Run `python -m src.data_collection` first.")
    df = pd.concat([load_season(f) for f in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["time"] = df["time"].fillna("00:00")
    df = df.sort_values(["date", "time", "season", "matchday", "home"]).reset_index(drop=True)
    df["season_start"] = df["season"].str[:4].astype(int)
    df["result"] = None
    p = df["played"]
    df.loc[p, "result"] = (df.loc[p, "home_goals"] - df.loc[p, "away_goals"]).apply(lambda d: "H" if d > 0 else ("A" if d < 0 else "D"))
    df["match_id"] = range(len(df))
    return df


def validate(df: pd.DataFrame) -> dict:
    """Sanity checks that must hold for the data to be usable; raises AssertionError otherwise."""
    out = {}
    for s, g in df.groupby("season"):
        teams = pd.concat([g["home"], g["away"]]).value_counts()
        assert g["home"].nunique() == 20 and g["away"].nunique() == 20, f"{s}: not 20 clubs"
        assert not (g["home"] == g["away"]).any(), f"{s}: club plays itself"
        pairs = g.groupby(["home", "away"]).size()
        assert (pairs == 1).all(), f"{s}: duplicated fixture"
        if g["played"].all():
            assert len(g) == config.MATCHES_PER_SEASON and (teams == 38).all(), f"{s}: incomplete season"
        out[s] = {"fixtures": len(g), "played": int(g["played"].sum())}
    assert df["home_goals"].dropna().ge(0).all()
    return out


def season_table(df: pd.DataFrame, season: str, upto_matchday: int | None = None) -> pd.DataFrame:
    g = df[(df["season"] == season) & df["played"]]
    if upto_matchday:
        g = g[g["matchday"] <= upto_matchday]
    rows = []
    for _, r in g.iterrows():
        hp = 3 if r["result"] == "H" else (1 if r["result"] == "D" else 0)
        ap = 3 if r["result"] == "A" else (1 if r["result"] == "D" else 0)
        rows.append((r["home"], hp, r["home_goals"], r["away_goals"], r["result"] == "H", r["result"] == "D", r["result"] == "A"))
        rows.append((r["away"], ap, r["away_goals"], r["home_goals"], r["result"] == "A", r["result"] == "D", r["result"] == "H"))
    t = pd.DataFrame(rows, columns=["team", "pts", "gf", "ga", "w", "d", "l"]).groupby("team").agg(
        played=("pts", "size"), w=("w", "sum"), d=("d", "sum"), l=("l", "sum"), gf=("gf", "sum"), ga=("ga", "sum"), pts=("pts", "sum"))
    t["gd"] = t["gf"] - t["ga"]
    t = t.sort_values(["pts", "gd", "gf"], ascending=False).reset_index()
    t.insert(0, "pos", range(1, len(t) + 1))
    return t


def save(df: pd.DataFrame) -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.MATCHES_FILE, index=False)


def load() -> pd.DataFrame:
    return pd.read_csv(config.MATCHES_FILE, parse_dates=["date"])


if __name__ == "__main__":
    m = build_matches()
    info = validate(m)
    save(m)
    for s, v in info.items():
        print(s, v)
    print(f"\n{len(m)} fixtures, {int(m.played.sum())} played, {m.home.nunique()} clubs -> {config.MATCHES_FILE}")
