"""Data-integrity, leakage and metric tests."""
import numpy as np
import pandas as pd
import pytest

import config
from src import data_processing as dp
from src import evaluation as ev
from src.features import FEATURES, build_features

pytestmark = pytest.mark.skipif(not config.MATCHES_FILE.exists(), reason="run the pipeline first")


@pytest.fixture(scope="module")
def matches():
    return dp.load()


@pytest.fixture(scope="module")
def feats(matches):
    return build_features(matches)[0]


def test_clean_name():
    assert dp.clean_name("Arsenal FC") == "Arsenal" and dp.clean_name("AFC Bournemouth") == "Bournemouth"
    assert dp.clean_name("Sunderland AFC") == "Sunderland" and dp.clean_name("Aston Villa") == "Aston Villa"


def test_consistent_clubs_and_complete_seasons(matches):
    info = dp.validate(matches)
    assert matches["home"].nunique() == 35
    complete = [s for s, v in info.items() if v["played"] == 380]
    assert len(complete) >= 11


def test_unplayed_placeholder_scores_are_not_results(matches):
    assert matches.loc[~matches["played"], "home_goals"].isna().all()


def test_features_use_only_past_matches(matches):
    """Changing the outcome of one match must not change that match's features nor any earlier match's."""
    base, _ = build_features(matches)
    k = int(matches[matches["played"]].iloc[2000]["match_id"])
    mod = matches.copy()
    i = mod.index[mod["match_id"] == k][0]
    mod.loc[i, ["home_goals", "away_goals"]] = [7, 0]
    mod.loc[i, "result"] = "H" if mod.loc[i, "result"] != "H" else "A"
    if mod.loc[i, "result"] == "A":
        mod.loc[i, ["home_goals", "away_goals"]] = [0, 7]
    new, _ = build_features(mod)
    upto = base["match_id"] <= k
    pd.testing.assert_frame_equal(base.loc[upto, FEATURES].reset_index(drop=True), new.loc[new["match_id"] <= k, FEATURES].reset_index(drop=True))
    later = base["match_id"] > k
    assert not np.allclose(base.loc[later, FEATURES].fillna(0).to_numpy(), new.loc[later, FEATURES].fillna(0).to_numpy())


def test_split_is_out_of_time(feats):
    tr, te, pa = ev.split_by_season(feats.dropna(subset=FEATURES))
    assert tr["date"].max() < te["date"].min() and te["date"].max() < pa["date"].min()
    assert "2014-15" not in set(tr["season"]) and len(pa) > 0 and set(pa["season"]) == {"2025-26"}


def test_rps_properties():
    y = np.array(["H", "D", "A"])
    perfect = ev.onehot(y)
    assert ev.rps(y, perfect) == 0
    uniform = np.full((3, 3), 1 / 3)
    assert ev.rps(y, uniform) > 0
    wrong_far = np.array([[0, 0, 1.0], [0, 0, 1.0], [0, 0, 1.0]])    # always H
    wrong_near = np.array([[0, 1.0, 0], [0, 1.0, 0], [0, 1.0, 0]])   # always D
    assert ev.rps(y, wrong_far) > ev.rps(y, wrong_near)             # ordering matters
