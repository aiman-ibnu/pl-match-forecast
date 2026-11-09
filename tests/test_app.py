"""Smoke tests: every page renders without exception."""
import pytest
from streamlit.testing.v1 import AppTest

import config

pytestmark = pytest.mark.skipif(not config.MODEL_FILE.exists(), reason="run the pipeline first")
APP = str(config.ROOT / "app.py") if hasattr(config, "ROOT") else str(__import__("pathlib").Path(__file__).resolve().parents[1] / "app.py")
PAGES = ["Overview", "Team Elo & Form", "Prediction", "Insights"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]


def test_prediction_other_matchup():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Prediction").run()
    at.selectbox[1].select_index(3).run()
    assert not at.exception


def test_overview_all_seasons():
    at = AppTest.from_file(APP, default_timeout=60).run()
    for opt in at.selectbox[0].options:
        at.selectbox[0].select(opt).run()
        assert not at.exception
