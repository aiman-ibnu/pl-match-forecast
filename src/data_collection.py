"""Download Premier League fixtures/results (openfootball, CC0) to data/raw.

Usage: python -m src.data_collection
"""
from __future__ import annotations

import json
import sys
import urllib.request

import config


def fetch(seasons=config.SEASONS) -> list[str]:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    done = []
    for s in seasons:
        url = f"{config.RAW_BASE_URL}/{s}/{config.LEAGUE_FILE}"
        with urllib.request.urlopen(url, timeout=60) as r:
            payload = json.loads(r.read().decode("utf-8"))
        (config.RAW_DIR / f"en.1_{s}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        n_done = sum(1 for m in payload["matches"] if "ft" in m.get("score", {}))
        print(f"{s}: {len(payload['matches'])} fixtures, {n_done} with a final score")
        done.append(s)
    return done


if __name__ == "__main__":
    fetch()
    sys.exit(0)
