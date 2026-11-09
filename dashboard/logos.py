"""Club crests: local PNGs in assets/logos/, with an initials badge as fallback."""
import base64
import re

import config

LOGO_DIR = config.ROOT / "assets" / "logos"

# Premier League badge ids (resources.premierleague.com/premierleague/badges/100/t<ID>.png)
PL_BADGE_ID = {
    "Manchester United": 1, "Leeds United": 2, "Arsenal": 3, "Newcastle United": 4, "Tottenham Hotspur": 6,
    "Aston Villa": 7, "Chelsea": 8, "Everton": 11, "Leicester City": 13, "Liverpool": 14,
    "Nottingham Forest": 17, "Southampton": 20, "West Ham United": 21, "Middlesbrough": 25,
    "Crystal Palace": 31, "West Bromwich Albion": 35, "Brighton & Hove Albion": 36, "Huddersfield Town": 38,
    "Wolverhampton Wanderers": 39, "Ipswich Town": 40, "Manchester City": 43, "Norwich City": 45,
    "Sheffield United": 49, "Queens Park Rangers": 52, "Fulham": 54, "Sunderland": 56, "Watford": 57,
    "Swansea City": 80, "Hull City": 88, "Burnley": 90, "Bournemouth": 91, "Brentford": 94,
    "Cardiff City": 97, "Luton Town": 102, "Stoke City": 110,
}

SHORT = {
    "Manchester United": "Man Utd", "Manchester City": "Man City", "Tottenham Hotspur": "Spurs",
    "Brighton & Hove Albion": "Brighton", "Wolverhampton Wanderers": "Wolves", "West Bromwich Albion": "West Brom",
    "Queens Park Rangers": "QPR", "Nottingham Forest": "Nott'm Forest", "Newcastle United": "Newcastle",
    "West Ham United": "West Ham", "Sheffield United": "Sheffield Utd", "Huddersfield Town": "Huddersfield",
    "Leicester City": "Leicester", "Leeds United": "Leeds",
}

_cache: dict[str, str] = {}


def slug(team: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", team.lower()).strip("-")


def short(team: str) -> str:
    return SHORT.get(team, team)


def _initials(team: str) -> str:
    words = [w for w in re.split(r"[\s&]+", team) if w and w[0].isalpha()]
    ini = "".join(w[0] for w in words[:3]).upper()
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64">'
           f'<circle cx="32" cy="32" r="30" fill="#2a201b" stroke="#D2733A" stroke-width="2"/>'
           f'<text x="32" y="38" font-family="Arial" font-size="18" font-weight="700" fill="#EDE6DF" '
           f'text-anchor="middle">{ini}</text></svg>')
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def uri(team: str) -> str:
    """Data URI of the crest on a light round chip, so dark crests (Spurs, Swansea) stay visible on the dark theme.

    Falls back to an initials badge when the PNG has not been downloaded (python -m scripts.fetch_logos).
    """
    if team not in _cache:
        f = LOGO_DIR / f"{slug(team)}.png"
        if not f.exists():
            _cache[team] = _initials(team)
        else:
            png = "data:image/png;base64," + base64.b64encode(f.read_bytes()).decode()
            svg = ('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="64" height="64" viewBox="0 0 64 64">'
                   '<circle cx="32" cy="32" r="31" fill="#F2ECE5"/>'
                   f'<image x="11" y="11" width="42" height="42" preserveAspectRatio="xMidYMid meet" href="{png}" xlink:href="{png}"/></svg>')
            _cache[team] = "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
    return _cache[team]
