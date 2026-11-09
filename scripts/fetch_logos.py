"""Download club crests into assets/logos/ (run once; the dashboard falls back to initials if a file is missing).

Crests are trademarks of the clubs; they are used here only to identify teams in an educational project.
"""
import urllib.request

from PIL import Image

from dashboard.logos import LOGO_DIR, PL_BADGE_ID, slug

URL = "https://resources.premierleague.com/premierleague/badges/100/t{}.png"


def main():
    LOGO_DIR.mkdir(parents=True, exist_ok=True)
    for team, tid in PL_BADGE_ID.items():
        out = LOGO_DIR / f"{slug(team)}.png"
        if out.exists():
            continue
        try:
            urllib.request.urlretrieve(URL.format(tid), out)
            im = Image.open(out).convert("RGBA")
            im.thumbnail((72, 72), Image.LANCZOS)   # small files: crests are embedded in the page as data URIs
            im.save(out, optimize=True)
            print("ok  ", team)
        except Exception as e:  # keep going; the app shows initials instead
            print("fail", team, e)


if __name__ == "__main__":
    main()
