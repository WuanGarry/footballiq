"""
fetch_fbref_extra.py
Scrapes match results from FBRef.com for leagues NOT available on
football-data.co.uk — including African, Middle Eastern, Asian leagues
and the Ghana Premier League.

FBRef URL pattern:
  https://fbref.com/en/comps/{id}/schedule/{name}-Scores-and-Fixtures

No API key, no browser, no payment required.
"""

import os, sys, time, gc, logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("fetch_fbref_extra")

DATA_DIR = Path(os.environ.get("DATA_DIR",
           str(Path(__file__).resolve().parent.parent / "data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://fbref.com/",
}

FBREF_BASE = "https://fbref.com"

# ── Leagues to fetch from FBRef ────────────────────────────────────────────────
# Format: (fbref_comp_id, url_slug, our_division_code, display_name)
FBREF_LEAGUES = [
    # ── Africa ────────────────────────────────────────────────────────────────
    (322, "Ghana-Premier-League",           "GPL",   "Ghana Premier League"),
    (325, "Nigerian-Premier-Football-League","NPFL", "Nigeria Premier League"),
    (335, "Premier-Division",               "PSL",   "South Africa Premier Division"),
    (330, "Egyptian-Premier-League",        "EPL_EG","Egypt Premier League"),
    (321, "Botola-Pro",                     "MAR1",  "Morocco Botola Pro"),
    (336, "Ligue-Professionnelle-1",        "TUN1",  "Tunisia Ligue Pro 1"),
    (6,   "CAF-Champions-League",           "CAFCL", "CAF Champions League"),
    # ── Middle East ───────────────────────────────────────────────────────────
    (70,  "Saudi-Professional-League",      "SAU1",  "Saudi Pro League"),
    (79,  "UAE-Pro-League",                 "UAE1",  "UAE Pro League"),
    (80,  "Qatar-Stars-League",             "QAT1",  "Qatar Stars League"),
    # ── Asia ──────────────────────────────────────────────────────────────────
    (323, "Indian-Super-League",            "ISL",   "Indian Super League"),
    (44,  "Chinese-Super-League",           "CSL",   "Chinese Super League"),
    (55,  "K-League-1",                     "KL1",   "Korea K League 1"),
    (98,  "A-League-Men",                   "AUS1",  "Australia A-League"),
    # ── More Americas ─────────────────────────────────────────────────────────
    (325, "CONMEBOL-Libertadores",          "COPLIB","Copa Libertadores"),
    (326, "CONMEBOL-Sudamericana",          "COPASUD","Copa Sudamericana"),
    (550, "CONCACAF-Champions-Cup",         "CONCACAF","CONCACAF Champions Cup"),
    (239, "Primera-División",               "URU1",  "Uruguayan Primera División"),
    (230, "Primera-División",               "COL1",  "Colombian Primera A"),
    (44,  "Primera-División",               "CHI1",  "Chilean Primera División"),
    (232, "División-Profesional",           "BOL1",  "Bolivian División Profesional"),
    (231, "Primera-División",               "ECU1",  "Ecuadorian LigaPro"),
    (233, "Primera-División",               "PAR1",  "Paraguayan División de Honor"),
    (234, "Primera-División",               "VEN1",  "Venezuelan Primera"),
    (235, "Primera-División",               "PER1",  "Peruvian Liga 1"),
]

KEEP_COLS = [
    "Division","MatchDate","MatchTime","HomeTeam","AwayTeam",
    "HomeElo","AwayElo","Form3Home","Form5Home","Form3Away","Form5Away",
    "FTHome","FTAway","FTResult","HTHome","HTAway","HTResult",
    "HomeCorners","AwayCorners","HomeYellow","AwayYellow",
    "HomeRed","AwayRed","HomeShots","AwayShots",
]


def _get(url, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=25)
            r.raise_for_status()
            return r.text
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(4 * (attempt + 1))
            else:
                log.debug(f"Failed {url}: {e}")
    return None


def _scrape_league(comp_id, slug, div_code):
    """
    Scrape the scores-and-fixtures page for one FBRef league.
    Returns a list of row dicts.
    """
    url  = f"{FBREF_BASE}/en/comps/{comp_id}/schedule/{slug}-Scores-and-Fixtures"
    html = _get(url)
    if not html:
        return []

    # FBRef hides some tables inside HTML comments — strip them
    html = html.replace("<!--", "").replace("-->", "")
    soup = BeautifulSoup(html, "html.parser")

    table = soup.find("table", {"id": lambda x: x and "sched" in x})
    if not table:
        # Try generic stats_table
        table = soup.find("table", {"class": "stats_table"})
    if not table:
        return []

    rows = []
    for tr in table.find_all("tr"):
        try:
            def get(stat):
                td = tr.find(["td","th"], {"data-stat": stat})
                return td.get_text(strip=True) if td else ""

            def get_link(stat):
                td = tr.find(["td","th"], {"data-stat": stat})
                if td:
                    a = td.find("a")
                    return a.get_text(strip=True) if a else td.get_text(strip=True)
                return ""

            home = get_link("home_team") or get("home_team")
            away = get_link("away_team") or get("away_team")
            if not home or not away or home == "Home":
                continue

            score_raw = get("score") or get("result")
            if not score_raw or "–" not in score_raw and "-" not in score_raw:
                continue   # skip unplayed matches

            score = score_raw.replace("–", "-")
            parts = score.split("-")
            if len(parts) != 2:
                continue
            fth, fta = int(parts[0].strip()), int(parts[1].strip())
            ftr = "H" if fth > fta else ("A" if fta > fth else "D")

            # Date
            date_raw = get("date") or get("gamedate")
            if not date_raw:
                continue
            try:
                dt = datetime.strptime(date_raw.strip(), "%Y-%m-%d")
                date_str = dt.strftime("%d-%m-%Y")
            except ValueError:
                try:
                    dt = datetime.strptime(date_raw.strip(), "%d/%m/%Y")
                    date_str = dt.strftime("%d-%m-%Y")
                except ValueError:
                    continue

            rows.append({
                "Division":    div_code,
                "MatchDate":   date_str,
                "MatchTime":   get("time") or "",
                "HomeTeam":    home,
                "AwayTeam":    away,
                "HomeElo":     "", "AwayElo":    "",
                "Form3Home":   "", "Form5Home":  "",
                "Form3Away":   "", "Form5Away":  "",
                "FTHome":      fth, "FTAway":    fta,
                "FTResult":    ftr,
                "HTHome":      "", "HTAway":     "", "HTResult": "",
                "HomeCorners": "", "AwayCorners": "",
                "HomeYellow":  "", "AwayYellow":  "",
                "HomeRed":     "", "AwayRed":     "",
                "HomeShots":   "", "AwayShots":   "",
            })
        except Exception:
            continue

    return rows


def fetch_all_extra():
    out_path = DATA_DIR / "Matches.csv"
    if not out_path.exists():
        log.error("Matches.csv not found — run fetch_data.py first")
        return False

    existing = pd.read_csv(out_path, low_memory=False)
    new_rows  = []
    fetched_divs = []

    log.info(f"\nFetching {len(FBREF_LEAGUES)} extra leagues from FBRef...")

    for comp_id, slug, div_code, name in FBREF_LEAGUES:
        rows = _scrape_league(comp_id, slug, div_code)
        if rows:
            new_rows.extend(rows)
            fetched_divs.append(div_code)
            log.info(f"  {div_code:10s}  {name:<40}  {len(rows):5,} rows")
        else:
            log.debug(f"  {div_code:10s}  {name:<40}  no data")
        time.sleep(4)   # FBRef rate limit: be polite
        gc.collect()

    if not new_rows:
        log.info("No new data fetched from FBRef extra leagues.")
        return True

    new_df   = pd.DataFrame(new_rows)
    for col in existing.columns:
        if col not in new_df.columns:
            new_df[col] = ""
    new_df = new_df[[c for c in existing.columns if c in new_df.columns]]

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["MatchDate","HomeTeam","AwayTeam"], keep="last"
    ).sort_values("MatchDate").reset_index(drop=True)

    combined.to_csv(out_path, index=False)
    log.info(f"\nTotal rows now: {len(combined):,}")
    log.info(f"New divisions: {fetched_divs}")
    return True


if __name__ == "__main__":
    fetch_all_extra()
