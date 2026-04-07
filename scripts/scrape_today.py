"""
scrape_today.py
Scrapes today's match fixtures from FBRef.com using requests + BeautifulSoup.
No API key required. No browser required. Completely free.

FBRef URL for today's matches:
  https://fbref.com/en/matches/YYYY-MM-DD
"""

import os, sys, time, logging
from datetime import datetime, date, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("scrape_today")

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

# FBRef competition name → our internal division code
COMP_MAP = {
    # England
    "Premier League":           "E0",
    "EFL Championship":         "E1",
    "EFL League One":           "E2",
    "EFL League Two":           "E3",
    # Spain
    "La Liga":                  "SP1",
    "Segunda División":         "SP2",
    # Germany
    "Bundesliga":               "D1",
    "2. Bundesliga":            "D2",
    # Italy
    "Serie A":                  "I1",
    # France
    "Ligue 1":                  "F1",
    "Ligue 2":                  "F2",
    # Other European
    "Eredivisie":               "N1",
    "Belgian First Division A": "B1",
    "Primeira Liga":            "P1",
    "Süper Lig":                "T1",
    "Scottish Premiership":     "SC0",
    "Eliteserien":              "NO1",
    # UEFA
    "Champions League":         "UCL",
    "Europa League":            "UEL",
    "Europa Conference League": "UECL",
    # Americas — domestic
    "Série A":                  "BSA",
    "Liga MX":                  "MX1",
    "Major League Soccer":      "MLS",
    "Primera División":         "ARG",
    # Americas — cup competitions
    "Copa Libertadores":        "COPLIB",
    "CONMEBOL Libertadores":    "COPLIB",
    "Copa Sudamericana":        "COPASUD",
    "CONMEBOL Sudamericana":    "COPASUD",
    "CONCACAF Champions Cup":   "CONCACAF",
    "CONCACAF Champions League":"CONCACAF",
    "Liga de Campeones CONCACAF":"CONCACAF",
    # Other European
    "Greek Super League":           "GR1",
    "Austrian Football Bundesliga": "AT1",
    "Swiss Super League":           "CH1",
    "Danish Superliga":             "DK1",
    "Allsvenskan":                  "SE1",
    "Ekstraklasa":                  "PL1",
    "Liga I":                       "RO1",
    "HNL":                          "HR1",
    # African leagues
    "Ghana Premier League":         "GPL",
    "Nigerian Premier Football League": "NPFL",
    "Premier Division":             "PSL",
    "Egyptian Premier League":      "EPL_EG",
    "Botola Pro":                   "MAR1",
    "CAF Champions League":         "CAFCL",
    "CAF Confederation Cup":        "CAFCC",
    # Middle East
    "Saudi Professional League":    "SAU1",
    "Saudi Pro League":             "SAU1",
    "UAE Pro League":               "UAE1",
    "Qatar Stars League":           "QAT1",
    # Asia / Pacific
    "Indian Super League":          "ISL",
    "Chinese Super League":         "CSL",
    "K League 1":                   "KL1",
    "A-League Men":                 "AUS1",
    # South America — cups
    "Copa Libertadores":            "COPLIB",
    "CONMEBOL Libertadores":        "COPLIB",
    "Copa Sudamericana":            "COPASUD",
    "CONMEBOL Sudamericana":        "COPASUD",
    # North/Central America
    "CONCACAF Champions Cup":       "CONCACAF",
    "CONCACAF Champions League":    "CONCACAF",
    # South America — domestic
    "Primera División":             "ARG",
    "Brasileirão":                  "BSA",
    "Liga BetPlay DIMAYOR":         "COL1",
    "Liga Profesional de Fútbol":   "ARG",
    "Torneo Apertura":              "ARG",
    "Torneo Clausura":              "ARG",
    "Primera División de Uruguay":  "URU1",
    "LigaPro":                      "ECU1",
    "División Profesional":         "BOL1",
    "Liga 1":                       "PER1",
    "División de Honor":            "PAR1",
}


def _get_page(url, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            return r.text
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
            else:
                log.warning(f"Failed to fetch {url}: {e}")
                return None
    return None


def scrape_fixtures(target_date: str = None) -> list[dict]:
    """
    Scrape fixtures for target_date (YYYY-MM-DD).
    Falls back to yesterday if today has nothing yet.
    Returns list of match dicts.
    """
    if target_date is None:
        target_date = date.today().strftime("%Y-%m-%d")

    url  = f"{FBREF_BASE}/en/matches/{target_date}"
    html = _get_page(url)
    if not html:
        return []

    soup    = BeautifulSoup(html, "html.parser")
    matches = []

    # FBRef renders each competition's fixtures in a <div id="all_sched_..."> block
    # Each block contains a <table class="stats_table"> with rows of fixtures
    sched_divs = soup.find_all("div", id=lambda x: x and x.startswith("div_sched"))
    if not sched_divs:
        # Try alternate structure
        sched_divs = soup.find_all("table", {"class": "stats_table"})

    for div in sched_divs:
        # Get competition name from the section heading above this block
        comp_name = ""
        heading = div.find_previous(["h2", "h3"])
        if heading:
            comp_name = heading.get_text(strip=True)

        # Find the table inside
        table = div.find("table") if div.name == "div" else div
        if not table:
            continue

        tbody = table.find("tbody")
        if not tbody:
            continue

        for row in tbody.find_all("tr"):
            try:
                # Skip spacer/header rows
                if "thead" in row.get("class", []) or not row.find("td"):
                    continue

                def cell(stat):
                    td = row.find("td", {"data-stat": stat})
                    return td.get_text(strip=True) if td else ""

                def cell_link(stat):
                    td = row.find("td", {"data-stat": stat})
                    if td:
                        a = td.find("a")
                        return a.get_text(strip=True) if a else td.get_text(strip=True)
                    return ""

                home_team = cell_link("home_team") or cell("home_team")
                away_team = cell_link("away_team") or cell("away_team")
                score     = cell("score")
                time_str  = cell("time") or cell("start_time")

                if not home_team or not away_team:
                    continue

                # Determine status
                if score and "–" in score or (score and "-" in score):
                    parts  = score.replace("–", "-").split("-")
                    status = "FINISHED" if len(parts) == 2 else "SCHEDULED"
                else:
                    status = "SCHEDULED"

                # Map competition name to our code
                div_code = COMP_MAP.get(comp_name, "")

                matches.append({
                    "home_team":   home_team,
                    "away_team":   away_team,
                    "division":    div_code,
                    "competition": comp_name,
                    "kick_off":    time_str,
                    "status":      status,
                    "score":       score if status == "FINISHED" else "",
                })
            except Exception:
                continue

    return matches


def get_today_matches(try_yesterday: bool = True) -> tuple[list, str]:
    """
    Returns (matches_list, date_string).
    If today has no fixtures, tries yesterday's results.
    """
    today = date.today().strftime("%Y-%m-%d")
    matches = scrape_fixtures(today)

    if not matches and try_yesterday:
        yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        matches   = scrape_fixtures(yesterday)
        return matches, yesterday

    return matches, today


if __name__ == "__main__":
    matches, match_date = get_today_matches()
    print(f"\nFixtures for {match_date} — {len(matches)} matches found\n")
    for m in matches[:20]:
        score = f"  [{m['score']}]" if m["score"] else ""
        print(f"  {m['competition']:<30} {m['home_team']:<25} vs {m['away_team']:<25} {m['kick_off']:>6}{score}")
