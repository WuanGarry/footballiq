"""
scrape_today.py
Scrapes today's fixtures from FBRef and returns them grouped by
Continent → Country → League, just like Flashscore.
Times are returned in GMT (UTC).
"""

import os, sys, time, logging
from datetime import datetime, date, timedelta
from pathlib import Path
from collections import defaultdict

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

# ── Continent / Country / League mapping ─────────────────────────────────────
# Format: "FBRef Competition Name": (continent, country, our_div_code)

LEAGUE_META = {
    # ═══════════════════════════════════════════════════
    # EUROPE
    # ═══════════════════════════════════════════════════
    # UEFA
    "Champions League":             ("Europe", "UEFA", "UCL"),
    "Europa League":                ("Europe", "UEFA", "UEL"),
    "Europa Conference League":     ("Europe", "UEFA", "UECL"),

    # England
    "Premier League":               ("Europe", "England", "E0"),
    "EFL Championship":             ("Europe", "England", "E1"),
    "EFL League One":               ("Europe", "England", "E2"),
    "EFL League Two":               ("Europe", "England", "E3"),
    "FA Cup":                       ("Europe", "England", "FAC"),

    # Spain
    "La Liga":                      ("Europe", "Spain", "SP1"),
    "Segunda División":             ("Europe", "Spain", "SP2"),
    "Copa del Rey":                 ("Europe", "Spain", "CDR"),

    # Germany
    "Bundesliga":                   ("Europe", "Germany", "D1"),
    "2. Bundesliga":                ("Europe", "Germany", "D2"),
    "DFB-Pokal":                    ("Europe", "Germany", "DFB"),

    # Italy
    "Serie A":                      ("Europe", "Italy", "I1"),
    "Serie B":                      ("Europe", "Italy", "I2"),
    "Coppa Italia":                 ("Europe", "Italy", "CIT"),

    # France
    "Ligue 1":                      ("Europe", "France", "F1"),
    "Ligue 2":                      ("Europe", "France", "F2"),
    "Coupe de France":              ("Europe", "France", "CDF"),

    # Netherlands
    "Eredivisie":                   ("Europe", "Netherlands", "N1"),
    "Eerste Divisie":               ("Europe", "Netherlands", "N2"),
    "KNVB Beker":                   ("Europe", "Netherlands", "KNVB"),

    # Belgium
    "Belgian First Division A":     ("Europe", "Belgium", "B1"),
    "Pro League":                   ("Europe", "Belgium", "B1"),
    "Belgian Cup":                  ("Europe", "Belgium", "BCUP"),

    # Portugal
    "Primeira Liga":                ("Europe", "Portugal", "P1"),
    "Taça de Portugal":             ("Europe", "Portugal", "TCP"),

    # Turkey
    "Süper Lig":                    ("Europe", "Turkey", "T1"),
    "TFF First League":             ("Europe", "Turkey", "T2"),

    # Scotland
    "Scottish Premiership":         ("Europe", "Scotland", "SC0"),
    "Scottish Championship":        ("Europe", "Scotland", "SC1"),

    # Norway
    "Eliteserien":                  ("Europe", "Norway", "NO1"),
    "Norwegian First Division":     ("Europe", "Norway", "NO2"),

    # Sweden
    "Allsvenskan":                  ("Europe", "Sweden", "SE1"),
    "Superettan":                   ("Europe", "Sweden", "SE2"),

    # Denmark
    "Danish Superliga":             ("Europe", "Denmark", "DK1"),

    # Switzerland
    "Swiss Super League":           ("Europe", "Switzerland", "CH1"),

    # Austria
    "Austrian Football Bundesliga": ("Europe", "Austria", "AT1"),
    "Bundesliga":                   ("Europe", "Austria", "AT1"),  # avoid clash

    # Greece
    "Greek Super League":           ("Europe", "Greece", "GR1"),
    "Super League 1":               ("Europe", "Greece", "GR1"),

    # Poland
    "Ekstraklasa":                  ("Europe", "Poland", "PL1"),

    # Romania
    "Liga I":                       ("Europe", "Romania", "RO1"),

    # Croatia
    "HNL":                          ("Europe", "Croatia", "HR1"),

    # Russia
    "Russian Premier League":       ("Europe", "Russia", "RU1"),

    # Ukraine
    "Ukrainian Premier League":     ("Europe", "Ukraine", "UA1"),

    # Czech Republic
    "Czech First League":           ("Europe", "Czech Republic", "CZ1"),
    "Czech Liga":                   ("Europe", "Czech Republic", "CZ1"),

    # Hungary
    "OTP Bank Liga":                ("Europe", "Hungary", "HU1"),

    # Serbia
    "Serbian SuperLiga":            ("Europe", "Serbia", "RS1"),

    # ═══════════════════════════════════════════════════
    # AFRICA
    # ═══════════════════════════════════════════════════
    "Ghana Premier League":         ("Africa", "Ghana", "GPL"),
    "Nigerian Premier Football League": ("Africa", "Nigeria", "NPFL"),
    "Premier Division":             ("Africa", "South Africa", "PSL"),
    "Egyptian Premier League":      ("Africa", "Egypt", "EPL_EG"),
    "Botola Pro":                   ("Africa", "Morocco", "MAR1"),
    "Ligue Professionnelle 1":      ("Africa", "Tunisia", "TUN1"),
    "CAF Champions League":         ("Africa", "CAF", "CAFCL"),
    "CAF Confederation Cup":        ("Africa", "CAF", "CAFCC"),

    # ═══════════════════════════════════════════════════
    # SOUTH AMERICA
    # ═══════════════════════════════════════════════════
    "Copa Libertadores":            ("South America", "CONMEBOL", "COPLIB"),
    "CONMEBOL Libertadores":        ("South America", "CONMEBOL", "COPLIB"),
    "Copa Sudamericana":            ("South America", "CONMEBOL", "COPASUD"),
    "CONMEBOL Sudamericana":        ("South America", "CONMEBOL", "COPASUD"),
    "Série A":                      ("South America", "Brazil", "BSA"),
    "Série B":                      ("South America", "Brazil", "BSB"),
    "Brasileirão":                  ("South America", "Brazil", "BSA"),
    "Liga Profesional de Fútbol":   ("South America", "Argentina", "ARG"),
    "Primera División":             ("South America", "Argentina", "ARG"),
    "Torneo Apertura":              ("South America", "Argentina", "ARG"),
    "Torneo Clausura":              ("South America", "Argentina", "ARG"),
    "Liga BetPlay DIMAYOR":         ("South America", "Colombia", "COL1"),
    "Primera División de Uruguay":  ("South America", "Uruguay", "URU1"),
    "LigaPro":                      ("South America", "Ecuador", "ECU1"),
    "División Profesional":         ("South America", "Bolivia", "BOL1"),
    "Liga 1":                       ("South America", "Peru", "PER1"),
    "División de Honor":            ("South America", "Paraguay", "PAR1"),
    "Primera División de Chile":    ("South America", "Chile", "CHI1"),
    "Primera División Venezuela":   ("South America", "Venezuela", "VEN1"),

    # ═══════════════════════════════════════════════════
    # NORTH & CENTRAL AMERICA
    # ═══════════════════════════════════════════════════
    "CONCACAF Champions Cup":       ("North America", "CONCACAF", "CONCACAF"),
    "CONCACAF Champions League":    ("North America", "CONCACAF", "CONCACAF"),
    "Major League Soccer":          ("North America", "USA / Canada", "MLS"),
    "Liga MX":                      ("North America", "Mexico", "MX1"),
    "Liga de Expansión MX":         ("North America", "Mexico", "MX2"),
    "Scotiabank Concacaf":          ("North America", "CONCACAF", "CONCACAF"),
    "Canadian Premier League":      ("North America", "Canada", "CPL"),

    # ═══════════════════════════════════════════════════
    # ASIA
    # ═══════════════════════════════════════════════════
    "AFC Champions League":         ("Asia", "AFC", "AFCCL"),
    "AFC Cup":                      ("Asia", "AFC", "AFCC"),
    "Indian Super League":          ("Asia", "India", "ISL"),
    "Chinese Super League":         ("Asia", "China", "CSL"),
    "K League 1":                   ("Asia", "South Korea", "KL1"),
    "K League 2":                   ("Asia", "South Korea", "KL2"),
    "J1 League":                    ("Asia", "Japan", "JP1"),
    "Saudi Professional League":    ("Asia", "Saudi Arabia", "SAU1"),
    "Saudi Pro League":             ("Asia", "Saudi Arabia", "SAU1"),
    "UAE Pro League":               ("Asia", "UAE", "UAE1"),
    "Qatar Stars League":           ("Asia", "Qatar", "QAT1"),
    "A-League Men":                 ("Asia / Pacific", "Australia", "AUS1"),
    "A-League Women":               ("Asia / Pacific", "Australia", "AUS_W"),

    # ═══════════════════════════════════════════════════
    # WORLD
    # ═══════════════════════════════════════════════════
    "FIFA World Cup":               ("World", "FIFA", "WC"),
    "FIFA World Cup qualification": ("World", "FIFA", "WCQ"),
    "UEFA Nations League":          ("World", "UEFA", "UNL"),
    "Africa Cup of Nations":        ("World", "CAF", "AFCON"),
    "Copa America":                 ("World", "CONMEBOL", "CA"),
    "EURO":                         ("World", "UEFA", "EURO"),
}

# Continent sort order (Flashscore-style: Europe first)
CONTINENT_ORDER = [
    "Europe", "Africa", "South America",
    "North America", "Asia", "Asia / Pacific", "World"
]

# Country sort order within Europe (big leagues first)
COUNTRY_PRIORITY = {
    "UEFA":        0,
    "England":     1, "Spain":   2, "Germany": 3,
    "Italy":       4, "France":  5, "Netherlands": 6,
    "Belgium":     7, "Portugal":8, "Turkey":  9,
    "Scotland":    10,"Norway":  11,"Sweden":  12,
    "Denmark":     13,"Switzerland":14,"Austria":15,
    "Greece":      16,"Poland":  17,"Romania": 18,
    "Croatia":     19,"Russia":  20,"Ukraine": 21,
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
                log.warning(f"Failed {url}: {e}")
    return None


def _parse_time_to_gmt(raw_time: str) -> str:
    """
    FBRef shows times in EST (UTC-5) during winter or EDT (UTC-4) during summer.
    We convert to GMT (UTC+0) and label it GMT.
    Alternatively FBRef may already show UTC — we keep it as is.
    """
    if not raw_time or raw_time.strip() in ("", "–", "-"):
        return ""
    raw_time = raw_time.strip()
    # If it already contains UTC or GMT, strip and reformat
    for suffix in (" UTC", " GMT", "UTC", "GMT"):
        if raw_time.endswith(suffix):
            return raw_time.replace(suffix, "").strip() + " GMT"
    # FBRef US times are EST (UTC-5) — add 5 hours
    try:
        t = datetime.strptime(raw_time, "%H:%M")
        # FBRef shows ET (UTC-5 standard, UTC-4 summer)
        # Simple heuristic: add 5 hours for GMT
        gmt = t.replace(hour=(t.hour + 5) % 24)
        return gmt.strftime("%H:%M") + " GMT"
    except ValueError:
        return raw_time + " GMT"


def scrape_fixtures(target_date: str = None) -> list:
    if target_date is None:
        target_date = date.today().strftime("%Y-%m-%d")

    url  = f"{FBREF_BASE}/en/matches/{target_date}"
    html = _get_page(url)
    if not html:
        return []

    html = html.replace("<!--", "").replace("-->", "")
    soup = BeautifulSoup(html, "html.parser")

    matches = []

    # Each competition block has an h2/h3 heading then a table
    # We iterate through all tables and find their preceding heading
    for table in soup.find_all("table", {"class": lambda x: x and "stats_table" in x}):
        # Find competition name from nearest heading
        comp_name = ""
        for tag in ["h2", "h3", "h4"]:
            h = table.find_previous(tag)
            if h:
                comp_name = h.get_text(strip=True)
                # Strip common suffixes like "Scores & Fixtures"
                for suffix in [" Scores & Fixtures", " Scores and Fixtures",
                                " — Scores & Fixtures", " Schedule"]:
                    comp_name = comp_name.replace(suffix, "")
                comp_name = comp_name.strip()
                break

        tbody = table.find("tbody")
        if not tbody:
            continue

        for tr in tbody.find_all("tr"):
            try:
                if not tr.find("td"):
                    continue

                def cell(stat):
                    td = tr.find(["td","th"], {"data-stat": stat})
                    return td.get_text(strip=True) if td else ""

                def cell_link(stat):
                    td = tr.find(["td","th"], {"data-stat": stat})
                    if td:
                        a = td.find("a")
                        return a.get_text(strip=True) if a else td.get_text(strip=True)
                    return ""

                home = cell_link("home_team") or cell("home_team")
                away = cell_link("away_team") or cell("away_team")
                if not home or not away or home in ("Home","Squad"):
                    continue

                score_raw = cell("score") or cell("result") or ""
                score     = score_raw.replace("–","-").strip()

                # Determine status
                if score and "-" in score and score.replace("-","").replace(" ","").isdigit():
                    status = "FINISHED"
                elif ":" in score or "Live" in score_raw:
                    status = "IN_PLAY"
                else:
                    status = "SCHEDULED"

                raw_time = cell("time") or cell("start_time") or ""
                kick_off = _parse_time_to_gmt(raw_time)

                # Look up meta
                meta = LEAGUE_META.get(comp_name)
                if not meta:
                    # Try partial match
                    for k, v in LEAGUE_META.items():
                        if k.lower() in comp_name.lower() or comp_name.lower() in k.lower():
                            meta = v
                            break
                if not meta:
                    meta = ("Other", comp_name, "")

                continent, country, div_code = meta

                matches.append({
                    "home_team":   home,
                    "away_team":   away,
                    "division":    div_code,
                    "competition": comp_name,
                    "country":     country,
                    "continent":   continent,
                    "kick_off":    kick_off,
                    "status":      status,
                    "score":       score if status == "FINISHED" else "",
                })
            except Exception:
                continue

    return matches


def get_today_matches(try_yesterday: bool = True) -> tuple:
    today   = date.today().strftime("%Y-%m-%d")
    matches = scrape_fixtures(today)
    if not matches and try_yesterday:
        yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        return scrape_fixtures(yesterday), yesterday
    return matches, today


def group_by_continent_country(matches: list) -> list:
    """
    Returns matches grouped as:
    [
      {
        "continent": "Europe",
        "countries": [
          {
            "country": "England",
            "leagues": [
              {
                "competition": "Premier League",
                "division": "E0",
                "matches": [...]
              }
            ]
          }
        ]
      }
    ]
    Sorted: Europe first, big leagues first within Europe.
    """
    # Build nested dict
    tree = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for m in matches:
        tree[m["continent"]][m["country"]][m["competition"]].append(m)

    result = []
    # Sort continents
    sorted_continents = sorted(
        tree.keys(),
        key=lambda c: CONTINENT_ORDER.index(c) if c in CONTINENT_ORDER else 99
    )

    for continent in sorted_continents:
        countries_data = []
        country_dict   = tree[continent]

        # Sort countries
        sorted_countries = sorted(
            country_dict.keys(),
            key=lambda c: COUNTRY_PRIORITY.get(c, 50)
        )

        for country in sorted_countries:
            leagues_data = []
            for comp, comp_matches in sorted(country_dict[country].items()):
                # Sort matches by kick-off time
                comp_matches_sorted = sorted(
                    comp_matches,
                    key=lambda m: m.get("kick_off","") or ""
                )
                div_code = comp_matches_sorted[0]["division"] if comp_matches_sorted else ""
                leagues_data.append({
                    "competition": comp,
                    "division":    div_code,
                    "matches":     comp_matches_sorted,
                })
            countries_data.append({
                "country": country,
                "leagues": leagues_data,
            })

        result.append({
            "continent": continent,
            "countries": countries_data,
        })

    return result


if __name__ == "__main__":
    matches, match_date = get_today_matches()
    grouped = group_by_continent_country(matches)
    print(f"\nFixtures for {match_date} — {len(matches)} total\n")
    for cont in grouped:
        print(f"  ═══ {cont['continent']} ═══")
        for ctry in cont["countries"]:
            print(f"    🏳 {ctry['country']}")
            for lg in ctry["leagues"]:
                print(f"      🏆 {lg['competition']}  ({len(lg['matches'])} matches)")
                for m in lg["matches"][:3]:
                    score = f"  [{m['score']}]" if m["score"] else ""
                    print(f"        {m['kick_off']:>8}  {m['home_team']} vs {m['away_team']}{score}")
