"""
history_manager.py
Auto-fetches actual results from FBRef for pending predictions.
Scrapes goals, corners, yellow cards, and red cards.
"""

import json, os, time, logging
from datetime import datetime, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

log = logging.getLogger("history")

DATA_DIR     = Path(os.environ.get("DATA_DIR",
               str(Path(__file__).resolve().parent.parent / "data")))

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://fbref.com/",
}


def check_results_from_fbref() -> int:
    """
    Auto-fetch actual results for all pending predictions older than 90 min.
    Tries to find goals, corners, yellows, and reds.
    Returns number of predictions updated.
    """
    import shared_store
    records = shared_store.load()
    pending = [r for r in records if r.get("status") == "pending"]
    updated = 0

    for r in pending:
        # Skip if prediction is too recent (match may still be playing)
        try:
            ts  = datetime.fromisoformat(r["timestamp"].rstrip("Z"))
            age = (datetime.utcnow() - ts).total_seconds()
            if age < 5400:   # 90 minutes
                continue
        except Exception:
            continue

        result = _lookup_full_result(
            r["home_team"], r["away_team"], r.get("match_date", "")
        )
        if result:
            shared_store.resolve_prediction(
                r["id"],
                result["home_goals"],       result["away_goals"],
                result.get("home_corners"), result.get("away_corners"),
                result.get("home_yellows"), result.get("away_yellows"),
                result.get("home_reds"),    result.get("away_reds"),
            )
            updated += 1
            log.info(f"Auto-resolved: {r['home_team']} vs {r['away_team']} "
                     f"→ {result['home_goals']}-{result['away_goals']}")
            time.sleep(3)

    if updated:
        log.info(f"Auto-resolved {updated} predictions")
    return updated


def _lookup_full_result(home: str, away: str, match_date: str = "") -> dict | None:
    """
    Search FBRef for the actual match result.
    Scans today and 14 days back.
    Returns dict with home_goals, away_goals, and optional corners/cards.
    """
    from team_aliases import normalise

    # Build date list — stored match_date first for efficiency
    dates = []
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            dt = datetime.strptime(match_date.strip(), fmt)
            for delta in range(-1, 2):
                d = (dt + timedelta(days=delta)).strftime("%Y-%m-%d")
                if d not in dates:
                    dates.append(d)
            break
        except (ValueError, AttributeError):
            continue

    for delta in range(0, 15):
        d = (datetime.utcnow() - timedelta(days=delta)).strftime("%Y-%m-%d")
        if d not in dates:
            dates.append(d)

    for check_date in dates:
        url = f"https://fbref.com/en/matches/{check_date}"
        try:
            html = requests.get(url, headers=HEADERS, timeout=15).text
            html = html.replace("<!--", "").replace("-->", "")
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            time.sleep(2)
            continue

        for tr in soup.find_all("tr"):
            def g(stat):
                td = tr.find(["td","th"], {"data-stat": stat})
                return td.get_text(strip=True) if td else ""
            def gl(stat):
                td = tr.find(["td","th"], {"data-stat": stat})
                if td:
                    a = td.find("a")
                    return a.get_text(strip=True) if a else td.get_text(strip=True)
                return ""

            h_raw = gl("home_team") or g("home_team")
            a_raw = gl("away_team") or g("away_team")
            if not h_raw or not a_raw:
                continue

            h_norm = normalise(h_raw)
            a_norm = normalise(a_raw)

            if not (_names_match(h_norm, home) and _names_match(a_norm, away)):
                continue

            # Found the match
            score = (g("score") or g("result") or "").replace("–", "-")
            if "-" not in score:
                continue
            parts = score.split("-")
            if len(parts) != 2:
                continue
            try:
                hg, ag = int(parts[0].strip()), int(parts[1].strip())
            except ValueError:
                continue

            result = {"home_goals": hg, "away_goals": ag}

            # Try to get detailed stats from match report page
            match_url = None
            for td in tr.find_all("td"):
                a_tag = td.find("a", href=True)
                if a_tag and "/matches/" in a_tag["href"] and \
                   len(a_tag["href"].split("/")) >= 4:
                    candidate = "https://fbref.com" + a_tag["href"]
                    if candidate != url:
                        match_url = candidate
                        break

            if match_url:
                extra = _scrape_match_report(match_url)
                result.update(extra)

            return result

        time.sleep(2)
    return None


def _scrape_match_report(url: str) -> dict:
    """
    Scrape a FBRef match report page for:
    - corners (HC/AC)
    - yellow cards (HY/AY)
    - red cards (HR/AR)
    """
    extra = {}
    try:
        html = requests.get(url, headers=HEADERS, timeout=15).text
        html = html.replace("<!--", "").replace("-->", "")
        soup = BeautifulSoup(html, "html.parser")

        # Method 1: team stats table (most reliable)
        stats_div = (soup.find("div", id="team_stats") or
                     soup.find("div", id="team_stats_extra"))
        if stats_div:
            for tr in stats_div.find_all("tr"):
                cells = tr.find_all(["td","th"])
                if len(cells) < 3:
                    continue
                label = cells[1].get_text(strip=True).lower() if len(cells) > 1 else ""

                if "corner" in label:
                    try:
                        extra["home_corners"] = int(cells[0].get_text(strip=True).split()[0])
                        extra["away_corners"] = int(cells[2].get_text(strip=True).split()[0])
                    except Exception:
                        pass

        # Method 2: look for shot stats tables which include CK (corner kicks)
        for table in soup.find_all("table"):
            table_id = table.get("id", "")
            # Shot stats tables have CK column
            if "shots" in table_id or "summary" in table_id:
                header_row = table.find("thead")
                if not header_row:
                    continue
                headers = [th.get("data-stat","").lower()
                           for th in header_row.find_all(["th","td"])]
                if "corner_kicks" in headers or "ck" in headers:
                    ck_idx = next((i for i,h in enumerate(headers)
                                   if h in ("corner_kicks","ck")), None)
                    if ck_idx is not None:
                        rows_data = table.find("tbody").find_all("tr") if table.find("tbody") else []
                        ck_vals = []
                        for row in rows_data:
                            cells = row.find_all(["td","th"])
                            if len(cells) > ck_idx:
                                try:
                                    ck_vals.append(int(cells[ck_idx].get_text(strip=True)))
                                except ValueError:
                                    pass
                        if len(ck_vals) >= 2 and "home_corners" not in extra:
                            extra["home_corners"] = ck_vals[0]
                            extra["away_corners"] = ck_vals[1]

        # Method 3: Scrape the events list for cards
        home_yellows = 0
        away_yellows = 0
        home_reds    = 0
        away_reds    = 0

        # FBRef events are in #events_wrap divs
        events_div = soup.find("div", id="events_wrap")
        if events_div:
            for event in events_div.find_all("div", class_=lambda x: x and "event" in x):
                text  = event.get_text(" ", strip=True).lower()
                cls   = " ".join(event.get("class", []))
                is_home = "a" not in cls  # home events typically don't have 'away' class
                is_away = "away" in cls or "b" in cls

                if "yellow card" in text or "caution" in text:
                    if is_away:
                        away_yellows += 1
                    else:
                        home_yellows += 1
                elif "red card" in text or "sending off" in text:
                    if is_away:
                        away_reds += 1
                    else:
                        home_reds += 1

        # Fallback: count card icons in the timeline
        if home_yellows == 0 and away_yellows == 0:
            for span in soup.find_all("span", class_=lambda x: x and "yellow" in str(x).lower()):
                parent_text = span.parent.get_text() if span.parent else ""
                # Very rough — left side = home
                pos = soup.get_text().find(span.get_text())
                if pos < len(soup.get_text()) // 2:
                    home_yellows += 1
                else:
                    away_yellows += 1

        if home_yellows or away_yellows or home_reds or away_reds:
            extra.update({
                "home_yellows": home_yellows,
                "away_yellows": away_yellows,
                "home_reds":    home_reds,
                "away_reds":    away_reds,
            })

    except Exception as e:
        log.debug(f"Match report scrape error: {e}")

    return extra


def _names_match(a: str, b: str) -> bool:
    """Fuzzy name match — true if names are close enough."""
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return True
    if len(a) > 4 and len(b) > 4:
        # One contains the other
        if a in b or b in a:
            return True
        # First 5 chars match
        if a[:5] == b[:5]:
            return True
    return False
