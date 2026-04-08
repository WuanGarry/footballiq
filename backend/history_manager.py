"""
history_manager.py
Saves every prediction, then auto-fetches actual results from FBRef
including goals, corners, and bookings — and compares to what was predicted.
"""

import json, os, time, logging
from datetime import datetime, date, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

log = logging.getLogger("history")

DATA_DIR     = Path(os.environ.get("DATA_DIR",
               str(Path(__file__).resolve().parent.parent / "data")))
HISTORY_FILE = DATA_DIR / "prediction_history.json"

_memory_store: list = []

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://fbref.com/",
}


# ── Persistence ────────────────────────────────────────────────────────────────

def _load() -> list:
    try:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return list(_memory_store)


def _save(records: list):
    global _memory_store
    _memory_store = records
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "w") as f:
            json.dump(records, f, indent=2)
    except Exception as e:
        log.debug(f"Could not write history: {e}")


# ── Public API ─────────────────────────────────────────────────────────────────

def save_prediction(prediction: dict) -> dict:
    records = _load()
    record = {
        "id":               _next_id(records),
        "timestamp":        datetime.utcnow().isoformat() + "Z",
        "match_date":       prediction.get("match_date", ""),
        "home_team":        prediction["home_team"],
        "away_team":        prediction["away_team"],
        "division":         prediction.get("division", ""),
        # Predicted values
        "pred_result":      prediction["predicted_result"],
        "pred_home_goals":  round(float(prediction.get("expected_goals_home") or 0), 2),
        "pred_away_goals":  round(float(prediction.get("expected_goals_away") or 0), 2),
        "pred_home_corners":round(float(prediction.get("expected_corners_home") or 0), 1),
        "pred_away_corners":round(float(prediction.get("expected_corners_away") or 0), 1),
        "pred_home_bookings":round(float(prediction.get("expected_bookings_home") or 0), 1),
        "pred_away_bookings":round(float(prediction.get("expected_bookings_away") or 0), 1),
        "pred_total_corners":round(float(prediction.get("expected_total_corners") or 0), 1),
        "result_probs":     prediction.get("result_probabilities", {}),
        "top_scorelines":   prediction.get("top_scorelines", [])[:3],
        "btts_prob":        prediction.get("betting_insights", {}).get("btts_probability"),
        "over25_prob":      prediction.get("betting_insights", {}).get("over_2_5_goals"),
        # Actual (filled later)
        "actual_result":        None,
        "actual_home_goals":    None,
        "actual_away_goals":    None,
        "actual_home_corners":  None,
        "actual_away_corners":  None,
        "actual_home_bookings": None,
        "actual_away_bookings": None,
        # Evaluation
        "result_correct":   None,
        "goals_error":      None,
        "corners_error":    None,
        "bookings_error":   None,
        "scoreline_in_top3":None,
        "status":           "pending",
    }
    records.append(record)
    _save(records)
    return record


def get_history(limit: int = 100, division: str = None) -> dict:
    records = _load()
    if division:
        records = [r for r in records if r.get("division") == division]
    records = sorted(records, key=lambda x: x.get("timestamp",""), reverse=True)

    all_rec  = _load()
    resolved = [r for r in all_rec if r.get("status") in ("correct","wrong")]
    correct  = [r for r in resolved if r.get("status") == "correct"]

    goal_errs    = [r["goals_error"]    for r in resolved if r.get("goals_error")    is not None]
    corner_errs  = [r["corners_error"]  for r in resolved if r.get("corners_error")  is not None]
    booking_errs = [r["bookings_error"] for r in resolved if r.get("bookings_error") is not None]
    exact_scores = [r for r in resolved if r.get("scoreline_in_top3")]

    stats = {
        "total_predictions": len(all_rec),
        "resolved":          len(resolved),
        "pending":           len([r for r in all_rec if r.get("status") == "pending"]),
        "correct":           len(correct),
        "wrong":             len(resolved) - len(correct),
        "accuracy_pct":      round(len(correct)/len(resolved)*100, 1) if resolved else None,
        "avg_goal_error":    round(sum(goal_errs)/len(goal_errs), 2)      if goal_errs    else None,
        "avg_corner_error":  round(sum(corner_errs)/len(corner_errs), 2)  if corner_errs  else None,
        "avg_booking_error": round(sum(booking_errs)/len(booking_errs), 2)if booking_errs else None,
        "exact_scorelines":  len(exact_scores),
        "exact_score_pct":   round(len(exact_scores)/len(resolved)*100,1) if resolved     else None,
    }
    return {"records": records[:limit], "stats": stats}


def update_result(record_id: int, actual_home: int, actual_away: int,
                  actual_home_corners: int = None, actual_away_corners: int = None,
                  actual_home_bookings: int = None, actual_away_bookings: int = None) -> dict:
    records = _load()
    for i, r in enumerate(records):
        if r["id"] == record_id:
            records[i] = _apply_result(r, actual_home, actual_away,
                                        actual_home_corners, actual_away_corners,
                                        actual_home_bookings, actual_away_bookings)
            break
    _save(records)
    return next((r for r in records if r["id"] == record_id), {})


def check_results_from_fbref() -> int:
    """
    Auto-fetch actual results from FBRef for all pending predictions
    older than 2 hours. Scrapes goals, corners, and bookings.
    """
    records = _load()
    pending = [r for r in records if r.get("status") == "pending"]
    updated = 0

    for r in pending:
        try:
            ts = datetime.fromisoformat(r["timestamp"].rstrip("Z"))
            if (datetime.utcnow() - ts).total_seconds() < 5400:  # 90 min
                continue
        except Exception:
            continue

        result = _lookup_full_result(r["home_team"], r["away_team"])
        if result:
            idx = next((i for i,x in enumerate(records) if x["id"]==r["id"]), None)
            if idx is not None:
                records[idx] = _apply_result(
                    records[idx],
                    result["home_goals"],      result["away_goals"],
                    result.get("home_corners"), result.get("away_corners"),
                    result.get("home_bookings"), result.get("away_bookings"),
                )
                updated += 1
            time.sleep(4)

    if updated:
        _save(records)
        log.info(f"Auto-resolved {updated} predictions from FBRef")
    return updated


def _lookup_full_result(home: str, away: str) -> dict | None:
    """
    Scrape FBRef for goals + corners + bookings for a given match.
    Checks today and previous 5 days.
    """
    from team_aliases import normalise

    # Build list of dates to check — if we have match_date, try it first
    dates_to_try = []
    if match_date:  # e.g. "2026-04-07" or "07-04-2026"
        try:
            for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
                try:
                    dt = datetime.strptime(match_date.strip(), fmt)
                    dates_to_try.append(dt.strftime("%Y-%m-%d"))
                    # Also try day before/after for timezone drift
                    dates_to_try.append((dt + timedelta(days=1)).strftime("%Y-%m-%d"))
                    dates_to_try.append((dt - timedelta(days=1)).strftime("%Y-%m-%d"))
                    break
                except ValueError:
                    continue
        except Exception:
            pass
    # Then scan last 14 days as fallback
    for delta in range(0, 14):
        d_str = (datetime.utcnow() - timedelta(days=delta)).strftime("%Y-%m-%d")
        if d_str not in dates_to_try:
            dates_to_try.append(d_str)

    for check_date in dates_to_try:
        url        = f"https://fbref.com/en/matches/{check_date}"
        try:
            html = requests.get(url, headers=HEADERS, timeout=15).text
            html = html.replace("<!--", "").replace("-->", "")
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
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

            # Found the match — get score
            score = (g("score") or g("result") or "").replace("–","-")
            if "-" not in score:
                continue
            parts = score.split("-")
            if len(parts) != 2:
                continue
            try:
                hg, ag = int(parts[0].strip()), int(parts[1].strip())
            except ValueError:
                continue

            # Try to get match report link for detailed stats
            match_link = None
            for td in tr.find_all("td"):
                a_tag = td.find("a", href=True)
                if a_tag and "/matches/" in a_tag["href"] and len(a_tag["href"]) > 20:
                    match_link = "https://fbref.com" + a_tag["href"]
                    break

            result = {"home_goals": hg, "away_goals": ag}

            # Try to get corners + bookings from match report
            if match_link:
                extra = _scrape_match_report(match_link)
                result.update(extra)

            return result

        time.sleep(2)
    return None


def _scrape_match_report(url: str) -> dict:
    """Scrape a FBRef match report page for corners and bookings."""
    extra = {}
    try:
        html = requests.get(url, headers=HEADERS, timeout=15).text
        html = html.replace("<!--","").replace("-->","")
        soup = BeautifulSoup(html, "html.parser")

        # Corners: look for "CK" stat in the match stats table
        for row in soup.find_all("tr"):
            label = row.find("td", {"data-stat": "stat"})
            if label and "corner" in label.get_text(strip=True).lower():
                cells = row.find_all("td")
                if len(cells) >= 3:
                    try:
                        extra["home_corners"] = int(cells[0].get_text(strip=True))
                        extra["away_corners"] = int(cells[-1].get_text(strip=True))
                    except ValueError:
                        pass

        # Bookings: count yellow/red cards
        home_bookings = 0
        away_bookings = 0
        for event in soup.find_all(["div","span"], class_=lambda x: x and "card" in x.lower()):
            text = event.get_text(strip=True).lower()
            if "yellow" in text:
                # Rough heuristic: events before half-time separator = home
                home_bookings += 1
            elif "red" in text:
                home_bookings += 1

        # Try the events table instead
        for tr in soup.find_all("tr"):
            event_type = ""
            for td in tr.find_all("td"):
                stat = td.get("data-stat","")
                if stat == "event_type":
                    event_type = td.get_text(strip=True).lower()
                elif stat in ("cards_yellow","cards_red"):
                    val_text = td.get_text(strip=True)
                    if val_text.isdigit():
                        if "home" in tr.get("class",[]).__str__().lower():
                            home_bookings += int(val_text)
                        else:
                            away_bookings += int(val_text)

        if home_bookings or away_bookings:
            extra["home_bookings"] = home_bookings
            extra["away_bookings"] = away_bookings

        # Try direct stats summary boxes (FBRef shows team stats in #team_stats)
        stats_div = soup.find("div", id="team_stats")
        if stats_div:
            for row in stats_div.find_all("tr"):
                cells = row.find_all(["td","th"])
                if len(cells) < 3:
                    continue
                label_text = cells[1].get_text(strip=True).lower() if len(cells) > 1 else ""
                if "corner" in label_text:
                    try:
                        extra["home_corners"] = int(cells[0].get_text(strip=True).split()[0])
                        extra["away_corners"] = int(cells[2].get_text(strip=True).split()[0])
                    except Exception:
                        pass

    except Exception as e:
        log.debug(f"Match report scrape failed: {e}")
    return extra


def _names_match(a: str, b: str) -> bool:
    a, b = a.lower().strip(), b.lower().strip()
    return a == b or a in b or b in a or (len(a) > 4 and a[:4] == b[:4])


def _apply_result(r: dict, actual_home: int, actual_away: int,
                  actual_home_corners=None, actual_away_corners=None,
                  actual_home_bookings=None, actual_away_bookings=None) -> dict:
    r["actual_home_goals"]    = actual_home
    r["actual_away_goals"]    = actual_away
    r["actual_home_corners"]  = actual_home_corners
    r["actual_away_corners"]  = actual_away_corners
    r["actual_home_bookings"] = actual_home_bookings
    r["actual_away_bookings"] = actual_away_bookings

    if actual_home > actual_away:
        r["actual_result"] = "Home Win"
    elif actual_away > actual_home:
        r["actual_result"] = "Away Win"
    else:
        r["actual_result"] = "Draw"

    r["result_correct"]    = (r["pred_result"] == r["actual_result"])
    r["status"]            = "correct" if r["result_correct"] else "wrong"

    # Goals MAE
    ph = float(r.get("pred_home_goals") or 0)
    pa = float(r.get("pred_away_goals") or 0)
    r["goals_error"] = round((abs(ph - actual_home) + abs(pa - actual_away)) / 2, 2)

    # Corners MAE (if available)
    if actual_home_corners is not None and actual_away_corners is not None:
        phc = float(r.get("pred_home_corners") or 0)
        pac = float(r.get("pred_away_corners") or 0)
        r["corners_error"] = round((abs(phc - actual_home_corners) + abs(pac - actual_away_corners)) / 2, 2)

    # Bookings MAE (if available)
    if actual_home_bookings is not None and actual_away_bookings is not None:
        phb = float(r.get("pred_home_bookings") or 0)
        pab = float(r.get("pred_away_bookings") or 0)
        r["bookings_error"] = round((abs(phb - actual_home_bookings) + abs(pab - actual_away_bookings)) / 2, 2)

    # Was the actual scoreline in our top-3 predicted scorelines?
    top3 = [s["scoreline"] for s in r.get("top_scorelines", [])]
    r["scoreline_in_top3"] = f"{actual_home}-{actual_away}" in top3

    return r


def _next_id(records: list) -> int:
    return max((r["id"] for r in records), default=0) + 1
