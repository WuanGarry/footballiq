"""
history_manager.py
Saves every prediction made, then checks FBRef for the actual result
and compares prediction vs outcome to track accuracy.

Storage: data/prediction_history.json  (persists on Render disk)
         Falls back to in-memory if disk not writable.
"""

import json, os, time, logging
from datetime import datetime, date, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

log = logging.getLogger("history")

DATA_DIR    = Path(os.environ.get("DATA_DIR",
              str(Path(__file__).resolve().parent.parent / "data")))
HISTORY_FILE = DATA_DIR / "prediction_history.json"

# In-memory fallback
_memory_store: list = []

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://fbref.com/",
}


# ── Persistence helpers ────────────────────────────────────────────────────────

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
        log.debug(f"Could not write history file: {e}")


# ── Public API ─────────────────────────────────────────────────────────────────

def save_prediction(prediction: dict) -> dict:
    """
    Save a new prediction. Returns the saved record with its ID.
    """
    records = _load()

    record = {
        "id":             _next_id(records),
        "timestamp":      datetime.utcnow().isoformat() + "Z",
        "match_date":     prediction.get("match_date", ""),
        "home_team":      prediction["home_team"],
        "away_team":      prediction["away_team"],
        "division":       prediction.get("division", ""),
        # What we predicted
        "pred_result":    prediction["predicted_result"],
        "pred_home_goals": prediction.get("expected_goals_home", 0),
        "pred_away_goals": prediction.get("expected_goals_away", 0),
        "pred_home_corners": prediction.get("expected_corners_home", 0),
        "pred_away_corners": prediction.get("expected_corners_away", 0),
        "result_probs":   prediction.get("result_probabilities", {}),
        "top_scorelines": prediction.get("top_scorelines", [])[:3],
        # Actual result (filled in later)
        "actual_result":      None,
        "actual_home_goals":  None,
        "actual_away_goals":  None,
        "actual_home_corners":None,
        "actual_away_corners":None,
        # Evaluation
        "result_correct":     None,
        "goals_error":        None,   # MAE on goals
        "status":             "pending",   # pending | correct | wrong | no_data
    }

    records.append(record)
    _save(records)
    return record


def get_history(limit: int = 50, division: str = None) -> dict:
    """Return prediction history with accuracy stats."""
    records = _load()

    # Filter
    if division:
        records = [r for r in records if r.get("division") == division]

    # Sort newest first
    records = sorted(records, key=lambda x: x.get("timestamp",""), reverse=True)

    # Compute accuracy stats from all resolved records
    all_records = _load()
    resolved    = [r for r in all_records if r.get("status") in ("correct","wrong")]
    correct     = [r for r in resolved if r.get("status") == "correct"]

    # Goals accuracy
    goal_errors = [r["goals_error"] for r in resolved if r.get("goals_error") is not None]

    # Scoreline exact match
    exact_score = [r for r in resolved
                   if r.get("actual_home_goals") is not None
                   and f"{int(r['actual_home_goals'])}-{int(r['actual_away_goals'])}" in
                      [s["scoreline"] for s in r.get("top_scorelines", [])]]

    stats = {
        "total_predictions": len(all_records),
        "resolved":          len(resolved),
        "pending":           len([r for r in all_records if r.get("status") == "pending"]),
        "correct":           len(correct),
        "wrong":             len(resolved) - len(correct),
        "accuracy_pct":      round(len(correct) / len(resolved) * 100, 1) if resolved else None,
        "avg_goal_error":    round(sum(goal_errors) / len(goal_errors), 2) if goal_errors else None,
        "exact_scorelines":  len(exact_score),
        "exact_score_pct":   round(len(exact_score) / len(resolved) * 100, 1) if resolved else None,
    }

    return {
        "records": records[:limit],
        "stats":   stats,
    }


def update_result(record_id: int, actual_home: int, actual_away: int,
                  actual_home_corners: int = None, actual_away_corners: int = None) -> dict:
    """Manually update a prediction with the actual result."""
    records = _load()
    for r in records:
        if r["id"] == record_id:
            r = _apply_result(r, actual_home, actual_away,
                              actual_home_corners, actual_away_corners)
            break
    _save(records)
    return next((r for r in records if r["id"] == record_id), {})


def check_results_from_fbref():
    """
    Auto-check FBRef for results of pending predictions older than 2 hours.
    Called by the scheduler daily.
    """
    records = _load()
    pending = [r for r in records if r.get("status") == "pending"]
    updated = 0

    for r in pending:
        # Only check predictions that are at least 2 hours old
        try:
            ts = datetime.fromisoformat(r["timestamp"].rstrip("Z"))
            if (datetime.utcnow() - ts).total_seconds() < 7200:
                continue
        except Exception:
            continue

        result = _lookup_fbref_result(r["home_team"], r["away_team"], r.get("match_date",""))
        if result:
            r = _apply_result(r, result["home"], result["away"])
            updated += 1
            time.sleep(3)   # polite delay

    if updated:
        _save(records)
        log.info(f"Auto-resolved {updated} pending predictions from FBRef")

    return updated


def _lookup_fbref_result(home: str, away: str, match_date: str) -> dict | None:
    """Try to find the actual result on FBRef for a given match."""
    # Try today and yesterday
    for delta in range(0, 4):
        try:
            check_date = (datetime.utcnow() - timedelta(days=delta)).strftime("%Y-%m-%d")
            url  = f"https://fbref.com/en/matches/{check_date}"
            html = requests.get(url, headers=HEADERS, timeout=15).text
            html = html.replace("<!--","").replace("-->","")
            soup = BeautifulSoup(html, "html.parser")

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

                h = gl("home_team") or g("home_team")
                a = gl("away_team") or g("away_team")
                score = g("score") or g("result")

                # Fuzzy match team names
                from team_aliases import normalise
                h_norm = normalise(h)
                a_norm = normalise(a)

                if (h_norm.lower() in home.lower() or home.lower() in h_norm.lower()) and \
                   (a_norm.lower() in away.lower() or away.lower() in a_norm.lower()):
                    if score and ("–" in score or "-" in score):
                        parts = score.replace("–","-").split("-")
                        if len(parts) == 2:
                            return {"home": int(parts[0].strip()),
                                    "away": int(parts[1].strip())}
        except Exception:
            pass
        time.sleep(2)
    return None


def _apply_result(r: dict, actual_home: int, actual_away: int,
                  actual_home_corners: int = None, actual_away_corners: int = None) -> dict:
    """Fill in actual result and compute evaluation metrics."""
    r["actual_home_goals"]  = actual_home
    r["actual_away_goals"]  = actual_away
    r["actual_home_corners"] = actual_home_corners
    r["actual_away_corners"] = actual_away_corners

    # Actual result label
    if actual_home > actual_away:
        r["actual_result"] = "Home Win"
    elif actual_away > actual_home:
        r["actual_result"] = "Away Win"
    else:
        r["actual_result"] = "Draw"

    # Was prediction correct?
    r["result_correct"] = (r["pred_result"] == r["actual_result"])
    r["status"]         = "correct" if r["result_correct"] else "wrong"

    # Goals error (MAE)
    pred_h = float(r.get("pred_home_goals", 0) or 0)
    pred_a = float(r.get("pred_away_goals", 0) or 0)
    r["goals_error"] = round(
        (abs(pred_h - actual_home) + abs(pred_a - actual_away)) / 2, 2
    )

    return r


def _next_id(records: list) -> int:
    return max((r["id"] for r in records), default=0) + 1
