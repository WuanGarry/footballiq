"""
shared_store.py
Shared prediction history storage accessible from ALL devices.

Storage backends (tried in order):
  1. JSONBin.io    – free, cloud JSON store, works on any device worldwide
                    Set: JSONBIN_KEY + JSONBIN_BIN_ID in Render env vars
  2. Local file    – fallback when JSONBin not configured (single-device only)
  3. In-memory     – last resort (resets on restart)

JSONBin.io setup (free, 2 minutes):
  1. Go to https://jsonbin.io  → Sign Up (free)
  2. Go to API Keys → Create API Key → copy it
  3. Click "New Bin" → paste [ ] → Create → copy the BIN ID from the URL
  4. In Render dashboard → Environment:
       JSONBIN_KEY   = <your api key>
       JSONBIN_BIN_ID = <your bin id>
"""

import os, json, logging, time
from datetime import datetime
from pathlib import Path

import requests

log = logging.getLogger("shared_store")

DATA_DIR     = Path(os.environ.get("DATA_DIR",
               str(Path(__file__).resolve().parent.parent / "data")))
HISTORY_FILE = DATA_DIR / "prediction_history.json"

JSONBIN_KEY    = os.environ.get("JSONBIN_KEY", "")
JSONBIN_BIN_ID = os.environ.get("JSONBIN_BIN_ID", "")
JSONBIN_BASE   = "https://api.jsonbin.io/v3"

_memory_cache: list = []
_last_fetch:   float = 0
CACHE_TTL = 15   # seconds before re-fetching from JSONBin

# ── JSONBin helpers ────────────────────────────────────────────────────────────

def _jb_headers():
    return {
        "X-Master-Key":   JSONBIN_KEY,
        "Content-Type":   "application/json",
    }


def _jb_read() -> list:
    """Read all records from JSONBin.
    Supports both formats:
      - {"predictions": [...]}   ← created with initial content
      - [...]                    ← raw array
    """
    try:
        r = requests.get(
            f"{JSONBIN_BASE}/b/{JSONBIN_BIN_ID}/latest",
            headers=_jb_headers(), timeout=10
        )
        r.raise_for_status()
        record = r.json().get("record", {})
        # Handle {"predictions": [...]} wrapper
        if isinstance(record, dict):
            return record.get("predictions", [])
        # Handle raw array
        if isinstance(record, list):
            return record
        return []
    except Exception as e:
        log.warning(f"JSONBin read failed: {e}")
        return None   # None = signal to fall back


def _jb_write(records: list) -> bool:
    """Write all records to JSONBin using {"predictions": [...]} wrapper."""
    try:
        r = requests.put(
            f"{JSONBIN_BASE}/b/{JSONBIN_BIN_ID}",
            headers=_jb_headers(),
            data=json.dumps({"predictions": records}),
            timeout=15
        )
        r.raise_for_status()
        return True
    except Exception as e:
        log.warning(f"JSONBin write failed: {e}")
        return False


# ── Local file helpers ─────────────────────────────────────────────────────────

def _file_read() -> list:
    try:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return list(_memory_cache)


def _file_write(records: list):
    global _memory_cache
    _memory_cache = records
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "w") as f:
            json.dump(records, f, indent=2)
    except Exception as e:
        log.debug(f"File write failed: {e}")


# ── Public API ─────────────────────────────────────────────────────────────────

def using_jsonbin() -> bool:
    return bool(JSONBIN_KEY and JSONBIN_BIN_ID)


def load() -> list:
    """Load all prediction records."""
    global _last_fetch, _memory_cache

    if using_jsonbin():
        # Use cache if fresh
        if time.time() - _last_fetch < CACHE_TTL and _memory_cache:
            return list(_memory_cache)
        data = _jb_read()
        if data is not None:
            _memory_cache = data
            _last_fetch   = time.time()
            return data
        # JSONBin failed — fall back to local
        log.warning("JSONBin unavailable, using local file")

    return _file_read()


def save(records: list):
    """Save all prediction records."""
    global _memory_cache, _last_fetch
    _memory_cache = records

    if using_jsonbin():
        ok = _jb_write(records)
        if ok:
            _last_fetch = time.time()
            return
        log.warning("JSONBin write failed — saving to local file as backup")

    _file_write(records)


def add_prediction(prediction: dict, match_date: str = "") -> dict:
    """Add a new prediction and return it with its assigned id."""
    records = load()
    new_id  = max((r.get("id", 0) for r in records), default=0) + 1

    record = {
        "id":               new_id,
        "timestamp":        datetime.utcnow().isoformat() + "Z",
        "match_date":       match_date or "",
        "home_team":        prediction.get("home_team", ""),
        "away_team":        prediction.get("away_team", ""),
        "division":         prediction.get("division", ""),
        "pred_result":      prediction.get("predicted_result", ""),
        "pred_home_goals":  round(float(prediction.get("expected_goals_home") or 0), 2),
        "pred_away_goals":  round(float(prediction.get("expected_goals_away") or 0), 2),
        "pred_home_corners":round(float(prediction.get("expected_corners_home") or 0), 1),
        "pred_away_corners":round(float(prediction.get("expected_corners_away") or 0), 1),
        "pred_home_bookings":round(float(prediction.get("expected_bookings_home") or 0), 1),
        "pred_away_bookings":round(float(prediction.get("expected_bookings_away") or 0), 1),
        "top_scorelines":   (prediction.get("top_scorelines") or [])[:3],
        "btts_prob":        (prediction.get("betting_insights") or {}).get("btts_probability"),
        "over25_prob":      (prediction.get("betting_insights") or {}).get("over_2_5_goals"),
        "result_probs":     prediction.get("result_probabilities") or {},
        # Actual result (filled later)
        "actual_result":        None,
        "actual_home_goals":    None,
        "actual_away_goals":    None,
        "actual_home_corners":  None,
        "actual_away_corners":  None,
        "actual_home_bookings": None,
        "actual_away_bookings": None,
        "result_correct":   None,
        "goals_error":      None,
        "corners_error":    None,
        "bookings_error":   None,
        "scoreline_in_top3":None,
        "status":           "pending",
    }

    records.insert(0, record)   # newest first
    save(records)
    return record


def resolve_prediction(record_id: int,
                       actual_home: int, actual_away: int,
                       actual_home_corners=None, actual_away_corners=None,
                       actual_home_bookings=None, actual_away_bookings=None) -> dict:
    """Update a prediction with actual result and compute accuracy metrics."""
    records = load()
    updated = {}

    for i, r in enumerate(records):
        if r.get("id") == record_id:
            r["actual_home_goals"]    = actual_home
            r["actual_away_goals"]    = actual_away
            r["actual_home_corners"]  = actual_home_corners
            r["actual_away_corners"]  = actual_away_corners
            r["actual_home_bookings"] = actual_home_bookings
            r["actual_away_bookings"] = actual_away_bookings

            r["actual_result"] = (
                "Home Win" if actual_home > actual_away else
                "Away Win" if actual_away > actual_home else "Draw"
            )
            r["result_correct"] = (r["pred_result"] == r["actual_result"])
            r["status"]         = "correct" if r["result_correct"] else "wrong"

            ph = float(r.get("pred_home_goals") or 0)
            pa = float(r.get("pred_away_goals") or 0)
            r["goals_error"] = round((abs(ph-actual_home)+abs(pa-actual_away))/2, 2)

            if actual_home_corners is not None and actual_away_corners is not None:
                phc = float(r.get("pred_home_corners") or 0)
                pac = float(r.get("pred_away_corners") or 0)
                r["corners_error"] = round((abs(phc-actual_home_corners)+abs(pac-actual_away_corners))/2, 2)

            if actual_home_bookings is not None and actual_away_bookings is not None:
                phb = float(r.get("pred_home_bookings") or 0)
                pab = float(r.get("pred_away_bookings") or 0)
                r["bookings_error"] = round((abs(phb-actual_home_bookings)+abs(pab-actual_away_bookings))/2, 2)

            top3 = [s.get("scoreline","") for s in (r.get("top_scorelines") or [])]
            r["scoreline_in_top3"] = f"{actual_home}-{actual_away}" in top3

            records[i] = r
            updated = r
            break

    save(records)
    return updated


def get_stats(records: list) -> dict:
    resolved   = [r for r in records if r.get("status") in ("correct","wrong")]
    correct    = [r for r in resolved if r.get("status") == "correct"]
    g_errs     = [r["goals_error"]    for r in resolved if r.get("goals_error")    is not None]
    c_errs     = [r["corners_error"]  for r in resolved if r.get("corners_error")  is not None]
    b_errs     = [r["bookings_error"] for r in resolved if r.get("bookings_error") is not None]
    exact      = [r for r in resolved if r.get("scoreline_in_top3")]
    avg        = lambda arr: round(sum(arr)/len(arr), 2) if arr else None

    return {
        "total_predictions": len(records),
        "resolved":          len(resolved),
        "pending":           len(records) - len(resolved),
        "correct":           len(correct),
        "wrong":             len(resolved) - len(correct),
        "accuracy_pct":      round(len(correct)/len(resolved)*100,1) if resolved else None,
        "avg_goal_error":    avg(g_errs),
        "avg_corner_error":  avg(c_errs),
        "avg_booking_error": avg(b_errs),
        "exact_scorelines":  len(exact),
        "exact_score_pct":   round(len(exact)/len(resolved)*100,1) if resolved else None,
        "backend":           "jsonbin" if using_jsonbin() else "local_file",
    }
