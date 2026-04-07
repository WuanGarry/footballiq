"""
app.py  –  Flask REST API for the football prediction system.

Endpoints
---------
GET  /                   → serves frontend/index.html
GET  /api/teams          → list of all teams  (optionally ?division=E0)
GET  /api/divisions      → list of all divisions
POST /api/predict        → { home_team, away_team, division } → prediction
POST /api/update-data    → triggers data update + model retrain (async)
GET  /api/model-status   → reports last train time, team count, etc.
"""

import os
import sys
import json
import subprocess
import threading
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
try:
    from flask_cors import CORS
except ImportError:
    def CORS(app, **kw): return app   # no-op fallback if flask-cors not installed

# Make sure /backend is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import predictor
import history_manager
from team_aliases import normalise

BASE_DIR     = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
MODELS_DIR   = BASE_DIR / "models"

app = Flask(__name__, static_folder=str(FRONTEND_DIR))
CORS(app)

_retrain_lock   = threading.Lock()
_retrain_status = {"running": False, "last_run": None, "message": "Not started yet"}


# ─────────────────────────────────────────────────────────────────────────────
# Static / Frontend
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(str(FRONTEND_DIR), filename)


# ─────────────────────────────────────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ok(data):
    return jsonify({"status": "ok", "data": data})


def _err(msg, code=400):
    return jsonify({"status": "error", "message": msg}), code


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/teams")
def api_teams():
    division = request.args.get("division")
    teams = predictor.get_teams(division)
    return _ok({"teams": teams, "count": len(teams)})


@app.route("/api/divisions")
def api_divisions():
    divisions = predictor.get_divisions()
    # Human-readable labels for all supported competitions
    labels = {
        # ── UEFA European Competitions ──────────────────────────────────
        "UCL":  "UEFA Champions League",
        "UEL":  "UEFA Europa League",
        "UECL": "UEFA Europa Conference League",
        # ── England ─────────────────────────────────────────────────────
        "E0":   "English Premier League",
        "E1":   "English Championship",
        "E2":   "English League One",
        "E3":   "English League Two",
        # ── Spain ───────────────────────────────────────────────────────
        "SP1":  "Spanish La Liga",
        "SP2":  "Spanish Segunda División",
        # ── Germany ─────────────────────────────────────────────────────
        "D1":   "German Bundesliga",
        "D2":   "German 2. Bundesliga",
        # ── Italy ───────────────────────────────────────────────────────
        "I1":   "Italian Serie A",
        # ── France ──────────────────────────────────────────────────────
        "F1":   "French Ligue 1",
        "F2":   "French Ligue 2",
        # ── Other European ──────────────────────────────────────────────
        "N1":   "Dutch Eredivisie",
        "B1":   "Belgian First Division A",
        "P1":   "Portuguese Primeira Liga",
        "T1":   "Turkish Süper Lig",
        "SC0":  "Scottish Premiership",
        "NO1":  "Norwegian Eliteserien",
        # ── Rest of Europe ──────────────────────────────────────────────
        "GR1":  "Greek Super League",
        "AT1":  "Austrian Bundesliga",
        "CH1":  "Swiss Super League",
        "DK1":  "Danish Superliga",
        "SE1":  "Swedish Allsvenskan",
        "PL1":  "Polish Ekstraklasa",
        "RO1":  "Romanian Liga I",
        # ── Americas ────────────────────────────────────────────────────
        "BSA":  "Brazilian Série A",
        "ARG":  "Argentine Liga Profesional",
        "MLS":  "MLS (USA / Canada)",
        "MX1":  "Liga MX (Mexico)",
        # ── Asia ────────────────────────────────────────────────────────
        "JP1":  "J1 League (Japan)",
    }
    result = [
        {"code": d, "label": labels.get(d, d)}
        for d in divisions
    ]
    return _ok({"divisions": result})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    body = request.get_json(force=True, silent=True) or {}

    home_team = body.get("home_team", "").strip()
    away_team = body.get("away_team", "").strip()
    division  = body.get("division",  "").strip() or None

    if not home_team or not away_team:
        return _err("Both 'home_team' and 'away_team' are required.")
    if home_team == away_team:
        return _err("Home team and away team must be different.")

    teams      = predictor.get_teams()
    home_team  = normalise(home_team, teams)
    away_team  = normalise(away_team, teams)

    if home_team not in teams:
        return _err(f"Unknown home team: '{home_team}' — not in model yet")
    if away_team not in teams:
        return _err(f"Unknown away team: '{away_team}' — not in model yet")

    try:
        result = predictor.predict(home_team, away_team, division)
        # Save to history if client requests it (default: yes)
        if body.get("save_history", True):
            match_date = body.get("match_date", "")
            result["match_date"] = match_date
            saved = history_manager.save_prediction(result)
            result["history_id"] = saved["id"]
        return _ok(result)
    except Exception as exc:
        app.logger.exception("Prediction error")
        return _err(f"Prediction failed: {str(exc)}", 500)


def _run_update():
    """Background thread that fetches new data and retrains models."""
    global _retrain_status
    _retrain_status["running"] = True
    _retrain_status["message"] = "Running update …"
    try:
        scripts_dir = BASE_DIR / "scripts"
        # Step 1: fetch new data
        update_script = scripts_dir / "update_data.py"
        if update_script.exists():
            subprocess.run(
                [sys.executable, str(update_script)],
                check=True, capture_output=True, timeout=300
            )
        # Step 2: retrain
        subprocess.run(
            [sys.executable, str(scripts_dir / "train.py")],
            check=True, capture_output=True, timeout=1800
        )
        # Step 3: clear LRU cache so new models are picked up
        predictor._load_artefacts.cache_clear()
        _retrain_status["message"] = "Update complete ✓"
    except subprocess.TimeoutExpired:
        _retrain_status["message"] = "Update timed out"
    except subprocess.CalledProcessError as exc:
        _retrain_status["message"] = f"Update failed: {exc.stderr.decode()[:200]}"
    except Exception as exc:
        _retrain_status["message"] = f"Update error: {str(exc)}"
    finally:
        _retrain_status["running"]  = False
        _retrain_status["last_run"] = datetime.utcnow().isoformat() + "Z"


@app.route("/api/update-data", methods=["POST"])
def api_update_data():
    if _retrain_status["running"]:
        return _ok({"message": "Update already in progress", "status": _retrain_status})
    t = threading.Thread(target=_run_update, daemon=True)
    t.start()
    return _ok({"message": "Update started in background", "status": _retrain_status})


@app.route("/api/model-status")
def api_model_status():
    meta_path = MODELS_DIR / "metadata.json"
    if not meta_path.exists():
        return _err("Models not trained yet. Run scripts/train.py first.", 503)

    with open(meta_path) as f:
        meta = json.load(f)

    stat = os.stat(meta_path)
    return _ok({
        "last_trained":    datetime.utcfromtimestamp(stat.st_mtime).isoformat() + "Z",
        "team_count":      len(meta.get("teams", [])),
        "division_count":  len(meta.get("divisions", [])),
        "outcome_model":   meta.get("outcome_model", "unknown"),
        "retrain_status":  _retrain_status,
    })


# ─────────────────────────────────────────────────────────────────────────────


@app.route("/api/today")
def api_today():
    """
    Returns today's fixtures scraped from FBRef.com.
    No API key required — completely free.
    Falls back to yesterday's results if no fixtures today.
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

    from datetime import date as _date

    try:
        from scrape_today import get_today_matches
    except ImportError:
        return _ok({"matches": [], "date": str(_date.today()),
                    "message": "scrape_today module not found"})

    try:
        all_teams = predictor.get_teams()
    except Exception:
        all_teams = []

    try:
        raw_matches, match_date = get_today_matches()
    except Exception as e:
        return _ok({"matches": [], "date": str(_date.today()),
                    "message": f"Scrape failed: {str(e)}"})

    matches = []
    for m in raw_matches:
        norm_home = normalise(m["home_team"], all_teams)
        norm_away = normalise(m["away_team"], all_teams)
        m["home_team_display"] = m["home_team"]  # keep original for display
        m["away_team_display"] = m["away_team"]
        m["home_team"] = norm_home               # use normalised for predict
        m["away_team"] = norm_away
        m["can_predict"] = (
            norm_home in all_teams and
            norm_away in all_teams and
            bool(m["division"])
        )
        matches.append(m)

    return _ok({
        "matches": matches,
        "date":    match_date,
        "count":   len(matches),
        "source":  "fbref.com"
    })




# ── Prediction History ─────────────────────────────────────────────────────────

@app.route("/api/history")
def api_history():
    limit    = int(request.args.get("limit", 100))
    division = request.args.get("division", None)
    data     = history_manager.get_history(limit=limit, division=division)
    return _ok(data)


@app.route("/api/history/update", methods=["POST"])
def api_history_update():
    body = request.get_json(force=True, silent=True) or {}
    record_id           = body.get("id")
    actual_home         = body.get("actual_home")
    actual_away         = body.get("actual_away")
    actual_home_corners = body.get("actual_home_corners")
    actual_away_corners = body.get("actual_away_corners")
    if record_id is None or actual_home is None or actual_away is None:
        return _err("Required: id, actual_home, actual_away")
    updated = history_manager.update_result(
        int(record_id), int(actual_home), int(actual_away),
        actual_home_corners, actual_away_corners
    )
    return _ok({"record": updated})


@app.route("/api/history/auto-check", methods=["POST"])
def api_history_auto_check():
    n = history_manager.check_results_from_fbref()
    return _ok({"resolved": n})


@app.route("/api/history/clear", methods=["POST"])
def api_history_clear():
    history_manager._save([])
    return _ok({"message": "History cleared"})


if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
