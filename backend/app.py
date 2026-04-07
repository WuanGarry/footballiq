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

    teams = predictor.get_teams()
    if home_team not in teams:
        return _err(f"Unknown home team: '{home_team}'")
    if away_team not in teams:
        return _err(f"Unknown away team: '{away_team}'")

    try:
        result = predictor.predict(home_team, away_team, division)
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
    Returns today's scheduled matches from football-data.org free API.
    Falls back to tomorrow/yesterday if nothing found today.
    Requires FOOTBALL_DATA_ORG_KEY env var (free key).
    Without a key, returns an empty list gracefully.
    """
    import requests as req
    from datetime import datetime, timedelta

    api_key = os.environ.get("FOOTBALL_DATA_ORG_KEY", "")
    if not api_key:
        return _ok({
            "matches": [],
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "message": "Set FOOTBALL_DATA_ORG_KEY for live fixtures"
        })

    # Our division code → football-data.org competition code
    COMP_CODES = {
        "E0":"PL","E1":"ELC","SP1":"PD","D1":"BL1","I1":"SA",
        "F1":"FL1","N1":"DED","P1":"PPL","BSA":"BSA",
        "UCL":"CL","UEL":"EL","UECL":"UCL",
    }
    REV_CODES = {v: k for k, v in COMP_CODES.items()}

    hdrs = {"X-Auth-Token": api_key}
    today = datetime.utcnow().strftime("%Y-%m-%d")

    try:
        r = req.get(
            "https://api.football-data.org/v4/matches",
            headers=hdrs,
            params={"dateFrom": today, "dateTo": today},
            timeout=15
        )
        if r.status_code == 403:
            return _ok({"matches": [], "date": today,
                        "message": "API key may need higher tier for fixtures"})
        r.raise_for_status()
        raw = r.json().get("matches", [])
    except Exception as e:
        return _ok({"matches": [], "date": today, "message": str(e)})

    try:
        all_teams = predictor.get_teams()
    except Exception:
        all_teams = []

    matches = []
    for m in raw:
        try:
            status     = m.get("status", "")
            comp_code  = m.get("competition", {}).get("code", "")
            our_div    = REV_CODES.get(comp_code, comp_code)
            home_name  = m["homeTeam"]["name"]
            away_name  = m["awayTeam"]["name"]
            utc_dt     = m.get("utcDate", "")
            kick_off   = utc_dt[11:16] if len(utc_dt) >= 16 else ""

            # Score if already played / in progress
            score      = m.get("score", {})
            ft         = score.get("fullTime", {})
            score_str  = (f"{ft.get('home','?')}-{ft.get('away','?')}"
                          if status == "FINISHED" else "")

            # Check if we can predict this match
            can_predict = (home_name in all_teams and away_name in all_teams)

            matches.append({
                "home_team":   home_name,
                "away_team":   away_name,
                "division":    our_div,
                "competition": m.get("competition", {}).get("name", comp_code),
                "kick_off":    kick_off,
                "status":      status,
                "score":       score_str,
                "can_predict": can_predict,
            })
        except Exception:
            continue

    return _ok({
        "matches": matches,
        "date":    today,
        "count":   len(matches)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
