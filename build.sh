#!/usr/bin/env bash
set -e

echo "============================================"
echo "  FootballIQ  -  Build Script"
echo "============================================"

echo ""
echo "[1/6] Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "[2/6] Downloading domestic league data (football-data.co.uk)..."
python scripts/fetch_data.py

echo ""
echo "[3/6] Downloading extra leagues (FBRef: Ghana, Nigeria, Africa, Asia...)..."
python scripts/fetch_fbref_extra.py

echo ""
echo "[4/6] Seeding UEFA + Copa Libertadores + CONCACAF data..."
python scripts/seed_uefa.py

echo ""
echo "[5/6] Building features..."
python scripts/build_features.py

echo ""
echo "[6/6] Training models..."
python scripts/train.py

echo ""
echo "============================================"
echo "  Build complete ✓"
echo "============================================"
