#!/usr/bin/env bash
set -e

echo "============================================"
echo "  FootballIQ  -  Build Script"
echo "============================================"

echo ""
echo "[1/5] Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "[2/5] Downloading latest match data..."
python scripts/fetch_data.py

echo ""
echo "[3/5] Seeding UEFA competitions (UCL/UEL/UECL)..."
python scripts/seed_uefa.py

echo ""
echo "[4/5] Building features..."
python scripts/build_features.py

echo ""
echo "[5/5] Training models..."
python scripts/train.py

echo ""
echo "============================================"
echo "  Build complete ✓"
echo "============================================"
