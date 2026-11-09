#!/usr/bin/env bash
set -e

echo "======================================================"
echo "  Sign Language Biomechanical Engine - Linux Setup"
echo "======================================================"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [ ! -f ".venv/bin/python" ]; then
    echo "[*] Creating virtual environment (.venv)..."
    python3 -m venv .venv
    echo "[*] Installing dependencies..."
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
    .venv/bin/pip install -e .
fi

echo "[*] Launching Real-Time Live Inspector..."
.venv/bin/python tools/live_inspector.py "$@"

