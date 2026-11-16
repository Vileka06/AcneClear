#!/bin/bash
set -e

# AcneClear AI one-click local starter for macOS.
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo ""
echo "========================================"
echo "        ACNECLEAR AI - START"
echo "========================================"
echo ""

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is missing. Install Python 3 and run this file again."
  read -r -p "Press Enter to close..."
  exit 1
fi

if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  echo "Node.js/npm is missing. Install the LTS version from https://nodejs.org"
  read -r -p "Press Enter to close..."
  exit 1
fi

echo "1/4 Preparing Python environment..."
if [ ! -d "$ROOT/backend/.venv" ]; then
  python3 -m venv "$ROOT/backend/.venv"
fi
source "$ROOT/backend/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$ROOT/backend/requirements.txt"

echo ""
echo "2/4 Installing website packages..."
cd "$ROOT/frontend/website"
npm config set fetch-retries 5
npm config set fetch-timeout 120000
npm install

echo ""
echo "3/4 Starting backend..."
if lsof -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Backend already running on port 8000."
else
  nohup "$ROOT/backend/.venv/bin/uvicorn" main:app --app-dir "$ROOT/backend" --host 0.0.0.0 --port 8000 > "$ROOT/backend/backend.log" 2>&1 &
fi

echo "4/4 Starting website..."
if lsof -iTCP:5173 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Website already running on port 5173."
else
  nohup npm run dev -- --host 0.0.0.0 > "$ROOT/frontend/website/website.log" 2>&1 &
fi

sleep 3
open "http://localhost:5173"
echo ""
echo "AcneClear website: http://localhost:5173"
echo "Backend API:       http://localhost:8000/docs"
echo ""
echo "If a service does not start, check backend/backend.log or frontend/website/website.log."
echo "You can close this window; the services run in the background."
read -r -p "Press Enter to close..."
