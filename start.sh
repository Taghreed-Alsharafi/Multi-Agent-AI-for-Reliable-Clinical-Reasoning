#!/usr/bin/env bash
# Single entry point: installs anything missing, then starts the backend,
# the frontend, and the browser.
#   ./start.sh          - default port 5173
#   ./start.sh 5199     - run the frontend on another port
set -uo pipefail
cd "$(dirname "$0")"

BACKEND_PORT=8000
FRONTEND_PORT="${1:-5173}"

echo "============================================================"
echo "  Multi-Agent Medical Assessment"
echo "============================================================"
echo

fail() { echo; echo "ERROR: $*"; exit 1; }

# Vite binds IPv6 [::1] only, so probe localhost rather than 127.0.0.1.
responds() { curl -s -m 2 -o /dev/null "http://localhost:$1" 2>/dev/null; }

# ── 1. Prerequisites ────────────────────────────────────────────
command -v python3 >/dev/null 2>&1 \
  || fail "Python 3 not found. Install 3.11+ from https://www.python.org/downloads/"
python3 - <<'PY' || fail "Python 3.11 or newer is required."
import sys
sys.exit(0 if sys.version_info >= (3, 11) else 1)
PY
command -v node >/dev/null 2>&1 \
  || fail "Node.js not found. Install the LTS build from https://nodejs.org/"

# ── 2. Install anything missing (skipped once installed) ────────
if ! python3 -c "import fastapi, uvicorn, openai, pydantic_settings" >/dev/null 2>&1; then
  echo "First run - installing Python packages, this takes a minute..."
  python3 -m pip install --upgrade pip --quiet
  python3 -m pip install -e ".[dev]" --quiet \
    || fail "Python package install failed. Run 'python3 -m pip install -e \".[dev]\"' to see why."
  echo "      Done."
fi

if [ ! -d frontend/node_modules ]; then
  echo "First run - installing frontend packages, this takes a few minutes..."
  (cd frontend && npm install --silent) \
    || fail "npm install failed. Run 'npm install' in the frontend folder to see why."
  echo "      Done."
fi

# ── 3. Configuration ────────────────────────────────────────────
if [ ! -f .env ]; then
  cp .env.example .env
  echo
  echo "============================================================"
  echo "  Almost there - your API key is needed."
  echo "============================================================"
  echo
  echo "  A new .env file was just created in this folder. Replace:"
  echo
  echo "      OPENAI_API_KEY=sk-your-key-here"
  echo
  echo "  with your real OpenAI key, save it, then run ./start.sh again."
  echo
  exit 1
fi

if grep -q "OPENAI_API_KEY=sk-your-key-here" .env; then
  fail "The .env file still contains the placeholder API key. Replace it with your real key, then run ./start.sh again."
fi

# ── 4. Ports ────────────────────────────────────────────────────
# Vite would otherwise silently move to the next free port and the browser
# would open on whatever is already there - which looks like this app failing.
if responds "$FRONTEND_PORT"; then
  echo "ERROR: Port $FRONTEND_PORT is already in use by another program."
  echo
  echo "       If that is a different copy of this project, close it first,"
  echo "       or start this one on its own port:"
  echo
  echo "           ./start.sh 5199"
  exit 1
fi

PIDS=()
cleanup() {
  echo
  echo "Shutting down..."
  for pid in "${PIDS[@]:-}"; do
    [ -n "$pid" ] && kill "$pid" 2>/dev/null
  done
}
trap cleanup EXIT INT TERM

# ── 5. Backend ──────────────────────────────────────────────────
if curl -s -m 2 -o /dev/null "http://localhost:$BACKEND_PORT/health" 2>/dev/null; then
  echo "[1/3] Backend already running on port $BACKEND_PORT - reusing it."
else
  echo "[1/3] Starting backend on port $BACKEND_PORT..."
  python3 -m uvicorn api.main:app --reload --port "$BACKEND_PORT" &
  PIDS+=($!)
  for i in $(seq 1 20); do
    sleep 2
    curl -s -m 2 -o /dev/null "http://localhost:$BACKEND_PORT/health" 2>/dev/null && break
    [ "$i" -eq 20 ] && fail "The backend did not start. See the output above."
  done
  echo "      Backend is up."
fi

# ── 6. Frontend ─────────────────────────────────────────────────
echo "[2/3] Starting frontend on port $FRONTEND_PORT..."
# --strictPort so it fails loudly instead of drifting to another port.
(cd frontend && npm run dev -- --port "$FRONTEND_PORT" --strictPort) &
PIDS+=($!)
for i in $(seq 1 30); do
  sleep 2
  responds "$FRONTEND_PORT" && break
  [ "$i" -eq 30 ] && fail "The frontend did not start. See the output above."
done
echo "      Frontend is up."

# ── 7. Browser ──────────────────────────────────────────────────
echo "[3/3] Opening the browser..."
URL="http://localhost:$FRONTEND_PORT"
if command -v open >/dev/null 2>&1; then open "$URL"
elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL"
else echo "      Open $URL yourself."
fi

echo
echo "============================================================"
echo "  Running."
echo "============================================================"
echo
echo "  App:      $URL"
echo "  API docs: http://localhost:$BACKEND_PORT/docs"
echo
echo "  Press Ctrl+C to stop both."
echo

wait
