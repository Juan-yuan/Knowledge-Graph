#!/usr/bin/env bash
# Start FastAPI + Streamlit together. Ctrl+C stops both.
# Usage: bash start.sh
# Override: FASTAPI_HOST / FASTAPI_PORT / STREAMLIT_PORT / API_BASE_URL / LOG_LEVEL
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Load common/.env if present (does not override vars already set in the shell)
if [[ -f "$PROJECT_ROOT/common/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/common/.env"
  set +a
fi

FASTAPI_HOST="${FASTAPI_HOST:-0.0.0.0}"
FASTAPI_PORT="${FASTAPI_PORT:-8000}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
API_BASE_URL="${API_BASE_URL:-http://localhost:${FASTAPI_PORT}}"
HEALTH_URL="${API_BASE_URL%/}/"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

API_PID=""
UI_PID=""

cleanup() {
  trap - EXIT INT TERM
  echo ""
  echo "Stopping services..."
  if [[ -n "$UI_PID" ]] && kill -0 "$UI_PID" 2>/dev/null; then
    kill "$UI_PID" 2>/dev/null || true
    wait "$UI_PID" 2>/dev/null || true
  fi
  if [[ -n "$API_PID" ]] && kill -0 "$API_PID" 2>/dev/null; then
    kill "$API_PID" 2>/dev/null || true
    wait "$API_PID" 2>/dev/null || true
  fi
  echo "Stopped."
}

trap cleanup EXIT INT TERM

echo "Starting FastAPI on http://localhost:${FASTAPI_PORT} ..."
python fastapi/main.py >>"$LOG_DIR/fastapi.console.log" 2>&1 &
API_PID=$!

echo "Waiting for API health at ${HEALTH_URL} ..."
for _ in $(seq 1 90); do
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "FastAPI exited early. See $LOG_DIR/fastapi.log and $LOG_DIR/fastapi.console.log"
    exit 1
  fi
  if curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
    echo "API is up."
    break
  fi
  sleep 1
done

if ! curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
  echo "Timed out waiting for FastAPI. See $LOG_DIR/fastapi.log and $LOG_DIR/fastapi.console.log"
  exit 1
fi

echo "Starting Streamlit on http://localhost:${STREAMLIT_PORT} ..."
echo "  UI:  http://localhost:${STREAMLIT_PORT}"
echo "  API: http://localhost:${FASTAPI_PORT}  (docs: /docs)"
echo "Ctrl+C to stop both."

API_BASE_URL="$API_BASE_URL" python -m streamlit run streamlit/app.py \
  --server.port "$STREAMLIT_PORT" \
  --server.headless true \
  --browser.gatherUsageStats false &
UI_PID=$!

# Exit if either process dies
while true; do
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "FastAPI stopped unexpectedly."
    exit 1
  fi
  if ! kill -0 "$UI_PID" 2>/dev/null; then
    echo "Streamlit stopped unexpectedly."
    exit 1
  fi
  sleep 2
done
