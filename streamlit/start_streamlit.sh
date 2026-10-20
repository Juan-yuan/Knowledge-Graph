#!/usr/bin/env bash
# Start Streamlit only. Prefer project root: bash start.sh (API + UI together).
# Usage: bash streamlit/start_streamlit.sh   (override with STREAMLIT_PORT / API_BASE_URL / LOG_LEVEL)
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PORT="${STREAMLIT_PORT:-8501}"
echo "Starting Streamlit on http://localhost:${PORT} (logs: ${PROJECT_ROOT}/logs/streamlit.log)"

exec python -m streamlit run streamlit/app.py \
  --server.port "$PORT" \
  --server.headless true \
  --browser.gatherUsageStats false
