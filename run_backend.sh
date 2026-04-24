#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR/backend"

if [[ ! -d ".venv" ]]; then
  echo "Missing backend virtual environment at backend/.venv"
  echo "Create it with: python3.10 -m venv backend/.venv"
  exit 1
fi

source .venv/bin/activate
exec uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
