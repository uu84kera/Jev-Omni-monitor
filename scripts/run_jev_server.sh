#!/usr/bin/env bash
set -euo pipefail

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

HOST="${JEV_SERVER_HOST:-127.0.0.1}"
PORT="${JEV_SERVER_PORT:-8100}"

exec python -m uvicorn jev_monitor.jev_server:app --host "$HOST" --port "$PORT"
