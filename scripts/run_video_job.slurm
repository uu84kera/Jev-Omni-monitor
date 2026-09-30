#!/usr/bin/env bash
#SBATCH --job-name=jev-video
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gpus-per-task=a100:1
#SBATCH --constraint=a100-80gb
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
VENV_PATH="${VENV_PATH:-$PROJECT_DIR/.venv}"
if [[ -z "${VIDEO_PATH:-}" ]]; then
    : "${DATA_DIR:?Set DATA_DIR or VIDEO_PATH before submitting the job}"
    VIDEO_PATH="$DATA_DIR/raw/smartbench_0014.mp4"
fi
CAMERA_ID="${CAMERA_ID:-$(basename "$VIDEO_PATH" .mp4)}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RESULT_DIR="$PROJECT_DIR/results/$RUN_ID"

cd "$PROJECT_DIR"
source "$VENV_PATH/bin/activate"

export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME/.cache}/huggingface}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export JEV_SERVER_HOST=127.0.0.1
export JEV_SERVER_PORT=8100
export JEV_MONITOR_JEV_PROVIDER=http
export JEV_MONITOR_JEV_ENDPOINT="http://${JEV_SERVER_HOST}:${JEV_SERVER_PORT}/v1/classify"
export JEV_MONITOR_VLM_PROVIDER=mock
export JEV_MONITOR_EVENT_LOG="$RESULT_DIR/events/${CAMERA_ID}.jsonl"

mkdir -p "$RESULT_DIR/events" "$RESULT_DIR/summaries"

scripts/run_jev_server.sh >"$RESULT_DIR/jev-server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 240); do
    if curl --fail --silent "http://${JEV_SERVER_HOST}:${JEV_SERVER_PORT}/health" >/dev/null; then
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        cat "$RESULT_DIR/jev-server.log"
        exit 1
    fi
    sleep 5
done

curl --fail --silent "http://${JEV_SERVER_HOST}:${JEV_SERVER_PORT}/health" >/dev/null || {
    echo "Jev server did not become ready within 20 minutes" >&2
    exit 1
}

jev-monitor run-video \
    "$VIDEO_PATH" \
    --camera-id "$CAMERA_ID" \
    --sample-interval 0.5 \
    >"$RESULT_DIR/summaries/${CAMERA_ID}.json"

cat "$RESULT_DIR/summaries/${CAMERA_ID}.json"
