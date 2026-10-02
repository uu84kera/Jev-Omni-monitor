#!/usr/bin/env bash
#SBATCH --job-name=jev-dataset
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gpus-per-task=1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}}"
VENV_PATH="${VENV_PATH:-$PROJECT_DIR/.venv}"
: "${DATA_DIR:?Set DATA_DIR to the directory containing raw/ and annotations.csv}"

ANNOTATIONS_PATH="${ANNOTATIONS_PATH:-$DATA_DIR/annotations.csv}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.5}"
SUSPICIOUS_THRESHOLD="${SUSPICIOUS_THRESHOLD:-0.70}"
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
export JEV_MONITOR_VLM_PROVIDER="${JEV_MONITOR_VLM_PROVIDER:-mock}"
export JEV_MONITOR_SUSPICIOUS_THRESHOLD="$SUSPICIOUS_THRESHOLD"
export JEV_MONITOR_ANALYSIS_MODE="${JEV_MONITOR_ANALYSIS_MODE:-deferred}"

mkdir -p \
    "$RESULT_DIR/events" \
    "$RESULT_DIR/summaries" \
    "$RESULT_DIR/evaluation" \
    "$RESULT_DIR/incidents"

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

mapfile -t VIDEOS < <(find "$DATA_DIR/raw" -maxdepth 1 -type f -name '*.mp4' -print | sort)
if [[ "${#VIDEOS[@]}" -eq 0 ]]; then
    echo "No MP4 videos found in $DATA_DIR/raw" >&2
    exit 1
fi

echo "Jev-Omni loaded once; processing ${#VIDEOS[@]} videos."
for video_path in "${VIDEOS[@]}"; do
    video_id="$(basename "$video_path" .mp4)"
    export JEV_MONITOR_EVENT_LOG="$RESULT_DIR/events/${video_id}.jsonl"
    export JEV_MONITOR_INCIDENT_LOG="$RESULT_DIR/incidents/${video_id}.jsonl"
    echo "Processing $video_id"
    jev-monitor run-video \
        "$video_path" \
        --camera-id "$video_id" \
        --sample-interval "$SAMPLE_INTERVAL" \
        >"$RESULT_DIR/summaries/${video_id}.json"
done

jev-monitor-summarize-incidents \
    --incidents-dir "$RESULT_DIR/incidents" \
    --output "$RESULT_DIR/incidents/summary.json" \
    | tee "$RESULT_DIR/incidents/report.txt"

python -m jev_monitor.evaluation \
    --summaries-dir "$RESULT_DIR/summaries" \
    --annotations "$ANNOTATIONS_PATH" \
    --output-dir "$RESULT_DIR/evaluation" \
    --threshold "$SUSPICIOUS_THRESHOLD" \
    | tee "$RESULT_DIR/evaluation/report.txt"

echo "Results: $RESULT_DIR"
