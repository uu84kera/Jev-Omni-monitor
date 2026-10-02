#!/usr/bin/env bash
#SBATCH --job-name=jev-qwen
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gpus-per-task=1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}}"
VENV_PATH="${VENV_PATH:-$PROJECT_DIR/.venv}"
: "${DATA_DIR:?Set DATA_DIR to the directory containing raw/}"
: "${RESULT_DIR:?Set RESULT_DIR to a completed Jev screening result directory}"

QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen2.5-Omni-7B}"
QWEN_MAX_CLIP_SECONDS="${QWEN_MAX_CLIP_SECONDS:-8.0}"
QWEN_OUTPUT_DIR="${QWEN_OUTPUT_DIR:-$RESULT_DIR/qwen-analysis}"

cd "$PROJECT_DIR"
source "$VENV_PATH/bin/activate"

export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME/.cache}/huggingface}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_THREAD_LIMIT=1

nvidia-smi
jev-monitor-qwen \
    --incidents-dir "$RESULT_DIR/incidents" \
    --video-dir "$DATA_DIR/raw" \
    --output-dir "$QWEN_OUTPUT_DIR" \
    --model-id "$QWEN_MODEL_ID" \
    --max-clip-seconds "$QWEN_MAX_CLIP_SECONDS"

echo "Qwen results: $QWEN_OUTPUT_DIR"
