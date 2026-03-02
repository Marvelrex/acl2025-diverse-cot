#!/bin/bash
#SBATCH --job-name=sgft_baseline_direct
#SBATCH --account=def-sdrew
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --output=logs/sgft_baseline_direct_%j.out
#SBATCH --error=logs/sgft_baseline_direct_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT_ROOT"
mkdir -p logs

module load StdEnv/2023
module load python/3.11
module load gcc arrow/21.0.0

VENV_ACTIVATE="${VENV_ACTIVATE:-/home/jyang001/jyang001/projects/envs/quin/bin/activate}"
source "$VENV_ACTIVATE"

DATASET_NAME=""
MODEL_NAME=""
TOKENIZER_NAME=""
TEST_FILE=""
RUN_ID=""
OUTPUT_ROOT=""
PRED_ROOT=""
PRED_FILE=""
PRED_NUM_COTS=3
PRED_MAX_NEW_TOKENS=2048
PRED_TORCH_DTYPE="bfloat16"

usage() {
  echo "Usage: sbatch SGFT_Baseline_Direct.slurm --dataset-name aqua|gsm8k|strategyqa --model-name HF_ID --run-id N [options]"
  echo "Options:"
  echo "  --tokenizer-name HF_ID"
  echo "  --test-file PATH"
  echo "  --output-root PATH"
  echo "  --pred-root PATH"
  echo "  --pred-file PATH"
  echo "  --pred-num-cots N"
  echo "  --pred-max-new-tokens N"
  echo "  --torch-dtype auto|bfloat16|float16|float32"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-name|--dataset)
      DATASET_NAME="$2"; shift 2 ;;
    --model-name|--base-model-path)
      MODEL_NAME="$2"; shift 2 ;;
    --tokenizer-name)
      TOKENIZER_NAME="$2"; shift 2 ;;
    --test-file)
      TEST_FILE="$2"; shift 2 ;;
    --run-id)
      RUN_ID="$2"; shift 2 ;;
    --output-root)
      OUTPUT_ROOT="$2"; shift 2 ;;
    --pred-root)
      PRED_ROOT="$2"; shift 2 ;;
    --pred-file)
      PRED_FILE="$2"; shift 2 ;;
    --pred-num-cots)
      PRED_NUM_COTS="$2"; shift 2 ;;
    --pred-max-new-tokens)
      PRED_MAX_NEW_TOKENS="$2"; shift 2 ;;
    --torch-dtype)
      PRED_TORCH_DTYPE="$2"; shift 2 ;;
    --help|-h)
      usage
      exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

if [[ -z "$DATASET_NAME" ]]; then
  echo "Set --dataset-name." >&2
  exit 2
fi
if [[ -z "$MODEL_NAME" ]]; then
  echo "Set --model-name." >&2
  exit 2
fi
if [[ -z "$RUN_ID" ]]; then
  echo "Set --run-id." >&2
  exit 2
fi
if [[ -z "$TOKENIZER_NAME" ]]; then
  TOKENIZER_NAME="$MODEL_NAME"
fi

case "${DATASET_NAME,,}" in
  aqua)
    DATASET_CANON="AQUA"
    CANDIDATE_TEST_FILES=(
      "$PROJECT_ROOT/Baseline/data/AQUA/test.jsonl"
      "$PROJECT_ROOT/data/aqua/test.json"
      "$PROJECT_ROOT/data/AQUA/test.json"
    )
    ;;
  gsm8k)
    DATASET_CANON="GSM8K"
    CANDIDATE_TEST_FILES=(
      "$PROJECT_ROOT/Baseline/data/GSM8K/test.jsonl"
      "$PROJECT_ROOT/data/gsm8k/test.json"
      "$PROJECT_ROOT/data/GSM8K/test.json"
    )
    ;;
  strategyqa)
    DATASET_CANON="StrategyQA"
    CANDIDATE_TEST_FILES=(
      "$PROJECT_ROOT/Baseline/data/StrategyQA/test.jsonl"
      "$PROJECT_ROOT/data/strategyqa/test.json"
      "$PROJECT_ROOT/data/StrategyQA/test.json"
    )
    ;;
  *)
    echo "Unsupported dataset-name: $DATASET_NAME" >&2
    exit 2
    ;;
esac

if ! [[ "$RUN_ID" =~ ^[0-9]+$ ]]; then
  echo "--run-id must be an integer." >&2
  exit 2
fi

RUN_TAG="$(printf "run_%02d" "$RUN_ID")"
MODEL_TAG="${MODEL_NAME//\//_}"

if [[ -z "$TEST_FILE" ]]; then
  for tf in "${CANDIDATE_TEST_FILES[@]}"; do
    if [[ -f "$tf" ]]; then
      TEST_FILE="$tf"
      break
    fi
  done
fi

if [[ -z "$TEST_FILE" ]]; then
  echo "Could not locate a test file. Pass --test-file." >&2
  exit 3
fi

if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="/home/jyang001/scratch/DCoT/base_model_direct/${MODEL_TAG}/${RUN_TAG}/${DATASET_CANON}"
fi
if [[ -z "$PRED_ROOT" ]]; then
  PRED_ROOT="${OUTPUT_ROOT%/}/predictions"
fi
mkdir -p "$PRED_ROOT"

if [[ -z "$PRED_FILE" ]]; then
  PRED_FILE="$PRED_ROOT/${RUN_TAG}__${DATASET_CANON}__base__${MODEL_TAG}__predictions__${DATASET_CANON}_test_predictions.jsonl"
fi

export LLM_QUERY_SEED=42
export PYTHONHASHSEED="$LLM_QUERY_SEED"

PRED_ARGS=(
  --dataset "$DATASET_CANON"
  --test-file "$TEST_FILE"
  --output-file "$PRED_FILE"
  --sft-type full
  --model-path "$MODEL_NAME"
  --tokenizer-name "$TOKENIZER_NAME"
  --num-cots "$PRED_NUM_COTS"
  --max-new-tokens "$PRED_MAX_NEW_TOKENS"
  --no-do-sample
  --torch-dtype "$PRED_TORCH_DTYPE"
)

echo "[RUN] dataset=${DATASET_CANON}"
echo "[RUN] model=${MODEL_NAME}"
echo "[RUN] run_id=${RUN_ID}"
echo "[RUN] llm_query_seed=${LLM_QUERY_SEED}"
echo "[RUN] do_sample=false"
echo "[RUN] test_file=${TEST_FILE}"
echo "[RUN] pred_file=${PRED_FILE}"

python "$PROJECT_ROOT/scripts/predict_dcot_test.py" "${PRED_ARGS[@]}"
