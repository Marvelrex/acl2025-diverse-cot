#!/bin/bash
set -euo pipefail

if [[ -z "${DATASET_NAME:-}" ]]; then
  echo "DATASET_NAME is required (AQUA|GSM8K|StrategyQA)." >&2
  exit 2
fi
if [[ -z "${SFT_TYPE:-}" ]]; then
  echo "SFT_TYPE is required (lora|full)." >&2
  exit 2
fi

module load StdEnv/2023
module load python/3.11
module load gcc arrow/21.0.0

VENV_ACTIVATE="${VENV_ACTIVATE:-/home/jyang001/jyang001/projects/envs/quin/bin/activate}"
source "$VENV_ACTIVATE"

PROJECT_ROOT="${PROJECT_ROOT:-/home/jyang001/jyang001/projects/BRIDGE}"
cd "$PROJECT_ROOT"
mkdir -p logs

DATA_FILE=""
STUDENT_MODEL="qwen"
MODEL_NAME=""
TOKENIZER_NAME=""
OUTPUT_ROOT=""
TRAIN_SCRIPT="${TRAIN_SCRIPT:-}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-file)
      DATA_FILE="$2"; shift 2 ;;
    --student-model)
      STUDENT_MODEL="$2"; shift 2 ;;
    --model-name)
      MODEL_NAME="$2"; shift 2 ;;
    --tokenizer-name)
      TOKENIZER_NAME="$2"; shift 2 ;;
    --output-root)
      OUTPUT_ROOT="$2"; shift 2 ;;
    --help|-h)
      echo "Usage: sbatch <DATASET>_DCoT_<LoRA|Full_SFT>.slurm [--student-model qwen|llama] [--model-name HF_ID] [--tokenizer-name HF_ID] [--data-file PATH] [--output-root PATH] [extra training_script args...]"
      exit 0 ;;
    *)
      EXTRA_ARGS+=("$1"); shift 1 ;;
  esac
done

MODEL_KEY="$(echo "$STUDENT_MODEL" | tr '[:upper:]' '[:lower:]')"
case "$MODEL_KEY" in
  qwen|qwen3|qwen-3|qwen3-1.7b|qwen-3-1.7b|qwen3-1.7b-instruct)
    DEFAULT_MODEL="Qwen/Qwen3-1.7B" ;;
  llama|llama3|llama-3.1|llama-3.1-8b|llama-3.1-8b-instruct)
    DEFAULT_MODEL="meta-llama/Llama-3.1-8B-Instruct" ;;
  *)
    echo "Unknown --student-model: $STUDENT_MODEL. Use qwen or llama, or pass --model-name directly." >&2
    exit 2 ;;
esac

if [[ -z "$MODEL_NAME" ]]; then
  MODEL_NAME="$DEFAULT_MODEL"
fi
if [[ -z "$TOKENIZER_NAME" ]]; then
  TOKENIZER_NAME="$MODEL_NAME"
fi

if [[ -z "$TRAIN_SCRIPT" ]]; then
  if [[ -f "$PROJECT_ROOT/training_script.py" ]]; then
    TRAIN_SCRIPT="$PROJECT_ROOT/training_script.py"
  elif [[ -f "$PROJECT_ROOT/Baseline/DCot/training_script.py" ]]; then
    TRAIN_SCRIPT="$PROJECT_ROOT/Baseline/DCot/training_script.py"
  elif [[ -f "$PROJECT_ROOT/Baselines/DCot/training_script.py" ]]; then
    TRAIN_SCRIPT="$PROJECT_ROOT/Baselines/DCot/training_script.py"
  else
    echo "Cannot locate training_script.py. Set TRAIN_SCRIPT env var." >&2
    exit 3
  fi
fi

if [[ -z "$DATA_FILE" ]]; then
  case "$DATASET_NAME" in
    AQUA)
      CANDIDATES=(
        "$PROJECT_ROOT/data/AQUA/cot_dataset.json"
        "$PROJECT_ROOT/data/aqua/cot_dataset.json"
        "$PROJECT_ROOT/data/AQUA/DCoT/results_dcot.jsonl"
        "$PROJECT_ROOT/data/AQUA/results_dcot.jsonl"
      )
      ;;
    GSM8K)
      CANDIDATES=(
        "$PROJECT_ROOT/data/GSM8K/cot_dataset.json"
        "$PROJECT_ROOT/data/gsm8k/cot_dataset.json"
        "$PROJECT_ROOT/data/GSM8K/DCoT/results_dcot.jsonl"
        "$PROJECT_ROOT/data/GSM8K/results_dcot.jsonl"
      )
      ;;
    StrategyQA)
      CANDIDATES=(
        "$PROJECT_ROOT/data/StrategyQA/cot_dataset.json"
        "$PROJECT_ROOT/data/strategyqa/cot_dataset.json"
        "$PROJECT_ROOT/data/StrategyQA/DCoT/results_dcot.jsonl"
        "$PROJECT_ROOT/data/StrategyQA/results_dcot.jsonl"
      )
      ;;
    *)
      echo "Unsupported dataset: $DATASET_NAME" >&2
      exit 2
      ;;
  esac

  for candidate in "${CANDIDATES[@]}"; do
    if [[ -f "$candidate" ]]; then
      DATA_FILE="$candidate"
      break
    fi
  done
fi

if [[ -z "$DATA_FILE" ]]; then
  echo "Could not auto-locate data file for ${DATASET_NAME}. Please pass --data-file." >&2
  exit 4
fi

if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="/home/jyang001/scratch/${DATASET_NAME}_dcot_${SFT_TYPE}"
fi

MODEL_TAG="${MODEL_NAME//\//_}"
OUTPUT_PATH="${OUTPUT_ROOT%/}/${MODEL_TAG}"
mkdir -p "$OUTPUT_PATH"

COMMON_ARGS=(
  --train
  --train_path "$DATA_FILE"
  --base_model_path "$MODEL_NAME"
  --tokenizer_name "$TOKENIZER_NAME"
  --output_path "$OUTPUT_PATH"
  --dcot
  --max_length 2048
  --learning_rate 1e-4
  --weight_decay 0.01
  --num_epochs 2
  --batch_size 2
  --grad_accum 8
  --warmup_ratio 0.03
  --logging_strategy steps
  --logging_steps 25
  --save_steps 250
  --optim adamw_torch
  --bf16
  --lr_scheduler_type linear
  --max_grad_norm 1.0
  --seed 42
)

if [[ "$SFT_TYPE" == "lora" ]]; then
  SFT_ARGS=(
    --sft_type lora
    --lora_r 64
    --lora_alpha 128
    --lora_dropout 0.05
    --lora_target_modules q_proj,k_proj,v_proj,o_proj
  )
elif [[ "$SFT_TYPE" == "full" ]]; then
  SFT_ARGS=(
    --sft_type full
    --no-load_in_8bit
  )
else
  echo "Unsupported SFT_TYPE: $SFT_TYPE" >&2
  exit 2
fi

echo "[RUN] dataset=${DATASET_NAME} sft=${SFT_TYPE}"
echo "[RUN] model=${MODEL_NAME}"
echo "[RUN] tokenizer=${TOKENIZER_NAME}"
echo "[RUN] data_file=${DATA_FILE}"
echo "[RUN] output_path=${OUTPUT_PATH}"

python "$TRAIN_SCRIPT" \
  "${COMMON_ARGS[@]}" \
  "${SFT_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
