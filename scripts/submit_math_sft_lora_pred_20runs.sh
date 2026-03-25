#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"
mkdir -p logs

SLURM_SCRIPT="${SLURM_SCRIPT:-$PROJECT_ROOT/MATH_DCoT_LoRA.slurm}"
RUNS_PER_MODEL="${RUNS_PER_MODEL:-10}"
START_SEED="${START_SEED:-42}"

TRAIN_FILE="${TRAIN_FILE:-$PROJECT_ROOT/Baseline/data/MATH/results_dcot_teacher_gpt51.jsonl}"
TEST_FILE="${TEST_FILE:-$PROJECT_ROOT/Baseline/data/MATH/gsm8k_format_test_5000.jsonl}"

OUTPUT_BASE="${OUTPUT_BASE:-/home/jyang001/scratch/MATH_dcot_lora}"
PRED_BASE="${PRED_BASE:-$OUTPUT_BASE/predictions}"
PRED_NUM_COTS="${PRED_NUM_COTS:-3}"
PRED_MAX_NEW_TOKENS="${PRED_MAX_NEW_TOKENS:-2048}"

if [[ ! -f "$SLURM_SCRIPT" ]]; then
  echo "Missing slurm script: $SLURM_SCRIPT" >&2
  exit 2
fi
if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "Missing train file: $TRAIN_FILE" >&2
  exit 2
fi
if [[ ! -f "$TEST_FILE" ]]; then
  echo "Missing test file: $TEST_FILE" >&2
  exit 2
fi

MODEL_SPECS=(
  "llama|meta-llama/Llama-3.1-8B-Instruct|configs/lora_llama3p1_8b_instruct.json"
  "qwen|Qwen/Qwen3-1.7B|configs/lora_qwen3_1p7b.json"
)

submitted=0
for spec in "${MODEL_SPECS[@]}"; do
  IFS='|' read -r student_key model_name config_rel <<<"$spec"
  config_file="$PROJECT_ROOT/$config_rel"

  if [[ ! -f "$config_file" ]]; then
    echo "Missing config file: $config_file" >&2
    exit 2
  fi

  for run_id in $(seq 1 "$RUNS_PER_MODEL"); do
    run_tag="$(printf "run_%02d" "$run_id")"
    seed=$((START_SEED + run_id - 1))
    output_root="${OUTPUT_BASE%/}/${student_key}/${run_tag}"
    pred_root="${PRED_BASE%/}/${student_key}/${run_tag}"

    echo "[SUBMIT] model=${model_name} run=${run_tag} seed=${seed}"
    sbatch "$SLURM_SCRIPT" \
      --student-model "$student_key" \
      --model-name "$model_name" \
      --tokenizer-name "$model_name" \
      --data-file "$TRAIN_FILE" \
      --test-file "$TEST_FILE" \
      --output-root "$output_root" \
      --pred-root "$pred_root" \
      --pred-overwrite \
      --pred-num-cots "$PRED_NUM_COTS" \
      --pred-max-new-tokens "$PRED_MAX_NEW_TOKENS" \
      --seed "$seed" \
      --config_file "$config_file"

    submitted=$((submitted + 1))
  done
done

echo "[DONE] submitted_jobs=${submitted} runs_per_model=${RUNS_PER_MODEL} models=${#MODEL_SPECS[@]}"
