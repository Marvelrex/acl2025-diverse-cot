#!/bin/bash
#SBATCH --job-name=dcot_lora_aqua
#SBATCH --account=def-sdrew
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --output=logs/dcot_lora_aqua_%j.out
#SBATCH --error=logs/dcot_lora_aqua_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
export DATASET_NAME="AQUA"
export SFT_TYPE="lora"

DEFAULT_DATA_FILE="$PROJECT_ROOT/Baseline/data/AQUA/results_dcot_teacher.jsonl"
bash "$PROJECT_ROOT/scripts/run_dcot_sft.sh" --data-file "$DEFAULT_DATA_FILE" "$@"
