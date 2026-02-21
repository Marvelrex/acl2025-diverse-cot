#!/bin/bash
#SBATCH --job-name=dcot_lora_gsm8k
#SBATCH --account=def-sdrew
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --output=logs/dcot_lora_gsm8k_%j.out
#SBATCH --error=logs/dcot_lora_gsm8k_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
export DATASET_NAME="GSM8K"
export SFT_TYPE="lora"

bash "$PROJECT_ROOT/scripts/run_dcot_sft.sh" "$@"
