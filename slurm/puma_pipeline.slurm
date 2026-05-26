#!/bin/bash
# =============================================================================
# PUMA offline pipeline — generic SLURM template
# =============================================================================
# This is a *template*. Replace the placeholders in the #SBATCH block below
# (account / partition) and adjust the environment-activation section to match
# your cluster, then submit:
#
#   sbatch slurm/puma_pipeline.slurm <CONFIG> <BASE_DIR> <MODEL>
#
# Example:
#   sbatch slurm/puma_pipeline.slurm configs/DS-7B.conf runs/ds7b_math500 \
#          deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
#
# <BASE_DIR> must already contain answers.json (the pre-generated reasoning
# responses — see the repo README "Input format"). Outputs are written there.
#
# Resource hints (set via #SBATCH or sbatch flags):
#   <= 14B models : 1 GPU,  ~110 GB RAM
#   30B / 32B     : 2 GPUs, ~220 GB RAM   (run_puma.py picks up tensor parallel
#                                          from the visible GPU count)
# =============================================================================
#SBATCH --job-name=puma
#SBATCH --account=<your_slurm_account>      # <-- EDIT
#SBATCH --partition=<gpu_partition>         # <-- EDIT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=110g
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

CONFIG="${1:?Usage: sbatch puma_pipeline.slurm <CONFIG> <BASE_DIR> <MODEL>}"
BASE_DIR="${2:?Usage: sbatch puma_pipeline.slurm <CONFIG> <BASE_DIR> <MODEL>}"
MODEL="${3:?Usage: sbatch puma_pipeline.slurm <CONFIG> <BASE_DIR> <MODEL>}"

# Submit from the repository root so the relative paths below resolve.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

echo "=== PUMA pipeline @ $(date) ==="
echo "Job ID:   ${SLURM_JOB_ID:-<none>}"
echo "Node:     ${SLURM_NODELIST:-<none>}"
echo "GPUs:     ${CUDA_VISIBLE_DEVICES:-<none>}"
command -v nvidia-smi >/dev/null && nvidia-smi || true

# --- Activate your environment ----------------------------------------------
# Adjust this block for your cluster. Example with conda + the bundled env.yml:
source ~/.bashrc 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate puma

# Some clusters preload an MPI/NCCL plugin that conflicts with vLLM's bundled
# NCCL; unload it if you hit NCCL init errors (harmless if not present):
#   module unload nccl-ofi-plugin 2>/dev/null || true

# Offline compute nodes: point HF at a pre-populated cache and go offline.
#   export HF_HOME=/path/to/hf_cache
#   export HF_HUB_OFFLINE=1
echo "Python: $(which python)"

# --- Run ---------------------------------------------------------------------
bash run_from_config.sh "$CONFIG" "$BASE_DIR" "$MODEL"

echo "=== Done @ $(date) ==="
