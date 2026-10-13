#!/usr/bin/env bash
#SBATCH --job-name=a2-correct-lite
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --gpus=a100-80
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#
# Quick correctness check: correctness.sh on scenes/cornell_box.rtscene only.
#
# Usage (from the repo root):
#   sbatch scripts/correctness_lite.sh
#   IMPLS="v3Occupancy v4Coherence" sbatch scripts/correctness_lite.sh
#   sbatch --gpus=h100-96 --constraint=xgpi scripts/correctness_lite.sh
#
# Takes the same environment variables as correctness.sh
# (IMPLS, SKIP_BUILD, RESULTS_DIR, RUN_TIMEOUT).

cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
exec bash scripts/correctness.sh scenes/cornell_box.rtscene
