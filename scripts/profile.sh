#!/usr/bin/env bash
#SBATCH --job-name=a2-profile
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --gpus=a100-80
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
#
# Profiles one implementation (default ./path_tracing) on one scene with Nsight Systems and/or Nsight Compute.
# Reports go to results/profile/; copy them to your machine and open them in the
# Nsight GUIs. nsys also prints its summary tables into the log.
#
# Usage (from the repo root):
#   sbatch scripts/profile.sh <scene> [extra path_tracing args...]
#   sbatch scripts/profile.sh scenes/cornell_box.rtscene
#   TOOL=ncu sbatch scripts/profile.sh scenes/dragon.rtscene --samples_per_pixel 64
#   TOOL=both sbatch --gpus=h100-96 --constraint=xgpi scripts/profile.sh scenes/ufo.rtscene
#   IMPL=v0_naive TOOL=ncu sbatch scripts/profile.sh scenes/cornell_box.rtscene --samples_per_pixel 16
#
# Env: IMPL     implementation name, as in bench.sh (default path_tracing)
#      TOOL=nsys|ncu|both (default nsys)
#      NCU_SET  ncu section set (default full)
#      NCU_COUNT number of kernel launches ncu profiles (default 3)
#      SKIP_BUILD=1
# ncu replays each kernel many times. With the full set, a lower
# --samples_per_pixel keeps the job short. Kernel behaviour per sample is unchanged.

source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/common.sh"

if (( $# < 1 )); then
    echo "usage: sbatch scripts/profile.sh <scene> [extra path_tracing args...]" >&2
    exit 1
fi
scene=$1
shift
tool=${TOOL:-nsys}
impl=${IMPL:-path_tracing}
exe=$(impl_path "$impl")

print_env
build
require_exe "$exe"

prof_dir="$RESULTS_DIR/profile"
mkdir -p "$prof_dir"
base="$prof_dir/$impl-$(basename "$scene" .rtscene)-$(gpu_tag)-$JOB_ID"
status=0

if [[ "$tool" == nsys || "$tool" == both ]]; then
    echo
    echo "== nsys -> ${base#$ROOT/}.nsys-rep"
    nsys profile --stats=true --force-overwrite=true --trace=cuda,nvtx,osrt -o "$base" \
        "$exe" "$scene" --output "$base-nsys.ppm" "$@" || status=1
fi

if [[ "$tool" == ncu || "$tool" == both ]]; then
    echo
    echo "== ncu (--set ${NCU_SET:-full}) -> ${base#$ROOT/}.ncu-rep"
    ncu --set "${NCU_SET:-full}" --import-source yes --launch-count "${NCU_COUNT:-3}" \
        --force-overwrite -o "$base" \
        "$exe" "$scene" --output "$base-ncu.ppm" "$@" || {
        status=1
        echo "ncu failed. ERR_NVGPUCTRPERM means GPU performance counters are disabled for users on this node." >&2
    }
fi

exit $status
