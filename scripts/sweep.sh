#!/usr/bin/env bash
#SBATCH --job-name=a2-sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --gpus=a100-80
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#
# Measures how runtime scales with one parameter, for the evidence pack's
# "how do input factors / CUDA parameters affect runtime" questions. Each value
# replaces {} in the template. Template tokens of the form NAME=... are exported
# as environment variables (the PT_* tuning knobs read by path_tracing.cu);
# all other tokens are passed to the executable as flags.
#
# If the template has no -- flags, the render settings are unchanged, so every run
# also gets --correctness-check. Otherwise the reference images no longer apply and
# no check is done.
# Raw data goes to results/sweep-<scene>-<gpu>-<job>.csv.
#
# Usage (from the repo root):
#   sbatch scripts/sweep.sh <scene> "<template with {}>" <value>...
#   sbatch scripts/sweep.sh scenes/cornell_box.rtscene "--samples_per_pixel {}" 100 1000 10000
#   sbatch scripts/sweep.sh scenes/dragon.rtscene "--width {} --height {}" 128 256 512 1024
#   sbatch scripts/sweep.sh scenes/ufo.rtscene "--max_bounces {}" 0 1 2 3 4 5
#   sbatch scripts/sweep.sh scenes/dragon.rtscene "PT_BLOCK_SIZE={}" 32 64 128 256 512
#   sbatch scripts/sweep.sh scenes/ufo.rtscene "PT_SAMPLES_PER_THREAD={}" 16 64 256 1024
#   IMPLS="path_tracing bench_4" sbatch scripts/sweep.sh ...   # compare scaling with a benchmark
#
# Env: RUNS (default 3), IMPLS (default "path_tracing"), SKIP_BUILD=1, RUN_TIMEOUT=<s>.
# Note: the scene's screen width/height ratio must keep matching width/height,
# so scale --width and --height together.

source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/common.sh"

if (( $# < 3 )); then
    echo "usage: sbatch scripts/sweep.sh <scene> \"<template with {}>\" <value>..." >&2
    exit 1
fi
scene=$1
template=$2
shift 2
values=("$@")
runs=${RUNS:-3}
read -ra impls <<<"${IMPLS:-path_tracing}"
name=$(basename "$scene" .rtscene)

check_args=()
if [[ " $template" != *" --"* ]]; then
    check_args=(--correctness-check "$(ref_for "$scene")")
fi

print_env
build "${impls[@]}"

csv="$RESULTS_DIR/sweep-$name-$(gpu_tag)-$JOB_ID.csv"
img_dir="$RESULTS_DIR/sweep-$JOB_ID"
mkdir -p "$img_dir"
echo "job,node,gpu,impl,scene,template,value,run,time_s,error,status" >"$csv"

for value in "${values[@]}"; do
    read -ra tokens <<<"${template//\{\}/$value}"
    env_args=()
    flag_args=()
    for token in "${tokens[@]}"; do
        if [[ "$token" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
            env_args+=("$token")
        else
            flag_args+=("$token")
        fi
    done

    for ((run = 1; run <= runs; run++)); do
        for impl in "${impls[@]}"; do
            echo
            echo "== $impl $scene ${tokens[*]} (run $run/$runs)"
            for assignment in "${env_args[@]}"; do export "$assignment"; done
            run_one "$(impl_path "$impl")" "$scene" "${flag_args[@]}" "${check_args[@]}" \
                --output "$img_dir/$impl-$name-$value.ppm"
            for assignment in "${env_args[@]}"; do unset "${assignment%%=*}"; done
            echo "$JOB_ID,$(hostname),$(gpu_name),$impl,$name,\"$template\",$value,$run,$RUN_TIME,$RUN_ERROR,$RUN_STATUS" >>"$csv"
        done
    done
done

echo
echo "== Mean time in seconds for $scene, $template  (raw data: ${csv#$ROOT/})"
awk -F, -v impls="${impls[*]}" '
    NR == 1 { next }
    {
        if (!($7 in seen)) { seen[$7]; order[++nv] = $7 }
        if ($9 != "" && ($11 == "OK" || $11 == "PASS")) { sum[$7, $4] += $9; cnt[$7, $4]++ }
        if ($11 != "OK" && $11 != "PASS") bad[$7, $4]++
    }
    END {
        ni = split(impls, im, " ")
        printf "%-10s", "value"
        for (i = 1; i <= ni; i++) printf " %14s", im[i]
        printf "\n"
        for (v = 1; v <= nv; v++) {
            printf "%-10s", order[v]
            for (i = 1; i <= ni; i++) {
                k = order[v] SUBSEP im[i]
                cell = cnt[k] ? sprintf("%.4f", sum[k] / cnt[k]) : "-"
                if (bad[k]) cell = cell "*"
                printf " %14s", cell
            }
            printf "\n"
        }
        print "(* = at least one run failed the correctness check, crashed, or timed out)"
    }' "$csv"
