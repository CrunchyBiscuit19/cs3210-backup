#!/usr/bin/env bash
#SBATCH --job-name=a2-bench
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --gpus=a100-80
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#
# Times ./path_tracing against benchmarks/bench_0..bench_4 on the graded
# (non-debug) scenes. Every run uses --correctness-check, so a fast but wrong
# result shows up as FAIL. Raw data goes to results/bench-<gpu>-<job>.csv
# (keep these for the evidence pack); a mean-time and speedup table is
# printed at the end of the log.
#
# Usage (from the repo root):
#   sbatch scripts/bench.sh
#   sbatch --gpus=h100-96 --constraint=xgpi scripts/bench.sh
#   RUNS=5 IMPLS="path_tracing bench_4" sbatch scripts/bench.sh scenes/dragon.rtscene
#   IMPLS="v1_culling v2_ray_cost v3_occupancy v4_coherence bench_4" sbatch scripts/bench.sh
#
# Env: RUNS (default 3), IMPLS (default: path_tracing and all five benchmarks;
#      v*_name runs build/<name> from versions/),
#      SKIP_BUILD=1, RUN_TIMEOUT=<s> per run (default 900).

source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/common.sh"

if (( $# > 0 )); then
    scenes=("$@")
else
    scenes=("${MAIN_SCENES[@]}")
fi
runs=${RUNS:-3}
read -ra impls <<<"${IMPLS:-$ALL_IMPLS}"

print_env
build

csv="$RESULTS_DIR/bench-$(gpu_tag)-$JOB_ID.csv"
img_dir="$RESULTS_DIR/bench-$JOB_ID"
mkdir -p "$img_dir"
echo "job,node,gpu,impl,scene,run,time_s,error,max_error,status" >"$csv"

# Runs are interleaved across implementations so drift on the node affects all equally.
for scene in "${scenes[@]}"; do
    name=$(basename "$scene" .rtscene)
    for ((run = 1; run <= runs; run++)); do
        for impl in "${impls[@]}"; do
            echo
            echo "== $impl $scene (run $run/$runs)"
            run_one "$(impl_path "$impl")" "$scene" \
                --output "$img_dir/$impl-$name.ppm" --correctness-check "$(ref_for "$scene")"
            echo "$JOB_ID,$(hostname),$(gpu_name),$impl,$name,$run,$RUN_TIME,$RUN_ERROR,$RUN_MAX_ERROR,$RUN_STATUS" >>"$csv"
        done
    done
done

echo
echo "== Mean time in seconds over successful runs on $(gpu_name)  (raw data: ${csv#$ROOT/})"
awk -F, -v impls="${impls[*]}" '
    NR == 1 { next }
    {
        if (!($5 in seen)) { seen[$5]; order[++ns] = $5 }
        if ($7 != "" && ($10 == "PASS" || $10 == "OK")) { sum[$5, $4] += $7; cnt[$5, $4]++ }
        if ($10 != "PASS" && $10 != "OK") bad[$5, $4]++
    }
    END {
        ni = split(impls, im, " ")
        printf "%-14s", "scene"
        for (i = 1; i <= ni; i++) printf " %14s", im[i]
        printf "\n"
        for (s = 1; s <= ns; s++) {
            sc = order[s]
            printf "%-14s", sc
            for (i = 1; i <= ni; i++) {
                k = sc SUBSEP im[i]
                cell = cnt[k] ? sprintf("%.4f", sum[k] / cnt[k]) : "-"
                if (bad[k]) cell = cell "*"
                printf " %14s", cell
            }
            printf "\n"
        }
        print "(* = at least one run failed the correctness check, crashed, or timed out)"

        if (!index(" " impls " ", " path_tracing ")) exit
        printf "\n%-14s", "speedup"
        for (i = 1; i <= ni; i++) if (im[i] != "path_tracing") printf " %14s", "vs " im[i]
        printf "\n"
        for (s = 1; s <= ns; s++) {
            sc = order[s]
            ours = sc SUBSEP "path_tracing"
            printf "%-14s", sc
            for (i = 1; i <= ni; i++) {
                if (im[i] == "path_tracing") continue
                k = sc SUBSEP im[i]
                if (cnt[ours] && cnt[k]) printf " %13.2fx", (sum[k] / cnt[k]) / (sum[ours] / cnt[ours])
                else printf " %14s", "-"
            }
            printf "\n"
        }
    }' "$csv"
