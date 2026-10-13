#!/usr/bin/env bash
#SBATCH --job-name=a2-correct
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --gpus=a100-80
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#
# Builds everything and runs the chosen implementations on every scene with
# --correctness-check, then prints a PASS/FAIL summary. Exits non-zero if any
# run fails.
#
# Usage (from the repo root):
#   sbatch scripts/correctness.sh                               # ./path_tracing, debug + main scenes, A100
#   sbatch --gpus=h100-96 --constraint=xgpi scripts/correctness.sh
#   sbatch scripts/correctness.sh scenes/debug/*.rtscene        # chosen scenes only
#   IMPLS="v0Naive v1Culling" sbatch scripts/correctness.sh   # versions from versions/
#
# Env: IMPLS (default "path_tracing"; also v*_name for build/<name>, bench_N),
#      SKIP_BUILD=1 reuses existing executables, RUN_TIMEOUT=<s> per run (default 900).
# v0Naive is slow on the large scenes; raise --time if you check it on all of them.
# Rendered images go to results/correctness-<job>/, the log to logs/.

source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/common.sh"

if (( $# > 0 )); then
    scenes=("$@")
else
    scenes=("${DEBUG_SCENES[@]}" "${MAIN_SCENES[@]}")
fi
read -ra impls <<<"${IMPLS:-path_tracing}"

print_env
build "${impls[@]}"
for impl in "${impls[@]}"; do
    require_exe "$(impl_path "$impl")"
done

out_dir="$RESULTS_DIR/correctness-$JOB_ID"
mkdir -p "$out_dir"

row_fmt='%-16s %-14s %-10s %12s %14s %10s\n'
summary=()
failed=0
for impl in "${impls[@]}"; do
    for scene in "${scenes[@]}"; do
        name=$(basename "$scene" .rtscene)
        echo
        echo "== $impl $scene"
        run_one "$(impl_path "$impl")" "$scene" \
            --output "$out_dir/$impl-$name.ppm" --correctness-check "$(ref_for "$scene")"
        [[ "$RUN_STATUS" == PASS ]] || failed=$((failed + 1))
        summary+=("$(printf "$row_fmt" "$impl" "$name" "$RUN_STATUS" "${RUN_TIME:--}" "${RUN_ERROR:--}" "${RUN_MAX_ERROR:--}")")
    done
done

echo
echo "== Summary on $(gpu_name)"
printf "$row_fmt" impl scene status time_s error max_error
printf '%s\n' "${summary[@]}"
echo "== $failed of $(( ${#impls[@]} * ${#scenes[@]} )) run(s) did not pass"
(( failed == 0 ))
