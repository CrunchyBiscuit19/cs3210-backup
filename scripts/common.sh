#!/usr/bin/env bash
# Shared helpers sourced by the sbatch scripts in this directory.
#
# Slurm copies a batch script into a spool directory before running it, so the
# repo is located through the submit directory: always run
# `sbatch scripts/<name>.sh` from the repository root.

ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
if [[ ! -f Makefile || ! -f main.cu ]]; then
    echo "error: submit from the repository root (got $ROOT)" >&2
    exit 1
fi

# nvcc / nsys / ncu are not always on PATH inside batch jobs.
if ! command -v nvcc >/dev/null 2>&1 && [[ -d /usr/local/cuda/bin ]]; then
    export PATH="/usr/local/cuda/bin:$PATH"
fi

JOB_ID="${SLURM_JOB_ID:-local$$}"
RESULTS_DIR="$ROOT/results"
mkdir -p "$RESULTS_DIR" "$ROOT/logs"

DEBUG_SCENES=(scenes/debug/*.rtscene)
MAIN_SCENES=(scenes/*.rtscene)
ALL_IMPLS="path_tracing bench_0 bench_1 bench_2 bench_3 bench_4"

gpu_name() {
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1
}

# Filename-safe GPU tag, e.g. "A100-SXM4-80GB".
gpu_tag() {
    local name
    name=$(gpu_name)
    name=${name#NVIDIA }
    name=${name// /_}
    echo "${name:-unknown-gpu}"
}

print_env() {
    echo "== job $JOB_ID on $(hostname) at $(date '+%F %T')"
    echo "== GPU: $(gpu_name)"
    echo "== nvcc: $(nvcc --version 2>/dev/null | tail -n1)"
}

# Builds ./path_tracing and every versions/v*.cu into build/ unless SKIP_BUILD=1.
# make -k keeps going past a version that fails to compile; runs of a missing
# executable are reported as CRASH(127) instead of aborting the whole job.
build() {
    if [[ "${SKIP_BUILD:-0}" == 1 ]]; then
        echo "== SKIP_BUILD=1, using existing executables"
    else
        echo "== make path_tracing versions"
        make -k path_tracing versions || echo "warning: some targets failed to build" >&2
    fi
    chmod +x benchmarks/bench_* 2>/dev/null
}

# Exits if an executable needed by the job is missing.
require_exe() {
    if [[ ! -x "$1" ]]; then
        echo "error: $1 was not built" >&2
        exit 1
    fi
}

# Maps an implementation name to its executable:
#   path_tracing -> ./path_tracing, bench_N -> ./benchmarks/bench_N, v2_ray_cost -> ./build/v2_ray_cost
impl_path() {
    case "$1" in
        path_tracing) echo ./path_tracing ;;
        bench_*) echo "./benchmarks/$1" ;;
        *) echo "./build/$1" ;;
    esac
}

# Reference image for a scene file: scenes/debug/debug_0.rtscene -> images/debug_0.ppm
ref_for() {
    echo "images/$(basename "$1" .rtscene).ppm"
}

# run_one <executable> <scene> [args...]
# Runs one render, echoes its output, and sets:
#   RUN_TIME       seconds reported by main.cu ("Time taken: ...")
#   RUN_ERROR      MSE vs. the reference (only with --correctness-check)
#   RUN_MAX_ERROR  threshold read from the reference image
#   RUN_STATUS     PASS | FAIL | OK | TIMEOUT | CRASH(<exit code>)
run_one() {
    local exe=$1 scene=$2
    shift 2
    local out rc
    out=$(timeout "${RUN_TIMEOUT:-900}" "$exe" "$scene" "$@" 2>&1)
    rc=$?
    echo "$out"

    RUN_TIME=$(sed -n 's/^Time taken: \([^ ]*\) seconds.*/\1/p' <<<"$out" | tail -n1)
    RUN_ERROR=$(sed -n 's/^Error: \([^ ]*\)$/\1/p' <<<"$out" | tail -n1)
    RUN_MAX_ERROR=$(sed -n 's/^Max error: \([0-9.eE+-]*\)$/\1/p' <<<"$out" | tail -n1)

    if (( rc == 124 )); then
        RUN_STATUS=TIMEOUT
    elif grep -q '^Correctness check: PASS' <<<"$out"; then
        RUN_STATUS=PASS
    elif grep -q '^Correctness check: FAIL' <<<"$out"; then
        RUN_STATUS=FAIL
    elif (( rc != 0 )); then
        RUN_STATUS="CRASH($rc)"
    else
        RUN_STATUS=OK
    fi
}
