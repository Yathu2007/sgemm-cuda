#!/usr/bin/env bash
# Regenerates every number in the README: builds, runs every kernel at every
# size, logs GPU clocks alongside, and renders the results table and chart.
#
#   usage: bench/sweep.sh
#   env:   WARMUP=5 ITERS=50 SIZES="128 256 ..." KERNELS="0 cublas"
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-50}"
SIZES="${SIZES:-128 256 512 1024 2048 4096}"
# cuBLAS first so the baseline row exists even if a later kernel aborts.
KERNELS="${KERNELS:-cublas 0}"

RESULTS="$ROOT/results/results.csv"
CLOCKS="$ROOT/results/clocks.csv"
DEVICE="$ROOT/results/device_query.txt"
mkdir -p "$ROOT/results"

export PATH="/usr/local/cuda/bin:$PATH"

echo "== building =="
cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "$ROOT/build" -j > "$ROOT/build/build.log" 2>&1
echo "   ok (register usage: build/build.log)"

echo "== recording device properties =="
"$ROOT/build/device_query" | tee "$DEVICE" | sed -n '1,3p'

echo "== starting clock logger =="
"$ROOT/bench/clocks.sh" "$CLOCKS" 1 &
CLOCKS_PID=$!
# Make sure the logger dies with the sweep, including on Ctrl-C.
trap 'kill "$CLOCKS_PID" 2>/dev/null || true' EXIT

echo "== sweeping (warmup=$WARMUP iters=$ITERS) =="
"$ROOT/build/sgemm" --csv-header > "$RESULTS"
for n in $SIZES; do
  for k in $KERNELS; do
    printf '   kernel=%-7s N=%-5s ' "$k" "$n"
    row=$("$ROOT/build/sgemm" --kernel "$k" --size "$n" \
            --warmup "$WARMUP" --iters "$ITERS" --csv)
    echo "$row" >> "$RESULTS"
    echo "$row" | awk -F, '{printf "%10.3f ms  %9.1f GFLOPS  rel_err=%s\n", $4, $7, $8}'
  done
done

kill "$CLOCKS_PID" 2>/dev/null || true
trap - EXIT
echo "== wrote $RESULTS and $CLOCKS =="

echo "== rendering table + chart =="
python3 "$ROOT/bench/plot.py" --results "$RESULTS" --clocks "$CLOCKS" \
        --out-dir "$ROOT/results"
