#!/usr/bin/env bash
# Logs SM clock, memory clock, power, temperature and utilisation once per
# second. Run alongside every sweep: a 140 W laptop part throttles under a
# sustained benchmark loop, and unstable clocks make absolute GFLOPS figures
# indefensible. sweep.sh starts and stops this automatically.
#
#   usage: bench/clocks.sh [output.csv] [interval_seconds]
set -euo pipefail

OUT="${1:-$(dirname "$0")/../results/clocks.csv}"
INTERVAL="${2:-1}"

mkdir -p "$(dirname "$OUT")"
exec nvidia-smi \
  --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu \
  --format=csv -l "$INTERVAL" > "$OUT"
