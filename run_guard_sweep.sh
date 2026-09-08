#!/usr/bin/env bash
# At what azimuth guard does validation accuracy stop falling?
# Below that point, val images still have near-duplicates in train and the
# score is inflated. Train size is held at 3476 QA pairs across every guard so
# the sweep measures leakage, not how much data each guard happens to leave.
set -u
for g in 0 1 2 3 5 10; do
  d="data_guard$g"; [ "$g" = "5" ] && d="data_v2"
  [ -f "results/g_$g.json" ] && { echo "skip guard=$g"; continue; }
  python -m sarvqa.train --model rvnn --no-channel --data "$d" \
    --epochs 50 --repeats 8 --limit-train 3476 --out "results/g_$g.json" \
    > "results/g_$g.log" 2>&1
  echo "guard=${g}deg  $(grep -E '^  accuracy' results/g_$g.log | head -1 | sed 's/  */ /g')"
done
echo "GUARD SWEEP DONE"
