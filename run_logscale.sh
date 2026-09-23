#!/usr/bin/env bash
# Does compressing dynamic range help? The whole ladder under log scaling so the
# comparison stays controlled, plus dB on the magnitude model to compare the two.
set -u
run () { name="$1"; shift
  [ -f "results/m_$name.json" ] && { echo "skip $name"; return; }
  python -u -m sarvqa.train --epochs 50 --repeats 10 --no-channel \
    --out "results/m_$name.json" "$@" > "results/m_$name.log" 2>&1
  echo "$(date +%H:%M)  $name  $(grep -E '^  accuracy' results/m_$name.log | head -1 | sed 's/  */ /g')"
}
run log-magnitude --model magnitude --normalize log
run log-rvnn      --model rvnn      --normalize log
run log-cvnn      --model cvnn      --normalize log
run db-magnitude  --model magnitude --normalize db
echo "LOGSCALE DONE"
