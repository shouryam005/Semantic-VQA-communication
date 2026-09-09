#!/usr/bin/env bash
# Tier 1: three claims that are currently hanging.
#   magnitude   -- does SAR phase carry anything at all on SAMPLE?
#   *-seeds2    -- second block of 10 seeds, taking the global-norm baselines to n=20
#   avg-cardioid-- separates the activation effect from the pooling effect
# python -u so progress is visible live rather than flushed at exit.
set -u
run () { name="$1"; shift
  [ -f "results/m_$name.json" ] && { echo "skip $name"; return; }
  python -u -m sarvqa.train --epochs 50 --out "results/m_$name.json" "$@" \
    > "results/m_$name.log" 2>&1
  echo "$(date +%H:%M)  $name  $(grep -E '^  accuracy' results/m_$name.log | head -1 | sed 's/  */ /g')"
}
run magnitude        --model magnitude --no-channel --repeats 10
run base-rvnn-seeds2 --model rvnn      --no-channel --repeats 10 --seed 10
run base-cvnn-seeds2 --model cvnn      --no-channel --repeats 10 --seed 10
run avg-cardioid     --model cvnn      --no-channel --repeats 10 --pooling avg --activation cardioid
echo "TIER 1 DONE"
