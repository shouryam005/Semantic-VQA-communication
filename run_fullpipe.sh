#!/usr/bin/env bash
# FULL PIPELINE (channel ON @10dB) -- the numbers that matter for a comms report.
# Ladder first so the headline table completes, then question conditioning.
set -u
# wait for the in-flight chan-cvnn to write its result (no pgrep in Git Bash)
while [ ! -f results/m_chan-cvnn.json ]; do sleep 30; done
run () { name="$1"; shift
  [ -f "results/m_$name.json" ] && { echo "skip $name"; return; }
  python -u -m sarvqa.train --epochs 50 --repeats 10 --snr 10 \
    --out "results/m_$name.json" "$@" > "results/m_$name.log" 2>&1
  echo "$(date +%H:%M)  $name  $(grep -E '^  accuracy' results/m_$name.log | head -1 | sed 's/  */ /g')"
}
run chan-magnitude      --model magnitude
run chan-rvnn           --model rvnn
run chan-magnitude-film --model magnitude --film
run chan-rvnn-film      --model rvnn      --film
run chan-cvnn-film      --model cvnn      --film
echo "FULL PIPELINE DONE"
