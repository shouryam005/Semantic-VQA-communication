#!/usr/bin/env bash
# Controlled RVNN vs CVNN comparison on the rebuilt benchmark.
# 50 epochs (both models were still improving at 20) and 5 seeds, since the
# effect being tested is smaller than the seed-to-seed spread.
set -u
E=${EPOCHS:-50}; R=${REPEATS:-5}
run () {
  name="$1"; shift
  echo "### $name"
  python -m sarvqa.train --epochs $E --repeats $R --out "results/$name.json" "$@" \
    > "results/$name.log" 2>&1
  echo "    $(grep -E '^  accuracy' "results/$name.log" | head -1)"
}
run question-only  --model question-only --no-channel
run rvnn-nochannel --model rvnn --no-channel
run cvnn-nochannel --model cvnn --no-channel
run rvnn-channel   --model rvnn
run cvnn-channel   --model cvnn
echo "ALL DONE"
