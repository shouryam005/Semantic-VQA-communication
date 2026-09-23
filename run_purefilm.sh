#!/usr/bin/env bash
# Standard FiLM (Perez et al.): conditioning REPLACES late fusion rather than
# supplementing it. The question then reaches the model only through gamma/beta
# on the transmitted code, so any accuracy retained is attributable purely to
# question-aware compression.
set -u
run () { name="$1"; shift
  [ -f "results/m_$name.json" ] && { echo "skip $name"; return; }
  python -u -m sarvqa.train --epochs 50 --repeats 10 --snr 10 --film --no-late-fusion \
    --out "results/m_$name.json" "$@" > "results/m_$name.log" 2>&1
  echo "$(date +%H:%M)  $name  $(grep -E '^  accuracy' results/m_$name.log | head -1 | sed 's/  */ /g')"
}
run purefilm-magnitude --model magnitude
run purefilm-rvnn      --model rvnn
run purefilm-cvnn      --model cvnn
echo "PURE FILM DONE"
