#!/usr/bin/env bash
# Full experiment matrix. All runs: 50 epochs, 10 seeds, global normalization.
set -u
E=50; R=10
run () { name="$1"; shift
  [ -f "results/m_$name.json" ] && { echo "skip $name"; return; }
  python -m sarvqa.train --epochs $E --repeats $R --out "results/m_$name.json" "$@" \
    > "results/m_$name.log" 2>&1
  echo "$name  $(grep -E '^  accuracy' results/m_$name.log | head -1 | sed 's/  */ /g')"
}

# A. baselines under global normalization
run base-rvnn  --model rvnn --no-channel
run base-cvnn  --model cvnn --no-channel

# B. pooling ablation (isolates the image representation)
run pool-coherence --model cvnn --no-channel --pooling coherence
run pool-modulus   --model cvnn --no-channel --pooling modulus

# C. activation ablation, on the pooling that preserves phase
run act-cardioid --model cvnn --no-channel --pooling coherence --activation cardioid
run act-modrelu  --model cvnn --no-channel --pooling coherence --activation modrelu
run act-zrelu    --model cvnn --no-channel --pooling coherence --activation zrelu

# D. same activations under the old cancelling pooling, to separate the two effects
run avg-cardioid --model cvnn --no-channel --pooling avg --activation cardioid

# E. semantic communication, with and without question conditioning
run chan-rvnn      --model rvnn
run chan-cvnn      --model cvnn
run chan-rvnn-film --model rvnn --film
run chan-cvnn-film --model cvnn --film
echo "MATRIX DONE"
