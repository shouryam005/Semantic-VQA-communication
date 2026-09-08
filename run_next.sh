#!/usr/bin/env bash
set -u
echo "### pool-modulus (phase-free final features)"
python -m sarvqa.train --model cvnn --no-channel --pooling modulus \
  --epochs 50 --repeats 10 --out results/m_pool-modulus.json \
  > results/m_pool-modulus.log 2>&1
echo "pool-modulus  $(grep -E '^  accuracy' results/m_pool-modulus.log | head -1 | sed 's/  */ /g')"
echo "### guard band sweep"
bash run_guard_sweep.sh
