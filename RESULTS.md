# SAMPLE SAR-VQA results

All runs: rebuilt benchmark (`data_v2`), 50 epochs, Adam lr=1e-3, global
normalization, image path only (no semantic-communication channel).
Chance is 50.00% by construction — every question string is balanced 50/50.

Regenerate with `run_matrix.sh` / `run_tier1.sh`; summarize with
`python diagnostics/summarize.py`.

## Headline table

| model | phase exposure | val accuracy | n |
|---|---|---|---|
| `question-only` (control) | — | 50.00 ± 0.00 | 10 |
| **`magnitude`** — `\|x\|` only | none | **88.87 ± 1.01** | 10 |
| `rvnn` — `[Re, Im]` as 2 channels | present, real arithmetic | 83.97 ± 2.10 | 20 |
| `cvnn` — complex, CReLU, avg pool | structural | 82.13 ± 2.31 | 20 |
| `cvnn` — modulus pool | structural, phase-free output | 80.70 ± 2.07 | 10 |
| `cvnn` — coherence pool | structural | 80.38 ± 3.03 | 10 |
| `cvnn` — avg pool, Cardioid | structural, phase-preserving act. | 80.25 ± 1.83 | 9 |

## Significance (Welch, two-sided)

| comparison | difference | p | Cohen d |
|---|---|---|---|
| magnitude vs rvnn | +4.90 pp | < 1e-5 | 2.98 |
| magnitude vs cvnn | +6.74 pp | < 1e-5 | 3.77 |
| rvnn vs cvnn | +1.84 pp | 0.012 | 0.83 |
| cardioid vs CReLU | −1.88 pp | 0.030 | −0.83 |

Seed variance: magnitude sd 1.01 against rvnn 2.10 (F-test p=0.030) and
cvnn 2.31 (p=0.015). The phase-blind model is also the most stable.

## What these say

Discarding phase before the first convolution beats every model that
receives it, by a margin with effect sizes near d=3. On single-look
single-channel SAR the per-pixel phase is uniform speckle
(`diagnostics/data_audit.py`: uniform to 0.002), so `|x|` is a sufficient
statistic and feeding `Re`/`Im` adds a nuisance dimension the network must
learn to ignore — expensive with 1706 training images.

No phase-preserving change helped. Cardioid sits below CReLU, and both
alternative pooling modes sit below plain average pooling. Within the CVNN
the variants cluster at 80–82%; the choice of activation and pooling matters
far less than whether phase enters at all.

## Superseded numbers

The notebooks reported RVNN ~89% vs CVNN ~69%. Both are artifacts:

- 89% came from a split with 90% near-duplicate leakage (see
  `diagnostics/split_audit.py`). Guarded, the same model gets ~84%.
- 69% was the question-only prior of the old benchmark
  (`diagnostics/benchmark_audit.py`), reached by a CVNN trained for 3 epochs
  against the RVNN's 20, with its AWGN channel commented out while the
  RVNN's was active, and complex layers initialized by casting real ones.

An earlier n=10 run under per-image normalization reported no significant
RVNN–CVNN gap (0.67 pp, p=0.478). That was underpowered; at n=20 the gap is
1.84 pp at p=0.012.

## Open

- Control not yet run: RVNN fed `[|x|, |x|]` — two channels, phase-free.
  Separates "phase hurts" from "one channel and all-positive inputs suit
  ReLU better". Needed before the phase claim goes in a writeup.
- Azimuth guard band is set to 5 degrees by assertion, not measurement.
  `run_guard_sweep.sh` is parked (`RUN_GUARD_SWEEP=1` to run).
- Semantic-communication arm: only a 10 dB point exists, at 5 seeds under
  per-image normalization. No SNR sweep, and the FiLM runs never executed.
