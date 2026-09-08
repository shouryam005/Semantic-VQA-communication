# Semantic Communication for Visual Question Answering

Do complex-valued neural networks (CVNNs) offer any advantage over real-valued
ones (RVNNs) for VQA transmitted over a noisy semantic-communication channel,
and does that change when the source data is natively complex?

The project has two experimental eras. **They use different datasets and their
numbers are not comparable.** Conflating them is the single easiest way to
misread this repository.

| era | data | notebooks |
|---|---|---|
| 1. DisasterVQA | RGB imagery, BLIP-2 features | `nb1_preprocess`, `nb2_blip_features`, `nb3_rvnn_pipeline`, `nb4_cvnn_pipeline`, `nb5_experimenting` |
| 2. SAMPLE SAR | natively complex SAR, 2690 images, 10 vehicle classes | `exploring_datasets`, `nb2_2_rvnn_baseline`, `nb3_2_cvnn_baseline` |

Era 2 is the current work. The `benchmark/`, `sarvqa/` and `diagnostics/`
packages supersede the era-2 notebooks; the notebooks are kept as the record of
what was originally run.

## Status: earlier SAR results were not measuring what they appeared to

The notebooks reported RVNN ≈89% against CVNN ≈69% on SAMPLE, which suggested a
large complex-valued deficit. An audit found that comparison was uncontrolled.
Each of the following is independently sufficient to invalidate it:

- The CVNN ran for **3 epochs**, the RVNN for **20**.
- The RVNN ran **with** its AWGN channel active; the CVNN's was commented out.
- The benchmark had a **69.1% question-only ceiling** — a model answering from
  the question alone, never seeing the image, scored what the CVNN scored. The
  CVNN's training accuracy flatlined at 69.13% against that ceiling, meaning it
  used no image information at all.
- **90%** of validation images had a training image of the same target and
  elevation within 1° of azimuth — near-duplicates, so validation measured
  memorization.
- Complex layers were built as `nn.Linear(...).to(torch.cfloat)`, which
  initializes a real layer and casts it, leaving every weight with a zero
  imaginary part under a variance wrong for complex fan-in.

Rebuilt and rerun with those controlled, the gap is **2.31 pp**
(RVNN 84.08 ± 2.72, CVNN 81.78 ± 2.40, n=20 seeds, p=0.0073).

Reproduce the audit:

```bash
python diagnostics/benchmark_audit.py   # question-only ceiling
python diagnostics/split_audit.py       # near-duplicate leakage
python diagnostics/data_audit.py        # dynamic range and phase content
```

## A finding that reframes the question

`data_audit.py` measures SAMPLE's per-pixel phase as uniform to within 0.002 —
it is single-look speckle. **Absolute phase carries no target identity.** CVNNs
win where phase is *coherent* (PolSAR, InSAR), not merely where it is present,
so SAMPLE is close to a worst case for demonstrating a complex-valued
advantage.

A second measurement compounds this. Global average pooling of complex features
cancels them: with a phase-preserving activation only **13.5%** of the signal
survives the collapse, because averaging complex numbers with unrelated phases
destroys them. CReLU hides this by forcing `Re ≥ 0` and `Im ≥ 0`, confining
every activation to the first quadrant — pooling then survives at 91%, but
phase has already been crushed into a 90° wedge. Under either activation almost
no phase reached the classifier, so the original CVNN could not have used phase
even had it been there.

## Rebuilt benchmark

```bash
python benchmark/build_benchmark.py                     # default
python benchmark/build_benchmark.py --split sample-protocol
```

Three fixes over `exploring_datasets.ipynb` cells 7–21:

- **Per-question balance.** The old generator balanced answers globally and per
  image but not per question; with only two mobility classes and 2248 of 2690
  images tracked, "Is this a tracked vehicle?" was 83.7% yes. Every question is
  now exactly 50/50, so the question-only ceiling is **50.00%**.
- **Balance inside each split**, by splitting images first and generating QA
  per split rather than the reverse.
- **Azimuth-blocked splits.** Contiguous azimuth arcs per (target, elevation)
  with a guard band. Near-duplicate leakage drops from 90.0% to 0.0%.

Everything is seeded. The old generator called `random.choice` unseeded, so
`sample_qa.pkl` could not be regenerated.

## Models

```bash
python -m sarvqa.models                                  # parameter budgets
python -m sarvqa.train --model rvnn --no-channel --repeats 10
python -m sarvqa.train --model cvnn --pooling coherence --activation cardioid
python -m sarvqa.train --model question-only             # must sit at 50%
python diagnostics/summarize.py                          # matrix with significance tests
```

Both models share the input tensor, question encoder, classifier, transmitted
width (32 real values), SNR definition and channel on/off switch. Only the
image path differs. Image-encoder parameters match within 2.3% (RVNN 23,440,
CVNN 22,900); a complex layer stores two real weight tensors, so CVNN channel
widths are chosen to match the parameter budget rather than the channel count.

`--pooling` selects how the complex spatial map is collapsed: `avg` (the
original, which cancels), `coherence` (`[mean|z|, |mean z|]` — energy plus the
standard SAR coherence estimator), or `modulus` (phase-free control).
`--activation` offers `crelu`, `modrelu`, `zrelu`, `cardioid`. `--film`
conditions the transmitted code on the question; without it the pipeline is
image compression followed by VQA rather than task-oriented semantic
communication.

## Known open questions

- Whether the residual RVNN advantage survives phase-preserving pooling.
- The right azimuth guard: pixel correlation collapses by 1° and plateaus
  thereafter, suggesting 5° is conservative. `run_guard_sweep.sh` measures it
  at fixed training-set size.
- Whether a coherent-phase dataset (PolSAR) shows the CVNN advantage that
  SAMPLE cannot.

## Layout

```
benchmark/      rebuilt benchmark generator
sarvqa/         models, complex layers, data pipeline, training
diagnostics/    audits and result summarization
data_v2/        generated benchmark (regenerate; not tracked)
results/        experiment outputs (not tracked)
*.ipynb         original notebooks, kept as the historical record
```
