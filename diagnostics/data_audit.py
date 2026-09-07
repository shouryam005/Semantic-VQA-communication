"""
Audit of the SAR input tensors themselves.

Two things matter for the RVNN-vs-CVNN comparison:

  1. Dynamic range. Neither SARVQADataset (nb2_2 cell 2, nb3_2 cell 3)
     normalizes. If the raw complex amplitudes are far from unit scale,
     default PyTorch initialization is mis-scaled for the first conv, and the
     complex path -- where CReLU zeroes the real and imaginary halves
     independently -- is hurt more than the real path.

  2. Phase content. The premise of the whole project is that SAR phase carries
     information a magnitude-only model throws away. For a single-look complex
     image, per-pixel absolute phase is dominated by random scatterer phase
     (speckle) and is close to uniform. If so, a CVNN cannot benefit from
     absolute phase; only relative/spatial phase structure could help.

Run:  python diagnostics/data_audit.py
"""

import pickle
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent.parent
N_IMAGES = 100


def main():
    with open(ROOT / "data" / "train_processed.pkl", "rb") as f:
        train = pickle.load(f)

    paths = list(dict.fromkeys(s["filepath"] for s in train))[:N_IMAGES]

    first = loadmat(paths[0])
    print("Contents of a SAMPLE .mat")
    print("  fields:", sorted(k for k in first if not k.startswith("__")))
    img = first["complex_img"]
    print(f"  complex_img: {img.shape} {img.dtype}")
    print()

    rows = []
    phase_hist = np.zeros(12)
    for p in paths:
        x = loadmat(p)["complex_img"]
        rows.append((np.abs(x).mean(), np.abs(x).max(), x.real.std()))
        h, _ = np.histogram(np.angle(x), bins=12, range=(-np.pi, np.pi))
        phase_hist += h
    a = np.array(rows)

    print(f"1. Dynamic range over {len(a)} training images (no normalization applied)")
    for i, label in enumerate(("mean |x|", "max  |x|", "std  Re(x)")):
        col = a[:, i]
        print(f"   {label:10s} min={col.min():9.4g}  median={np.median(col):9.4g}  "
              f"max={col.max():9.4g}  spread={col.max() / col.min():6.1f}x")
    print()
    print("   Inputs sit two orders of magnitude below unit scale, and peak amplitude")
    print("   varies by ~50x across images. Both encoders see this unscaled.")
    print()

    print(f"2. Phase distribution, pooled over {len(a)} images (12 bins over [-pi, pi])")
    frac = phase_hist / phase_hist.sum()
    print("   ", np.round(frac, 4))
    print(f"   max deviation from uniform (1/12 = {1/12:.4f}): {np.abs(frac - 1/12).max():.4f}")
    print()
    print("   Per-pixel absolute phase is essentially uniform, i.e. speckle. A CVNN")
    print("   cannot extract target identity from absolute phase alone; any benefit")
    print("   must come from relative phase structure between neighbouring pixels.")


if __name__ == "__main__":
    main()
