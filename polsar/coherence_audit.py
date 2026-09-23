"""
Does this PolSAR scene carry phase a network could use?

Run this before building anything else on a new scene. On SAMPLE the equivalent
check (diagnostics/data_audit.py) found per-pixel phase uniform to within 0.002
-- pure speckle -- and that single measurement predicted the entire result: a
phase-blind magnitude model beat both phase-carrying models by 5-7 points.

PolSAR should behave differently, and this script tests whether it actually
does. Two measurements decide it.

1. Polarimetric coherence between channels i and j:

       gamma_ij = |<z_i conj(z_j)>| / sqrt(<|z_i|^2> <|z_j|^2>)

   Near 0 means the two channels are phase-unrelated and there is nothing to
   difference against; near 1 means their relative phase is stable and carries
   information. This is the quantity SAMPLE structurally cannot have, having
   only one channel.

2. Co-polar phase difference, arg(T12) or arg<S_HH conj(S_VV)>, split BY CLASS.
   Physics says roughly 0 degrees for surface scattering and 180 for
   double-bounce, so classes dominated by different mechanisms should occupy
   different parts of the circle.

The decisive output is the last one: if the per-class phase distributions are
separable, a complex-valued network has real signal to exploit and the
experiment is worth running. If they all look uniform, PolSAR is SAMPLE again
and no architecture will help.

Usage:
    python -m polsar.coherence_audit <product_dir> [--labels labels.npy]
    python -m polsar.coherence_audit --self-test
"""

import argparse
from pathlib import Path

import numpy as np

from .envi import describe, load_coherency


def circular_stats(angles):
    """Mean direction and resultant length. R near 0 = uniform, near 1 = peaked."""
    resultant = np.exp(1j * angles).mean()
    return np.angle(resultant), np.abs(resultant)


def coherence(matrix, names, i, j):
    """gamma between channels i and j, from coherency-matrix elements."""
    index = {n: k for k, n in enumerate(names)}
    off = index.get("T%d%d" % (i, j))
    ii, jj = index.get("T%d%d" % (i, i)), index.get("T%d%d" % (j, j))
    if off is None or ii is None or jj is None:
        return None, None
    power = np.sqrt(np.abs(matrix[ii]) * np.abs(matrix[jj])) + 1e-12
    return np.abs(matrix[off]) / power, np.angle(matrix[off])


def report(matrix, names, labels=None, class_names=None):
    print("\n1. Polarimetric coherence between channel pairs")
    print("   0 = phase-unrelated (nothing to exploit), 1 = stable relative phase")
    found = False
    for i in range(1, 7):
        for j in range(i + 1, 7):
            gamma, _ = coherence(matrix, names, i, j)
            if gamma is None:
                continue
            found = True
            print("   T%d%d   mean gamma = %.3f   median = %.3f" % (i, j, gamma.mean(), np.median(gamma)))
    if not found:
        print("   no off-diagonal elements -- this product has no usable phase")
        return

    print("\n2. Per-pixel phase of T12, pooled over the whole scene")
    _, phase = coherence(matrix, names, 1, 2)
    _, resultant = circular_stats(phase.ravel())
    counts, _ = np.histogram(phase, bins=12, range=(-np.pi, np.pi))
    fractions = counts / counts.sum()
    print("   12-bin histogram: %s" % np.round(fractions, 3))
    print("   max deviation from uniform (1/12 = %.4f): %.4f"
          % (1 / 12, np.abs(fractions - 1 / 12).max()))
    print("   circular resultant R = %.3f" % resultant)
    if resultant < 0.05:
        print("   Pooled phase looks uniform -- but do NOT read that as 'no signal'.")
        print("   Opposite mechanisms cancel when pooled: a scene that is half")
        print("   surface (0 deg) and half double-bounce (180 deg) pools to R=0")
        print("   while being perfectly separable. Only the per-class test below")
        print("   decides. This is where the SAMPLE analogy breaks down, because")
        print("   there the uniformity was within every class, not between them.")

    if labels is None:
        print("\n   No labels supplied, so the decisive per-class test is skipped.")
        return

    print("\n3. Co-polar phase difference BY CLASS -- the decisive test")
    print("   %-16s %8s %10s %10s" % ("class", "pixels", "mean deg", "R"))
    rows = []
    for value in np.unique(labels):
        if value == 0:            # 0 is the unlabelled class in every candidate
            continue
        mask = labels == value
        if mask.sum() < 100:
            continue
        mean, R = circular_stats(phase[mask])
        name = class_names[value] if class_names and value in class_names else "class %d" % value
        rows.append((name, mask.sum(), np.rad2deg(mean), R))
        print("   %-16s %8d %10.1f %10.3f" % rows[-1])

    if len(rows) > 1:
        spread = max(r[2] for r in rows) - min(r[2] for r in rows)
        print("\n   spread of per-class mean phase: %.1f degrees" % spread)
        if spread > 30 and max(r[3] for r in rows) > 0.1:
            print("   -> classes separate in phase. A CVNN has real signal here.")
        else:
            print("   -> classes do NOT separate in phase. This is SAMPLE again;")
            print("      expect the magnitude baseline to win and plan accordingly.")


def self_test():
    """
    Synthetic scene with known answer, so the script can be trusted before real
    data arrives. Two classes sharing a random speckle phase per pixel: one with
    a 0 degree co-pol phase difference (surface), one with 180 (double bounce).
    A correct implementation must recover that 180 degree separation despite the
    speckle, because conj() cancels the common term.
    """
    rng = np.random.default_rng(0)
    h = w = 128
    labels = np.where(rng.random((h, w)) < 0.5, 1, 2)

    speckle = np.exp(1j * rng.uniform(-np.pi, np.pi, (h, w)))
    true_phase = np.where(labels == 1, 0.0, np.pi)
    hh = speckle * rng.rayleigh(1.0, (h, w))
    vv = speckle * np.exp(1j * -true_phase) * rng.rayleigh(1.0, (h, w))

    # form a 2x2 coherency-style product with a small averaging window
    def boxcar(a, k=5):
        pad = k // 2
        padded = np.pad(a, pad, mode="reflect")
        out = np.zeros_like(a)
        for dy in range(k):
            for dx in range(k):
                out += padded[dy:dy + h, dx:dx + w]
        return out / (k * k)

    matrix = np.stack([boxcar(np.abs(hh) ** 2).astype(np.complex64),
                       boxcar(hh * np.conj(vv)).astype(np.complex64),
                       boxcar(np.abs(vv) ** 2).astype(np.complex64)])
    names = ["T11", "T12", "T22"]

    print("SELF TEST -- synthetic scene, class 1 at 0 deg, class 2 at 180 deg")
    report(matrix, names, labels, {1: "surface", 2: "double-bounce"})
    print("\nExpected: ~0 and ~180 recovered, spread near 180 degrees.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", nargs="?", help="PolSAR product directory of .bin/.hdr files")
    ap.add_argument("--labels", help=".npy label map aligned to the scene")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test or not args.directory:
        self_test()
        return

    describe(args.directory)
    matrix, names = load_coherency(args.directory)
    labels = np.load(args.labels) if args.labels else None
    if labels is not None and labels.shape != matrix.shape[1:]:
        raise ValueError("labels %s do not match scene %s" % (labels.shape, matrix.shape[1:]))
    report(matrix, names, labels)


if __name__ == "__main__":
    main()
