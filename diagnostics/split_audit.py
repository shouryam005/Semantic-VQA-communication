"""
Audit of how the SAMPLE SAR-VQA train/val/test splits were made.

The splits come from exploring_datasets.ipynb cell 13: a uniformly random
80/10/10 split over the 2690 unique .mat filepaths.

SAMPLE filenames encode the acquisition geometry:

    zsu23_synth_A_elevDeg_015_azCenter_014_99_serial_d08.mat
    ^target    ^real/synth      ^elevation  ^azimuth

Two images of the same target at the same elevation and adjacent azimuth are
near-identical. A random split over filepaths therefore scatters
near-duplicates across train and val, which inflates validation accuracy.

Run:  python diagnostics/split_audit.py
"""

import collections
import pickle
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

FNAME = re.compile(r"(.+?)_(real|synth)_A_elevDeg_(\d+)_azCenter_(\d+)")


def load(name):
    with open(DATA / name, "rb") as f:
        return pickle.load(f)


def basename(path):
    return re.split(r"[\/]", path)[-1]


def geometry(path):
    """(target, source, elevation, azimuth) parsed from the filename."""
    m = FNAME.match(basename(path))
    return (m.group(1), m.group(2), m.group(3), int(m.group(4)))


def main():
    splits = {
        "train": load("train_processed.pkl"),
        "val": load("val_processed.pkl"),
        "test": load("test_processed.pkl"),
    }
    images = {k: {s["filepath"] for s in v} for k, v in splits.items()}

    print("A. Paired real/synthetic leakage")
    print("   Does a val image's twin (same target, elevation, azimuth, other source)")
    print("   appear in train?")
    scenes = {k: {re.sub(r"_(real|synth)_", "_X_", basename(p)) for p in v} for k, v in images.items()}
    for split in ("val", "test"):
        n = len(scenes["train"] & scenes[split])
        print(f"     {split}: {n}/{len(images[split])} ({100 * n / len(images[split]):.1f}%)")
    print()

    print("B. Near-duplicate azimuth leakage")
    print("   Does a val image have a train image of the same target and elevation")
    print("   at a nearby azimuth?")
    train_az = collections.defaultdict(set)
    for p in images["train"]:
        target, _, elev, az = geometry(p)
        train_az[(target, elev)].add(az)

    for tol in (0, 1, 2, 3):
        parts = []
        for split in ("val", "test"):
            n = 0
            for p in images[split]:
                target, _, elev, az = geometry(p)
                if any(abs(az - z) <= tol for z in train_az.get((target, elev), ())):
                    n += 1
            parts.append(f"{split}={n:3d}/{len(images[split])} ({100 * n / len(images[split]):5.1f}%)")
        print(f"     within {tol} deg azimuth:  " + "   ".join(parts))
    print()

    print("C. Real / synthetic composition")
    print("   SAMPLE's own protocol is train-on-synthetic, test-on-real, which")
    print("   measures the sim-to-real gap. These splits mix both, which is easier.")
    for name, paths in images.items():
        c = collections.Counter(geometry(p)[1] for p in paths)
        print(f"     {name:5s} {dict(c)}")


if __name__ == "__main__":
    main()
