"""
Collect the experiment matrix into one table with significance tests.

Every comparison is against a stated reference, because a bare accuracy on 718
validation samples is not interpretable on its own: chance is 50%, seed spread
is roughly +/-2.5 points, and the effects being chased are a few points wide.

Run:  python diagnostics/summarize.py
"""

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent


def load(pattern):
    runs = {}
    for path in sorted(glob.glob(str(ROOT / pattern))):
        name = os.path.basename(path).rsplit(".", 1)[0]
        with open(path) as f:
            blob = json.load(f)
        runs[name] = {
            "acc": np.array([100 * r["accuracy"] for r in blob["results"]]),
            "args": blob["args"],
        }
    return runs


def describe(values):
    mean = values.mean()
    if len(values) < 2:
        return mean, 0.0, (mean, mean)
    return mean, values.std(ddof=1), stats.t.interval(0.95, len(values) - 1,
                                                      mean, stats.sem(values))


def compare(a, b):
    """Welch t-test plus Cohen's d. Returns (diff, p, d)."""
    t, p = stats.ttest_ind(a, b, equal_var=False)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return a.mean() - b.mean(), p, (a.mean() - b.mean()) / pooled if pooled else 0.0


def table(title, runs, order, reference=None):
    print("\n" + title)
    print("  %-22s %7s %7s %-16s %8s" % ("run", "mean", "sd", "95% CI", "n"))
    print("  " + "-" * 64)
    for name in order:
        if name not in runs:
            continue
        mean, sd, ci = describe(runs[name]["acc"])
        print("  %-22s %7.2f %7.2f  [%5.2f, %5.2f] %7d"
              % (name, mean, sd, ci[0], ci[1], len(runs[name]["acc"])))

    if reference and reference in runs:
        print("\n  versus %s" % reference)
        for name in order:
            if name == reference or name not in runs:
                continue
            diff, p, d = compare(runs[name]["acc"], runs[reference]["acc"])
            mark = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print("    %-22s %+6.2f pp   p=%.4f  d=%+.2f  %s" % (name, diff, p, d, mark))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="results/m_*.json")
    args = ap.parse_args()

    runs = load(args.pattern)
    if not runs:
        print("no results matching %s" % args.pattern)
        return

    strip = lambda n: n[2:] if n.startswith("m_") else n
    runs = {strip(k): v for k, v in runs.items()}

    table("A. Baselines under global normalization (image path only)",
          runs, ["base-rvnn", "base-cvnn"], reference="base-rvnn")

    table("B. Complex pooling: does the collapse destroy phase?",
          runs, ["base-cvnn", "pool-coherence", "pool-modulus"], reference="base-cvnn")

    table("C. Activation, with phase-preserving pooling",
          runs, ["pool-coherence", "act-cardioid", "act-modrelu", "act-zrelu"],
          reference="pool-coherence")

    table("D. Same activation under the old cancelling pooling",
          runs, ["base-cvnn", "avg-cardioid"], reference="base-cvnn")

    table("E. Semantic communication, and question conditioning",
          runs, ["chan-rvnn", "chan-cvnn", "chan-rvnn-film", "chan-cvnn-film"],
          reference="chan-rvnn")

    # headline comparison: best complex configuration against the real baseline
    complex_runs = [n for n in runs if n.startswith(("base-cvnn", "pool-", "act-", "avg-"))]
    if complex_runs and "base-rvnn" in runs:
        best = max(complex_runs, key=lambda n: runs[n]["acc"].mean())
        diff, p, d = compare(runs["base-rvnn"]["acc"], runs[best]["acc"])
        print("\nHeadline")
        print("  best complex configuration: %s at %.2f%%" % (best, runs[best]["acc"].mean()))
        print("  real baseline:              base-rvnn at %.2f%%" % runs["base-rvnn"]["acc"].mean())
        print("  RVNN advantage %+.2f pp  p=%.4f  d=%+.2f -> %s"
              % (diff, p, d, "real still ahead" if p < 0.05 and diff > 0
                 else "no significant difference"))
    print("\n  * p<0.05   ** p<0.01   *** p<0.001   (Welch t-test, two-sided)")


if __name__ == "__main__":
    main()
