"""
Audit of the SAMPLE SAR-VQA benchmark.

Answers one question: how much of the reported VQA accuracy can be obtained
without ever looking at the image?

The benchmark is generated in exploring_datasets.ipynb (cell 7). For each of
the 2690 SAMPLE images it emits four Yes/No questions:

    1. "Is this a {true_class}?"      -> yes
    2. "Is this a {random_other}?"    -> no
    3. "Is this a {true_mobility}?"   -> yes
    4. "Is this a {other_mobility}?"  -> no

That construction balances the answers per image (2 yes / 2 no) and globally
(50/50). It does not balance them per question, which is what a model with a
question encoder can actually exploit.

Run:  python diagnostics/benchmark_audit.py
"""

import collections
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def load(name):
    with open(DATA / name, "rb") as f:
        return pickle.load(f)


def question_text(sample, idx2word):
    return " ".join(idx2word[t] for t in sample["question_tokens"] if t != 0)


def main():
    idx2word = load("idx2word_2.pkl")

    splits = {
        "train": load("train_processed.pkl"),
        "val": load("val_processed.pkl"),
        "test": load("test_processed.pkl"),
    }

    # --- sanity: are the splits image-disjoint? -------------------------
    images = {k: {s["filepath"] for s in v} for k, v in splits.items()}

    print("Split sizes and image-level leakage")
    for name, ds in splits.items():
        dist = collections.Counter(s["answer"] for s in ds)
        print(f"  {name:5s} n={len(ds):5d}  images={len(images[name]):4d}  answers={dict(dist)}")
    print(f"  train/val  shared images: {len(images['train'] & images['val'])}")
    print(f"  train/test shared images: {len(images['train'] & images['test'])}")
    print()

    # --- fit the language prior on train only --------------------------
    per_question = collections.defaultdict(collections.Counter)
    for s in splits["train"]:
        per_question[question_text(s, idx2word)][s["answer"]] += 1

    majority = {q: c.most_common(1)[0][0] for q, c in per_question.items()}

    print("Per-question answer skew, measured on train")
    print(f"  {'question':32s} {'n':>6s} {'P(yes)':>8s} {'predict':>8s}")
    for q, c in sorted(per_question.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(c.values())
        print(f"  {q:32s} {n:6d} {c[1] / n:8.3f} {majority[q]:8d}")
    print()

    # --- the actual test -----------------------------------------------
    print("Question-only baseline (no image is ever read)")
    for name, ds in splits.items():
        correct = sum(majority.get(question_text(s, idx2word), 1) == s["answer"] for s in ds)
        print(f"  {name:5s} {100 * correct / len(ds):6.2f}%")
    print()
    print("Reference: reported RVNN val ~89%, reported CVNN val ~69%.")


if __name__ == "__main__":
    main()
