"""
Rebuild the SAMPLE SAR-VQA benchmark.

Replaces exploring_datasets.ipynb cells 7-21. Three defects in that version
are fixed here; see diagnostics/ for the measurements that motivate each.

1. Per-question answer balance.
   The old generator emitted, per image, one positive and one negative class
   question plus one positive and one negative mobility question. That
   balances answers globally and per image, but not per question: only two
   mobility classes exist and 2248 of 2690 images are tracked, so
   "Is this a tracked vehicle?" was 83.7% yes. A question-only classifier
   scored 69.1% on val -- the same as the reported CVNN result.

   Here every question string is balanced to exactly 50/50 by construction,
   which caps the question-only baseline at 50%.

2. Balance is enforced inside each split.
   The old pipeline generated QA first and split afterwards, so nothing
   constrained the per-question prior within train or val. Here images are
   split first and QA is generated independently per split.

3. Near-duplicate leakage across splits.
   The old split was uniformly random over filepaths. SAMPLE filenames encode
   target, elevation and azimuth, and images 1 degree apart in azimuth are
   near-identical: 90% of val images had a train counterpart within 1 degree.
   The default split here assigns contiguous azimuth arcs per
   (target, elevation) group and drops train images inside a guard band around
   the held-out arcs.

Usage:
    python benchmark/build_benchmark.py
    python benchmark/build_benchmark.py --split sample-protocol
    python benchmark/build_benchmark.py --mobility drop --guard 10
"""

import argparse
import collections
import pickle
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TARGET_TYPE = {
    "m60_tank": "tank", "m1_tank": "tank", "m2_tank": "tank",
    "t72_tank": "tank", "bmp2_tank": "tank",
    "2s1_gun": "artillery", "zsu23-4_gun": "artillery",
    "m35_truck": "truck",
    "m548_transport": "transport", "btr70_transport": "transport",
}

MOBILITY = {
    "m60_tank": "tracked", "m1_tank": "tracked", "m2_tank": "tracked",
    "t72_tank": "tracked", "bmp2_tank": "tracked",
    "2s1_gun": "tracked", "zsu23-4_gun": "tracked", "m548_transport": "tracked",
    "m35_truck": "wheeled", "btr70_transport": "wheeled",
}

FNAME = re.compile(r"(.+?)_(real|synth)_A_elevDeg_(\d+)_azCenter_(\d+)")


# --------------------------------------------------------------------------
# image inventory
# --------------------------------------------------------------------------

def parse(path):
    """Pull source and geometry out of a SAMPLE filename."""
    name = re.split(r"[\\/]", path)[-1]
    m = FNAME.match(name)
    if m is None:
        raise ValueError("unparseable SAMPLE filename: " + name)
    return {
        "filepath": path,
        "source": m.group(2),
        "elevation": int(m.group(3)),
        "azimuth": int(m.group(4)),
    }


def inventory(metadata_pkl):
    """One record per image, with target and mobility labels attached."""
    with open(metadata_pkl, "rb") as f:
        md = pickle.load(f)

    records = []
    for path, target in zip(md["filepath"], md["target_name"]):
        rec = parse(path)
        rec["target_name"] = target
        rec["target_type"] = TARGET_TYPE[target]
        rec["mobility"] = MOBILITY[target]
        records.append(rec)
    return records


# --------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------

def circular_gap(a, b, period=360):
    d = abs(a - b) % period
    return min(d, period - d)


def split_azimuth_blocked(records, guard, seed, val_frac=0.1, test_frac=0.1):
    """
    Contiguous azimuth arcs per (target, elevation), with a guard band.

    Within a group the images are ordered by azimuth and cut into three arcs.
    Any train image within `guard` degrees of a val or test image (same group,
    circular distance) is dropped, so no held-out image has a near-duplicate
    sitting in train.
    """
    rng = random.Random(seed)
    groups = collections.defaultdict(list)
    for r in records:
        groups[(r["target_name"], r["elevation"])].append(r)

    train, val, test, dropped = [], [], [], []

    for key in sorted(groups):
        items = sorted(groups[key], key=lambda r: r["azimuth"])
        n = len(items)
        if n > 2:
            # rotate so the same azimuths are not always the held-out ones
            k = rng.randrange(n)
            items = items[k:] + items[:k]

        n_val = max(1, int(round(n * val_frac)))
        n_test = max(1, int(round(n * test_frac)))
        held_val = items[:n_val]
        held_test = items[n_val:n_val + n_test]
        rest = items[n_val + n_test:]

        held_az = [r["azimuth"] for r in held_val + held_test]
        for r in rest:
            if any(circular_gap(r["azimuth"], a) <= guard for a in held_az):
                dropped.append(r)
            else:
                train.append(r)
        val.extend(held_val)
        test.extend(held_test)

    return {"train": train, "val": val, "test": test}, dropped


def split_sample_protocol(records, seed, val_frac=0.15):
    """
    The protocol SAMPLE was published for: train on synthetic, test on measured.

    This measures the sim-to-real gap rather than in-distribution accuracy, and
    is the harder and more standard benchmark for this dataset.
    """
    rng = random.Random(seed)
    synth = [r for r in records if r["source"] == "synth"]
    real = [r for r in records if r["source"] == "real"]
    rng.shuffle(synth)
    cut = int(len(synth) * (1 - val_frac))
    return {"train": synth[:cut], "val": synth[cut:], "test": real}, []


def split_random(records, seed):
    """The original scheme, kept so the leakage can be reproduced on demand."""
    rng = random.Random(seed)
    items = list(records)
    rng.shuffle(items)
    n = len(items)
    a, b = int(n * 0.8), int(n * 0.9)
    return {"train": items[:a], "val": items[a:b], "test": items[b:]}, []


# --------------------------------------------------------------------------
# question generation
# --------------------------------------------------------------------------

def generate_qa(records, rng, include_mobility=True):
    """
    Emit Yes/No questions such that every question string is exactly 50/50.

    Identity questions: for a class C holding n images, emit those n as "yes"
    and sample n non-C images as "no".

    Mobility questions: the positive and negative pools differ greatly in size
    (2248 tracked vs 442 wheeled), so both are subsampled to the smaller pool.
    Balance is bought with fewer instances rather than left skewed.
    """
    qa = []

    by_target = collections.defaultdict(list)
    for r in records:
        by_target[r["target_name"]].append(r)

    for target in sorted(by_target):
        positives = by_target[target]
        others = [r for r in records if r["target_name"] != target]
        if not others:
            continue
        negatives = rng.sample(others, min(len(positives), len(others)))
        positives = rng.sample(positives, len(negatives))

        question = "Is this a " + target.replace("_", " ") + "?"
        for r in positives:
            qa.append({"filepath": r["filepath"], "question": question, "answer": "yes"})
        for r in negatives:
            qa.append({"filepath": r["filepath"], "question": question, "answer": "no"})

    if include_mobility:
        by_mob = collections.defaultdict(list)
        for r in records:
            by_mob[r["mobility"]].append(r)

        for mob in sorted(by_mob):
            positives = by_mob[mob]
            negatives = [r for r in records if r["mobility"] != mob]
            if not negatives:
                continue
            k = min(len(positives), len(negatives))
            question = "Is this a " + mob + " vehicle?"
            for r in rng.sample(positives, k):
                qa.append({"filepath": r["filepath"], "question": question, "answer": "yes"})
            for r in rng.sample(negatives, k):
                qa.append({"filepath": r["filepath"], "question": question, "answer": "no"})

    rng.shuffle(qa)
    return qa


# --------------------------------------------------------------------------
# tokenization
# --------------------------------------------------------------------------

def build_vocab(train_qa):
    counter = collections.Counter()
    for row in train_qa:
        counter.update(row["question"].lower().split())
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for word in sorted(counter):
        word2idx[word] = len(word2idx)
    return word2idx


def encode(question, word2idx, max_len):
    tokens = [word2idx.get(w, word2idx["<UNK>"]) for w in question.lower().split()]
    tokens = tokens[:max_len]
    return tokens + [word2idx["<PAD>"]] * (max_len - len(tokens))


# --------------------------------------------------------------------------

def report(name, qa):
    per_q = collections.defaultdict(collections.Counter)
    for row in qa:
        per_q[row["question"]][row["answer"]] += 1

    total = collections.Counter(r["answer"] for r in qa)
    worst_q, worst_c = max(
        per_q.items(),
        key=lambda kv: abs(kv[1]["yes"] / sum(kv[1].values()) - 0.5),
    )
    worst_rate = worst_c["yes"] / sum(worst_c.values())

    # always answering each question with its own majority
    ceiling = sum(max(c.values()) for c in per_q.values()) / len(qa)

    images = len({r["filepath"] for r in qa})
    print("  %-5s n=%5d  images=%4d  yes/no=%d/%d  questions=%d"
          % (name, len(qa), images, total["yes"], total["no"], len(per_q)))
    print("        most skewed question: P(yes)=%.3f  %r" % (worst_rate, worst_q))
    print("        question-only ceiling = %.2f%%" % (100 * ceiling))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["azimuth-blocked", "sample-protocol", "random"],
                    default="azimuth-blocked")
    ap.add_argument("--mobility", choices=["balanced", "drop"], default="balanced")
    ap.add_argument("--guard", type=int, default=5,
                    help="degrees of azimuth separation between train and held-out images")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data_v2")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(exist_ok=True)

    records = inventory(ROOT / "sample_metadata.pkl")
    print("%d images, %d target classes"
          % (len(records), len({r["target_name"] for r in records})))
    print("split=%s  mobility=%s  guard=%ddeg  seed=%d"
          % (args.split, args.mobility, args.guard, args.seed))
    print()

    if args.split == "azimuth-blocked":
        splits, dropped = split_azimuth_blocked(records, args.guard, args.seed)
    elif args.split == "sample-protocol":
        splits, dropped = split_sample_protocol(records, args.seed)
    else:
        splits, dropped = split_random(records, args.seed)

    print("Image split")
    for name in ("train", "val", "test"):
        comp = collections.Counter(r["source"] for r in splits[name])
        print("  %-5s %4d images  %s" % (name, len(splits[name]), dict(comp)))
    if dropped:
        print("  dropped %d train images inside the %ddeg guard band" % (len(dropped), args.guard))
    print()

    rng = random.Random(args.seed)
    qa = {name: generate_qa(rows, rng, include_mobility=(args.mobility == "balanced"))
          for name, rows in splits.items()}

    print("Generated benchmark")
    for name in ("train", "val", "test"):
        report(name, qa[name])
    print()

    word2idx = build_vocab(qa["train"])
    idx2word = {v: k for k, v in word2idx.items()}
    max_len = max(len(r["question"].split()) for r in qa["train"])

    for name in ("train", "val", "test"):
        processed = [{
            "filepath": r["filepath"],
            "question_tokens": encode(r["question"], word2idx, max_len),
            "answer": 1 if r["answer"] == "yes" else 0,
        } for r in qa[name]]
        with open(out / (name + "_processed.pkl"), "wb") as f:
            pickle.dump(processed, f)

    with open(out / "word2idx.pkl", "wb") as f:
        pickle.dump(word2idx, f)
    with open(out / "idx2word.pkl", "wb") as f:
        pickle.dump(idx2word, f)
    config = dict(vars(args))
    config["max_len"] = max_len
    with open(out / "config.pkl", "wb") as f:
        pickle.dump(config, f)

    print("vocab=%d words, max question length=%d" % (len(word2idx), max_len))
    print("written to %s/" % out.name)


if __name__ == "__main__":
    main()
