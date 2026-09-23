"""
Training and evaluation for the SAR-VQA models.

Everything the notebooks left implicit is explicit here.

  Seeds. The notebooks seeded nothing, so no run was reproducible and no
  difference between two runs could be attributed. --seed controls torch,
  numpy, python random and the DataLoader workers; --repeats runs the whole
  thing several times and reports mean +/- std, because a 2-point gap on 718
  validation samples is inside the noise of a single run.

  Equal training budget. The reported 89%-vs-69% compared a 20-epoch RVNN
  against a CVNN that was interrupted after 3 epochs. Both get --epochs here.

  Metrics beyond accuracy. Balanced accuracy, per-class recall and the
  per-question breakdown are printed, so a model that has collapsed onto the
  answer prior is visible rather than hidden behind a single number.

Usage:
    python -m sarvqa.train --model rvnn
    python -m sarvqa.train --model cvnn --no-channel --repeats 3
    python -m sarvqa.train --model question-only --epochs 10
"""

import argparse
import collections
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import ImageCache, SARVQADataset, global_scale, load_split, load_vocab
from .models import MODELS, count_parameters

ROOT = Path(__file__).resolve().parent.parent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device, idx2word=None):
    """Returns accuracy, balanced accuracy, per-class recall, per-question accuracy."""
    model.eval()
    confusion = np.zeros((2, 2), dtype=int)
    per_question = collections.defaultdict(lambda: [0, 0])

    with torch.no_grad():
        for images, questions, answers in loader:
            images, questions = images.to(device), questions.to(device)
            predictions = model(images, questions).argmax(1).cpu()

            for a, p in zip(answers.tolist(), predictions.tolist()):
                confusion[a, p] += 1

            if idx2word is not None:
                for q, a, p in zip(questions.cpu().tolist(), answers.tolist(), predictions.tolist()):
                    key = " ".join(idx2word[t] for t in q if t != 0)
                    per_question[key][0] += int(a == p)
                    per_question[key][1] += 1

    accuracy = confusion.trace() / confusion.sum()
    recalls = [confusion[c, c] / confusion[c].sum() if confusion[c].sum() else 0.0 for c in (0, 1)]
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)),
        "recall_no": recalls[0],
        "recall_yes": recalls[1],
        "predicted_yes_rate": confusion[:, 1].sum() / confusion.sum(),
        "per_question": {k: v[0] / v[1] for k, v in sorted(per_question.items())},
    }


SNR_SWEEP = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0]


def sweep_snr(model, loader, device, points=SNR_SWEEP):
    """
    Evaluate one already-trained model across test SNRs.

    Training at a single SNR and testing across a range is nearly free -- it is
    an evaluation pass, not a retrain -- and mismatched train/test SNR is a
    standard robustness test. A single operating point cannot establish
    robustness: at high SNR every architecture works, so differences only show
    where noise dominates the transmitted vector.
    """
    if not getattr(model, "use_channel", False):
        return {}
    original = model.channel.snr_db
    curve = {}
    for snr in points:
        model.channel.snr_db = snr
        curve[snr] = evaluate(model, loader, device)["accuracy"]
    model.channel.snr_db = original
    return curve


def run_once(args, datasets, vocab, device, seed):
    set_seed(seed)

    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False)

    model = MODELS[args.model](
        vocab_size=len(vocab),
        use_channel=not args.no_channel,
        snr_db=args.snr,
        activation=args.activation,
        pooling=args.pooling,
        film=args.film,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, questions, answers in train_loader:
            images, questions, answers = images.to(device), questions.to(device), answers.to(device)
            optimizer.zero_grad()
            logits = model(images, questions)
            loss = criterion(logits, answers)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(1) == answers).sum().item()
            total += answers.size(0)

        train_acc = 100 * correct / total
        val = evaluate(model, val_loader, device)
        history.append({"epoch": epoch + 1, "loss": total_loss / len(train_loader),
                        "train_acc": train_acc, "val_acc": 100 * val["accuracy"]})

        if args.verbose:
            print("    epoch %2d/%d  loss %.4f  train %5.2f%%  val %5.2f%%"
                  % (epoch + 1, args.epochs, history[-1]["loss"], train_acc, 100 * val["accuracy"]))

    return model, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="rvnn")
    ap.add_argument("--data", default="data_v2")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--snr", type=float, default=10.0)
    ap.add_argument("--no-channel", action="store_true",
                    help="disable semantic encoder/AWGN/decoder; isolates the image path")
    ap.add_argument("--normalize", choices=["global", "per-image", "log", "db", "none"],
                    default="global")
    ap.add_argument("--activation", choices=["crelu", "modrelu", "zrelu", "cardioid"],
                    default="crelu", help="CVNN only")
    ap.add_argument("--pooling", choices=["avg", "coherence", "modulus"], default="avg",
                    help="CVNN only; how the complex spatial map is collapsed")
    ap.add_argument("--film", action="store_true",
                    help="condition the semantic encoder on the question")
    ap.add_argument("--limit-train", type=int, default=None,
                    help="subsample training QA pairs to this many; use when comparing "
                         "benchmarks whose guard bands leave different amounts of data, "
                         "so leakage is not confounded with training set size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=None, help="write results as JSON here")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = load_vocab(args.data)
    idx2word = {v: k for k, v in vocab.items()}

    print("model=%s  data=%s  channel=%s  norm=%s  act=%s  pool=%s  film=%s"
          % (args.model, args.data, "off" if args.no_channel else "on@%gdB" % args.snr,
             args.normalize, args.activation, args.pooling, args.film))
    print("epochs=%d  seed=%d  repeats=%d  device=%s"
          % (args.epochs, args.seed, args.repeats, device))

    t0 = time.time()
    splits = {name: load_split(args.data, name) for name in ("train", "val", "test")}
    train_paths = [r["filepath"] for r in splits["train"]]
    scale = global_scale(train_paths) if args.normalize in ("global", "log", "db") else None
    if scale is not None:
        print("global normalization scale (train only): %.6f" % scale)
    cache = ImageCache([r["filepath"] for rows in splits.values() for r in rows],
                       normalize=args.normalize, scale=scale)
    if args.limit_train is not None and len(splits["train"]) > args.limit_train:
        rng = random.Random(args.seed)
        splits["train"] = rng.sample(splits["train"], args.limit_train)
        print("subsampled train to %d QA pairs" % len(splits["train"]))
    datasets = {name: SARVQADataset(rows, cache) for name, rows in splits.items()}
    print("cached %d unique images in %.1fs  (train %d / val %d / test %d QA pairs)"
          % (len(cache), time.time() - t0,
             *(len(datasets[n]) for n in ("train", "val", "test"))))

    val_loader = DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False)

    results = []
    for i in range(args.repeats):
        seed = args.seed + i
        print("\n  run %d/%d (seed %d)" % (i + 1, args.repeats, seed))
        model, history = run_once(args, datasets, vocab, device, seed)
        final = evaluate(model, val_loader, device, idx2word)
        final["history"] = history
        final["seed"] = seed
        final["snr_curve"] = sweep_snr(model, val_loader, device)
        results.append(final)
        print("    val acc %.2f%%  balanced %.2f%%  recall(no)=%.3f recall(yes)=%.3f  "
              "predicted-yes %.3f"
              % (100 * final["accuracy"], 100 * final["balanced_accuracy"],
                 final["recall_no"], final["recall_yes"], final["predicted_yes_rate"]))
        if final["snr_curve"]:
            print("    SNR sweep: " + "  ".join(
                "%+gdB=%.1f%%" % (k, 100 * v) for k, v in sorted(final["snr_curve"].items())))

    print()
    print("=" * 72)
    print("%s  |  %d parameters  |  %d epochs  |  %d run(s)"
          % (args.model, count_parameters(MODELS[args.model](
              len(vocab), use_channel=not args.no_channel, activation=args.activation,
              pooling=args.pooling, film=args.film)), args.epochs, args.repeats))
    for key in ("accuracy", "balanced_accuracy"):
        values = np.array([100 * r[key] for r in results])
        print("  %-18s %6.2f%%  +/- %.2f   %s"
              % (key, values.mean(), values.std(ddof=1) if len(values) > 1 else 0.0,
                 np.round(values, 2).tolist()))
    print("  chance                50.00%   (rebuilt benchmark is 50/50 per question)")

    curves = [r["snr_curve"] for r in results if r.get("snr_curve")]
    if curves:
        print("")
        print("  SNR robustness curve (mean over runs, trained at %g dB)" % args.snr)
        for snr in sorted(curves[0]):
            values = np.array([100 * c[snr] for c in curves])
            print("    %+6.1f dB   %6.2f%%  +/- %.2f" % (snr, values.mean(), values.std(ddof=1)
                                                        if len(values) > 1 else 0.0))

    per_q = collections.defaultdict(list)
    for r in results:
        for q, acc in r["per_question"].items():
            per_q[q].append(acc)
    print("\n  per-question accuracy (mean over runs)")
    for q, accs in sorted(per_q.items(), key=lambda kv: -np.mean(kv[1])):
        print("    %-32s %6.2f%%" % (q, 100 * np.mean(accs)))

    if args.out:
        path = ROOT / args.out
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=2)
        print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
