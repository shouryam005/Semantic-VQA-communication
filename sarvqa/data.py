"""
Dataset for the rebuilt SAMPLE SAR-VQA benchmark.

Two changes from the SARVQADataset defined in nb2_2 cell 2 and nb3_2 cell 3.

Caching. The old version called loadmat(filepath) inside __getitem__, so every
epoch re-parsed every .mat from disk. Each image appears in several QA pairs,
so the same file was parsed several times per epoch. Here the unique images are
parsed once into an in-memory array (2690 x 128 x 128 complex64 is ~176 MB) and
indexed thereafter.

Normalization. The old version applied none. diagnostics/data_audit.py measures
mean |x| ~= 0.033 with a 58x spread in peak amplitude across images, so the
first convolution sees inputs two orders of magnitude below unit scale. That
penalizes the complex path more than the real path, because CReLU zeroes the
real and imaginary halves independently and a near-zero input leaves little
signal on either side.

Two normalizations are available, and the choice is not cosmetic:

  "global" (default)
      Divide every image by one constant, the mean modulus over the training
      images. Fixes the scale problem while preserving relative brightness
      between images. SAR amplitude is calibrated -- it is radar cross-section,
      a physical property of the target -- so brightness differences between a
      tank and a truck are real signal.

  "per-image"
      Divide each image by its own mean modulus. Also fixes scale, but discards
      absolute radar cross-section, deleting a legitimate discriminative cue.

  "log" / "db"
      Compress dynamic range, which dividing by a constant does not do.
      data_audit.py measures mean |x| ~= 0.033 against peaks up to 32.8 -- a
      ~1000x spread inside a single image, so a ReLU network sees a handful of
      bright scatterers and near-zero everywhere else. SAR intensity is roughly
      log-normal, and log or dB scaling is standard practice in SAR deep
      learning; the "Convert to dB" step of the usual L1 pipeline. Neither adds
      information (both are monotone per-pixel maps) but both change the
      optimization landscape, which matters at 1706 training images.

      Compression is applied to the MODULUS and the phase is carried through
      unchanged:

          z -> f(|z|) * exp(j arg z)

      so all three models receive the same transformation and the comparison
      stays controlled. "log" uses log1p(|z|/s); "db" uses 20 log10(|z|/s)
      floored at -40 dB and rescaled to [0, 1].

Both preserve phase exactly: dividing a complex number by a positive real
scales its modulus and leaves its argument untouched. The constant for "global"
is computed from the training split only, so no test statistics leak into it.
"""

import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent


def load_split(data_dir, name):
    with open(ROOT / data_dir / (name + "_processed.pkl"), "rb") as f:
        return pickle.load(f)


def load_vocab(data_dir):
    with open(ROOT / data_dir / "word2idx.pkl", "rb") as f:
        return pickle.load(f)


def global_scale(filepaths):
    """Mean modulus over the given images -- pass training paths only."""
    total, count = 0.0, 0
    for path in dict.fromkeys(filepaths):
        x = loadmat(path)["complex_img"]
        total += np.abs(x).sum()
        count += x.size
    return float(total / count)


class ImageCache:
    """Parses each unique .mat once and keeps the complex images in memory."""

    def __init__(self, filepaths, normalize="global", scale=None):
        if normalize in ("global", "log", "db") and scale is None:
            raise ValueError("normalize=%r needs a scale from the training split" % normalize)

        self.index = {}
        images = []
        for path in dict.fromkeys(filepaths):
            x = loadmat(path)["complex_img"].astype(np.complex64)
            if normalize == "global":
                x = x / scale
            elif normalize == "per-image":
                own = np.abs(x).mean()
                if own > 0:
                    x = x / own
            elif normalize in ("log", "db"):
                modulus = np.abs(x)
                phase = np.exp(1j * np.angle(x)).astype(np.complex64)
                if normalize == "log":
                    compressed = np.log1p(modulus / scale)
                else:
                    floor = -40.0
                    decibels = 20.0 * np.log10(np.maximum(modulus / scale, 1e-12))
                    compressed = (np.maximum(decibels, floor) - floor) / (-floor)
                x = (compressed.astype(np.float32) * phase).astype(np.complex64)
            self.index[path] = len(images)
            images.append(x)
        self.images = torch.from_numpy(np.stack(images))

    def __getitem__(self, path):
        return self.images[self.index[path]]

    def __len__(self):
        return len(self.images)


class SARVQADataset(Dataset):
    """
    Yields (complex_image, question_tokens, answer).

    The image stays complex here. The real-valued model splits it into two
    channels itself, so both models are handed byte-identical inputs and no
    information is discarded before the encoder.
    """

    def __init__(self, records, cache):
        self.records = records
        self.cache = cache

    @classmethod
    def from_split(cls, data_dir, name, cache=None, normalize="global"):
        records = load_split(data_dir, name)
        if cache is None:
            paths = [r["filepath"] for r in records]
            scale = global_scale(paths) if normalize in ("global", "log", "db") else None
            cache = ImageCache(paths, normalize=normalize, scale=scale)
        return cls(records, cache)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        return (
            self.cache[record["filepath"]],
            torch.tensor(record["question_tokens"], dtype=torch.long),
            torch.tensor(record["answer"], dtype=torch.long),
        )
