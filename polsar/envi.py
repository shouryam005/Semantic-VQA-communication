"""
Reader for ENVI .bin/.hdr rasters, the format every PolSAR candidate ships in.

Pol-InSAR-Island (RADAR4KIT), Flevoland and the PolSARpro San Francisco scenes
all distribute PolSARpro/ENVI flat binary: one .bin per matrix element plus a
small ASCII .hdr describing shape and dtype. Coherency matrices are stored
element-wise because they are Hermitian -- the diagonal is real, and each
off-diagonal element arrives as a pair of files:

    T11.bin  T22.bin  T33.bin ...        real, the diagonal
    T12_real.bin  T12_imag.bin           one complex off-diagonal element
    T13_real.bin  T13_imag.bin
    ...

`load_coherency` reassembles those into a complex array, which is where the
phase we care about lives: the off-diagonal T_ij carries the relative phase
between polarimetric channels i and j. That relative phase is the quantity
SAMPLE does not have -- there is only one channel there, so nothing to
difference against and the speckle phase stays as noise.

Nothing here assumes a particular sensor or matrix size; T3 (polarimetric) and
T6 (Pol-InSAR) both parse.
"""

import re
from pathlib import Path

import numpy as np

# ENVI type codes -> numpy dtypes, restricted to what PolSAR products use
ENVI_DTYPES = {
    1: np.uint8, 2: np.int16, 3: np.int32, 4: np.float32,
    5: np.float64, 12: np.uint16, 13: np.uint32,
}


def read_header(path):
    """Parse an ENVI .hdr into a dict. Values stay strings except known ints."""
    text = Path(path).read_text(errors="replace")
    # header entries are `key = value`, with braces wrapping multi-line values
    text = re.sub(r"\{[^}]*\}", lambda m: m.group(0).replace("\n", " "), text)

    header = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        header[key.strip().lower()] = value.strip().strip("{}").strip()

    for key in ("samples", "lines", "bands", "data type", "header offset", "byte order"):
        if key in header:
            header[key] = int(header[key])
    return header


def read_band(bin_path, header=None):
    """
    Read one ENVI .bin as a (lines, samples) array.

    Only single-band rasters are expected; PolSAR products store one matrix
    element per file rather than stacking them into bands.
    """
    bin_path = Path(bin_path)
    if header is None:
        header = read_header(bin_path.with_suffix(".hdr"))

    dtype = np.dtype(ENVI_DTYPES[header["data type"]])
    if header.get("byte order", 0) == 1:
        dtype = dtype.newbyteorder(">")

    count = header["lines"] * header["samples"]
    data = np.fromfile(bin_path, dtype=dtype, count=count,
                       offset=header.get("header offset", 0))
    if data.size != count:
        raise ValueError("%s holds %d values, header declares %d"
                         % (bin_path.name, data.size, count))
    return data.reshape(header["lines"], header["samples"])


def find_elements(directory):
    """
    Group the .bin files in a directory into matrix elements.

    Returns {(i, j): [paths]} where a diagonal element has one path and an
    off-diagonal element has its _real and _imag pair.
    """
    pattern = re.compile(r"^[TC](\d)(\d)(?:_(real|imag))?$", re.IGNORECASE)
    elements = {}
    for path in sorted(Path(directory).glob("*.bin")):
        m = pattern.match(path.stem)
        if m is None:
            continue
        elements.setdefault((int(m.group(1)), int(m.group(2))), []).append(path)
    return elements


def load_coherency(directory, dtype=np.complex64):
    """
    Load a full coherency/covariance matrix as a complex array.

    Returns (matrix, names) where matrix has shape (n_elements, lines, samples).
    Diagonal elements come back with a zero imaginary part -- they are real by
    construction, since T_ii is a power -- and off-diagonals carry the relative
    phase between channels.
    """
    elements = find_elements(directory)
    if not elements:
        raise FileNotFoundError("no T*/C*.bin files under %s" % directory)

    planes, names = [], []
    for (i, j) in sorted(elements):
        paths = elements[(i, j)]
        by_part = {p.stem.lower().rsplit("_", 1)[-1]: p for p in paths}

        if "real" in by_part and "imag" in by_part:
            plane = read_band(by_part["real"]) + 1j * read_band(by_part["imag"])
        elif len(paths) == 1:
            plane = read_band(paths[0]).astype(np.float64) + 0j
        else:
            raise ValueError("element T%d%d has an unexpected file set: %s"
                             % (i, j, [p.name for p in paths]))

        planes.append(plane.astype(dtype))
        names.append("T%d%d" % (i, j))

    return np.stack(planes), names


def describe(directory):
    """Print what a PolSAR product directory actually contains."""
    directory = Path(directory)
    elements = find_elements(directory)
    print("%s" % directory)
    print("  matrix elements found: %d  ->  %s"
          % (len(elements), ", ".join("T%d%d" % k for k in sorted(elements))))

    if elements:
        first = elements[sorted(elements)[0]][0]
        header = read_header(first.with_suffix(".hdr"))
        print("  shape: %d lines x %d samples   dtype code %d"
              % (header["lines"], header["samples"], header["data type"]))

    complex_elements = [k for k, v in elements.items() if len(v) == 2]
    print("  complex (off-diagonal) elements: %d" % len(complex_elements))
    if not complex_elements:
        print("  WARNING: no _real/_imag pairs. This product carries no phase;")
        print("  the whole RVNN-vs-CVNN comparison is meaningless on it.")
    return elements
