"""Dataset loading (with SHA-256 verification) and the in-memory DB representation.

A DB is a list `tx` of dicts {item (int): iu (int)}; the transaction index k (0-based) is the tid,
printed as T(k+1). Items are ints so every tie-break can use the integer item ID.
Weights (external utilities) are {item (int): eu (float)}.
"""

import gzip
import hashlib
import io
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "data")
DATASET_DIR = os.path.join(DATA_DIR, "datasets")
UTILITY_DIR = os.path.join(DATA_DIR, "utilities")
VARIANT_DIR = os.path.join(UTILITY_DIR, "variants")
CHECKSUMS = os.path.join(DATA_DIR, "checksums.txt")

# dataset -> (internal utilities file, external utilities file)
DATASETS = {
    "chess": ("chess_fimi_quantities.txt.gz", "chess_fimi_weights.txt"),
    "mushroom": ("mushroom_quantities.txt.gz", "mushroom_weights.txt"),
    "accident": ("accident_quantities.txt.gz", "accident_weights.txt"),
    "retail": ("retail_quantities.txt.gz", "retail_weights.txt"),
    "kosarak": ("kosarak_quantities.txt.gz", "kosarak_weights.txt"),
    "bms-pos": ("bms-pos_quantities.txt.gz", "bms-pos_weights.txt"),
    "chainstore": ("chainstore_quantities.txt.gz", "chainstore_weights.txt"),
}
# thresholds of the main comparison (xi at multiplier 1.0)
XI = {
    "chess": 0.89,
    "mushroom": 0.4,
    "accident": 0.75,
    "retail": 0.014,
    "kosarak": 0.022,
    "bms-pos": 0.05,
    "chainstore": 0.003,
}


def xi_of(ds, mult):
    """xi = main-comparison xi x multiplier."""
    return XI[ds] * float(mult)


# ------------------------------------------------------------------ checksums
_SUMS = None


def expected_sha256(path):
    """SHA-256 of the uncompressed content of a data file, from data/checksums.txt."""
    global _SUMS
    if _SUMS is None:
        _SUMS = {}
        with open(CHECKSUMS) as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    content, _stored, rel = line.split()
                    _SUMS[rel] = content
    rel = os.path.relpath(os.path.abspath(path), REPO)
    if rel not in _SUMS:
        raise RuntimeError(f"{rel} is not listed in data/checksums.txt")
    return _SUMS[rel]


def read_verified(path):
    """Uncompressed bytes of a data file (.gz is decompressed); raises if the SHA-256 of the
    content differs from data/checksums.txt."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        raw = f.read()
    if hashlib.sha256(raw).hexdigest() != expected_sha256(path):
        raise RuntimeError(f"SHA-256 mismatch for {os.path.relpath(path, REPO)}")
    return raw


def _lines(raw):
    return io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8")


# ------------------------------------------------------------------ parsers
def load_transactions(path):
    """One transaction per line, `item:iu` tokens separated by blanks (a bare item has iu 1)."""
    tx = []
    for line in _lines(read_verified(path)):
        items = {}
        for part in line.strip().split():
            if ":" in part:
                item, qty = part.split(":")
                items[int(item)] = int(qty)
            else:
                items[int(part)] = 1
        tx.append(items)
    return tx


def load_weights(path):
    """One `item:eu` (or `item,eu`) per line."""
    w = {}
    for line in _lines(read_verified(path)):
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            parts = line.split(":")
        elif "," in line:
            parts = line.split(",")
        else:
            continue
        if len(parts) == 2:
            w[int(parts[0])] = float(parts[1])
    return w


def weights_path(name):
    """Path of a utility table named in a workload file: a bare file name is a dataset's own
    table (data/utilities/); a name with a directory part is a generated variant
    (data/utilities/variants/)."""
    if "/" in name:
        return os.path.join(VARIANT_DIR, os.path.basename(name))
    return os.path.join(UTILITY_DIR, name)


def load_db(ds, weights_file=None):
    qf, wf = DATASETS[ds]
    tx = load_transactions(os.path.join(DATASET_DIR, qf))
    w = load_weights(weights_file or os.path.join(UTILITY_DIR, wf))
    return tx, w


# ------------------------------------------------------------------ twu, tidsets
def tu_of(t, w):
    return sum(w.get(i, 0.0) * q for i, q in t.items())


def twu_of(t, w):
    """twu(T) = TU(T) / |S(T)|, |S(T)| = number of items with iu > 0."""
    if not t:
        return 0.0
    return sum(w.get(i, 0.0) * q for i, q in t.items()) / len(t)


def twu_array(tx, w):
    return np.fromiter((twu_of(t, w) for t in tx), dtype=np.float64, count=len(tx))


def item_index(tx):
    """item -> sorted np.int64 array of transaction indices containing it."""
    idx = {}
    for k, t in enumerate(tx):
        for i in t:
            idx.setdefault(i, []).append(k)
    return {i: np.asarray(v, dtype=np.int64) for i, v in idx.items()}


def tidset(pattern, iidx):
    """Sorted index array of the transactions containing every item of `pattern`
    (smallest posting list first)."""
    arrs = sorted((iidx.get(i, np.empty(0, np.int64)) for i in pattern), key=len)
    ts = arrs[0]
    for a in arrs[1:]:
        if ts.size == 0:
            break
        ts = np.intersect1d(ts, a, assume_unique=True)
    return ts


def ptuple(p):
    """Canonical pattern key: tuple of int item IDs, ascending."""
    return tuple(sorted(int(i) for i in p))


# ------------------------------------------------------------------ running example
# The paper's running example (Tables 2-3, xi = 0.4); items A..E -> 1..5
PAPER_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
PAPER_INV = {v: k for k, v in PAPER_MAP.items()}
_PAPER_DB = [
    {"A": 2, "C": 5, "D": 1, "E": 2},
    {"B": 2, "D": 1, "E": 1},
    {"A": 3, "B": 1, "C": 2, "D": 1, "E": 1},
    {"A": 2, "D": 2, "E": 1},
    {"A": 1, "C": 2, "D": 3},
]
_PAPER_W = {"A": 0.4, "B": 0.6, "C": 0.2, "D": 0.5, "E": 0.3}


def paper_example():
    tx = [{PAPER_MAP[i]: q for i, q in t.items()} for t in _PAPER_DB]
    w = {PAPER_MAP[i]: v for i, v in _PAPER_W.items()}
    return tx, w


def letters(p):
    return "".join(PAPER_INV[i] for i in sorted(p))
