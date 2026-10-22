"""FWUP miner: Eclat on tidsets, no length cap.

A pattern with |wus - xi| < 1e-9 is decided exactly with Fractions (such cases are listed in
`borderline`). Optionally collects the utility u(Y) = sum_{T ⊇ Y} sum_{i in Y} u(i, T) of every
FWUP (Definition 3; used by IUS).
"""

import hashlib
import math

import numpy as np

from . import exact as EX
from .data import ptuple

DENSE_MAX_N = 200_000  # boolean-mask tidsets up to this many transactions, index arrays above


class Miner:
    def __init__(self, tx, w, xi, max_len=None, collect_u=False):
        self.tx, self.w, self.xi = tx, w, xi
        self.max_len = max_len if max_len is not None else 10**9
        self.collect_u = collect_u
        n = len(tx)
        self.N = n
        twu = np.zeros(n, dtype=np.float64)
        D = 0.0
        idx, util = {}, {}
        for k, t in enumerate(tx):
            s = len(t)
            if s > 0:
                utility = sum(w.get(i, 0.0) * q for i, q in t.items())
                tw = utility / s
                D += tw
                twu[k] = tw
                for i, q in t.items():
                    idx.setdefault(i, []).append(k)
                    if collect_u:
                        util.setdefault(i, []).append(w.get(i, 0.0) * q)
        self.twu, self.D = twu, D
        self.item_idx = {i: np.asarray(v, dtype=np.int64) for i, v in idx.items()}
        self.item_util = {i: np.asarray(v, dtype=np.float64) for i, v in util.items()}
        self.dense = n <= DENSE_MAX_N
        self.borderline = []
        self.u = {}
        self._wf = None
        self._dex = None

    def _mk(self, idx):
        if self.dense:
            m = np.zeros(self.N, dtype=bool)
            m[idx] = True
            return m
        return idx

    def _and(self, a, b):
        if self.dense:
            return a & b
        return np.intersect1d(a, b, assume_unique=True)

    def _positions(self, ts):
        return np.flatnonzero(ts) if self.dense else ts

    def _check(self, ts, pattern):
        w = float(self.twu[ts].sum()) / self.D
        if abs(w - self.xi) < EX.BAND:
            if self._wf is None:
                self._wf = EX.exact_weights(self.w)
                self._dex = EX.d_exact(self.tx, self._wf)
            pos = self._positions(ts)
            ok = EX.n_exact(self.tx, self._wf, pos) / self._dex >= EX.frac(self.xi)
            w2 = math.fsum(self.twu[pos].tolist()) / self.D
            self.borderline.append((ptuple(pattern), w, w2, ok))
            return ok, w2
        return w >= self.xi, w

    def _u_def3(self, ts, pattern):
        tot = 0.0
        for i in sorted(pattern):
            idx_i, ut_i = self.item_idx[i], self.item_util[i]
            if self.dense:
                tot += float(ut_i[ts[idx_i]].sum())
            else:
                tot += float(ut_i[np.searchsorted(idx_i, ts)].sum())
        return tot

    def mine(self):
        level1 = []
        for item, idx in self.item_idx.items():
            ts = self._mk(idx)
            ok, w = self._check(ts, [item])
            if ok:
                level1.append((item, ts, w))
        level1.sort(key=lambda x: (len(self.item_idx[x[0]]), x[0]))
        self.result = {}
        for item, ts, w in level1:
            self._emit((item,), ts, w)
        self._dfs([], level1)
        return self.result

    def _emit(self, pat, ts, w):
        key = ptuple(pat)
        self.result[key] = w
        if self.collect_u:
            self.u[key] = self._u_def3(ts, key)

    def _dfs(self, prefix, cls):
        depth = len(prefix) + 1
        if depth + 1 > self.max_len:
            return
        for i in range(len(cls)):
            it_i, ts_i, _ = cls[i]
            new_prefix = prefix + [it_i]
            nxt = []
            for j in range(i + 1, len(cls)):
                it_j, ts_j, _ = cls[j]
                ts = self._and(ts_i, ts_j)
                if self.dense:
                    if not ts.any():
                        continue
                elif ts.size == 0:
                    continue
                pat = new_prefix + [it_j]
                ok, w = self._check(ts, pat)
                if ok:
                    self._emit(pat, ts, w)
                    nxt.append((it_j, ts, w))
            if nxt:
                self._dfs(new_prefix, nxt)


def mine(tx, w, xi, max_len=None, collect_u=False):
    """-> ({pattern tuple: wus}, Miner). Patterns are ascending int tuples."""
    m = Miner(tx, w, xi, max_len=max_len, collect_u=collect_u)
    res = m.mine()
    return res, m


def fwup_hash(res):
    return hashlib.sha256(
        "\n".join(",".join(map(str, p)) for p in sorted(res)).encode()
    ).hexdigest()
