"""Exact threshold decisions.

wus(X) = N_X / D with N_X = sum_{T ⊇ X} twu(T), D = sum_T twu(T), twu(T) = TU(T) / |S(T)|.
With integer iu and decimal eu every twu is rational, so wus >= xi can be decided exactly.
Policy: compare in floating point; only when |wus - xi| < BAND re-decide exactly. eu and xi are
read as the decimal the user wrote, Fraction(repr(x)) (0.4 -> 2/5, 7.0 -> 7).
"""

import math
from fractions import Fraction

BAND = 1e-9


def frac(x):
    return Fraction(repr(float(x)))


def tu_exact(t, wf):
    return sum((wf[i] * q for i, q in t.items()), Fraction(0))


def twu_exact(t, wf):
    if not t:
        return Fraction(0)
    return tu_exact(t, wf) / len(t)


def exact_weights(w):
    return {i: frac(v) for i, v in w.items()}


def d_exact(tx, wf):
    """D as a Fraction, grouped by |S(T)| so that there is one division per size."""
    by_s = {}
    for t in tx:
        if t:
            by_s[len(t)] = by_s.get(len(t), Fraction(0)) + tu_exact(t, wf)
    return sum((v / s for s, v in by_s.items()), Fraction(0))


def n_exact(tx, wf, tids):
    by_s = {}
    for k in tids:
        t = tx[int(k)]
        if t:
            by_s[len(t)] = by_s.get(len(t), Fraction(0)) + tu_exact(t, wf)
    return sum((v / s for s, v in by_s.items()), Fraction(0))


def wus_ge(n_float, d_float, xi, exact_fn):
    """Decide wus >= xi from the float numerator/denominator; exact_fn() -> (N, D) as Fractions
    is called only inside the band."""
    if d_float <= 0:
        return False
    w = n_float / d_float
    if abs(w - xi) >= BAND:
        return w >= xi
    n, d = exact_fn()
    if d == 0:
        return False
    return n / d >= frac(xi)


def fsum(xs):
    return math.fsum(xs)


CHECK = False  # tests only: re-derive A_k from scratch after every update (ExactTU.check)


class ExactTU:
    """Exact N_X and D for the hiding code (hiding.py, baselines.py) in integer arithmetic.

    eu are taken as Fraction(repr(eu)); L = lcm of their denominators, so W_i = eu_i * L and
    TU_L(T) = sum_i W_i * iu(i, T) are integers and TU(T) = TU_L(T) / L. Kept per transaction:
    TU_L(T) and |S(T)|; for the whole DB A_k = sum{TU_L(T) : |S(T)| = k}.
      D   = sum_k A_k / (k L)
      N_X = sum_k (sum{TU_L(T) : T in T_X, |S(T)| = k}) / (k L)
    The owner calls update(k, t) after every change of transaction k. Built lazily at the first
    exact decision, so the floating-point path is untouched."""

    def __init__(self, tx, w):
        wf = exact_weights(w)
        L = 1
        for v in wf.values():
            L = L * v.denominator // math.gcd(L, v.denominator)
        self.L = L
        self.W = {i: int(v * L) for i, v in wf.items()}
        self.tx = tx
        self.tu = [0] * len(tx)
        self.sz = [0] * len(tx)
        self.A = {}
        for k, t in enumerate(tx):
            self._set(k, t)

    def _tu(self, t):
        W = self.W
        return sum(W.get(i, 0) * q for i, q in t.items())  # as twu_of: a missing eu counts 0

    def _set(self, k, t):
        s = len(t)
        u = self._tu(t) if s else 0
        self.tu[k], self.sz[k] = u, s
        if s and u:
            self.A[s] = self.A.get(s, 0) + u

    def update(self, k, t):
        s0 = self.sz[k]
        if s0 and self.tu[k]:
            a = self.A[s0] - self.tu[k]
            if a:
                self.A[s0] = a
            else:
                del self.A[s0]
        self._set(k, t)
        if CHECK:
            self.check()

    def check(self):
        A = {}
        for k, t in enumerate(self.tx):
            if t:
                A[len(t)] = A.get(len(t), 0) + self._tu(t)
                assert self.tu[k] == self._tu(t) and self.sz[k] == len(t), k
        A = {s: a for s, a in A.items() if a}
        assert A == self.A, "A_k incremental != recomputed"

    def d(self):
        return sum((Fraction(a, s * self.L) for s, a in self.A.items() if a), Fraction(0))

    def n(self, tids):
        by_s = {}
        tu, sz = self.tu, self.sz
        for k in tids:
            k = int(k)
            if sz[k]:
                by_s[sz[k]] = by_s.get(sz[k], 0) + tu[k]
        return sum((Fraction(a, s * self.L) for s, a in by_s.items() if a), Fraction(0))

    def twu(self, k):
        return Fraction(self.tu[k], self.sz[k] * self.L) if self.sz[k] else Fraction(0)

    def eu(self, i):
        return Fraction(self.W[i], self.L)
