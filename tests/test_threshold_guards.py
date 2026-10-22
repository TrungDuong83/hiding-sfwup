"""Exact decisions at the threshold (|wus - xi| < 1e-9).

tests/data/threshold_guards_cases.json holds minimal DBs with the expected output (Q', trace) of
an independent exact-arithmetic reference implementation, recorded when the cases were built,
and two flags telling whether a float-only implementation (no exact branch) gave the same Q' /
trace. Cases (kind):
  alpha          float alpha = 0.0, exact alpha > 0 -> Prop. 4, not Sub-Algorithm 4
  skip           X skipped at its turn by a float test (wus = xi + 1e-17), MSU-MIU and SMSE
  qstar          floor(alpha |S(T)| / eu) one below (seed 708) or one above (seed 18, xi a few
                 ulps above the xi that makes alpha |S(T)| / eu an integer) the exact value
  smse_loop      SMSE: float test in the step loop vs exact test at the turn disagreed and the
                 loop never ended; must finish (30 s alarm)
  qstar_skipped  eu(10) = 1e-06 (huge x = alpha |S(T)| / eu): x inside the band but
                 q_lo = floor(x - delta) + 1 >= iu(I, T), so min(iu, q*) = iu for every q* in the
                 band; the exact branch is skipped
"""

import heapq
import json
import os
import random
import signal

import numpy as np
import pytest

from hiding_sfwup import baselines as B
from hiding_sfwup import exact as EX
from hiding_sfwup import hiding as H
from hiding_sfwup import miner as M

CASES = json.load(
    open(os.path.join(os.path.dirname(__file__), "data", "threshold_guards_cases.json"))
)


def _db(c):
    tx = [{int(i): q for i, q in t.items()} for t in c["tx"]]
    w = {int(i): v for i, v in c["w"].items()}
    return tx, w, [tuple(X) for X in c["S"]], {tuple(p): v for p, v in c["O"]}


def _alarm(*_):
    raise TimeoutError("did not finish in 30 s")


@pytest.mark.parametrize("c", CASES, ids=[c["name"] for c in CASES])
def test_matches_exact_reference(c):
    skip = c["kind"] == "qstar_skipped"
    if not skip:  # a float-only implementation differs from the reference on these cases
        assert not (c["float_only_qprime_eq_ref"] and c["float_only_trace_eq_ref"])
    tx, w, S, O = _db(c)
    m, xi = c["method"], c["xi"]
    tr = []
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(30)
    try:
        if m == "SMSE":
            q, st = B.smse(tx, w, O, S, xi=xi, deadline_s=None, trace=tr)
            tr = [
                [list(t["X"]), int(t["T"]), int(t["I"]), int(t["q"]), bool(t["delete"])] for t in tr
            ]
        elif m == "MSU-MIU":
            q, st = B.msumiu(tx, w, S, xi=xi, deadline_s=None, trace=tr)
            tr = [[list(t["X"]), int(t["T"]), int(t["I"]), t["q"], bool(t["delete"])] for t in tr]
        else:
            h = H.Hider(tx, w, xi, S, O, m, H.HideConfig(trace=True))
            q, st = h.run(), h.stats
            tr = [
                [
                    list(t["X"]),
                    t["victim"],
                    sorted(map(list, t["changes"])),
                    t.get("escalation") or "regular",
                ]
                for t in h.trace
            ]
    finally:
        signal.alarm(0)
    assert [{str(i): v for i, v in t.items()} for t in q] == c["ref_qprime"]
    assert tr == c["ref_trace"]
    if skip:
        assert (
            st["n_guard_qstar_skipped"] >= 1
            and st["n_guard_qstar"] == 0
            and st["n_guard_flip"] == 0
        )
        return
    assert st["n_guard_flip"] >= 1  # the exact branch overrode the float decision
    if c["kind"] == "alpha":
        assert st["n_guard_alpha"] >= 1
    if c["kind"] == "qstar":
        assert st["n_guard_qstar"] >= 1
    if c["kind"] in ("skip", "smse_loop"):
        assert st["n_guard_skip"] >= 1


def _rand_db(seed):
    rng = random.Random(seed)
    ids = sorted(rng.sample(range(1, 40), rng.randint(4, 9)))
    w = {i: float(rng.randint(1, 10)) for i in ids}
    tx = [
        {i: rng.randint(1, 10) for i in sorted(rng.sample(ids, rng.randint(1, len(ids))))}
        for _ in range(rng.randint(5, 30))
    ]
    return tx, w


@pytest.mark.parametrize("seed", range(40))
def test_below_turn_equals_float_outside_band(seed, monkeypatch):
    """Outside |N_X/D - xi| < 1e-9 below_turn() must return exactly below()."""
    tx, w = _rand_db(seed)
    xi = random.Random(seed + 1).choice([0.2, 0.3, 0.35, 0.4, 0.5])
    fw, _ = M.mine(tx, w, xi)
    if not fw:
        pytest.skip("no FWUP")
    S = sorted(fw)[: 1 + seed % 4]
    seen = []
    orig = B.Tables.below_turn

    def spy(self, p):
        r = orig(self, p)
        if not self._in_band(p, self.xi):
            seen.append((r, self.below(p)))
        return r

    monkeypatch.setattr(B.Tables, "below_turn", spy)
    B.smse(tx, w, fw, S, xi=xi, deadline_s=None)
    B.msumiu(tx, w, S, xi=xi, deadline_s=None)
    assert seen and all(a == b for a, b in seen)


def test_band_constant_unchanged():
    assert EX.BAND == 1e-9


# ---------------------------------------------------------------- integer A_k bookkeeping
def _direct(tx, w, tids):
    wf = EX.exact_weights(w)
    return EX.n_exact(tx, wf, tids), EX.d_exact(tx, wf)


@pytest.mark.parametrize("seed", range(40))
def test_exact_tu_equals_direct_fractions(seed):
    """ExactTU (integer A_k, lcm of eu denominators) = Fraction sums computed directly."""
    tx, w = _rand_db(seed)
    if seed % 2:  # decimal eu as in the paper example (L > 1)
        w = {i: v / 10 for i, v in w.items()}
    ex = EX.ExactTU(tx, w)
    ex.check()
    items = sorted(w)
    for X in [(i,) for i in items] + list(zip(items, items[1:])):
        tids = [k for k, t in enumerate(tx) if all(i in t for i in X)]
        assert (ex.n(tids), ex.d()) == _direct(tx, w, tids)


@pytest.mark.parametrize("seed", range(20))
def test_exact_tu_incremental_during_runs(seed, monkeypatch):
    """After every change of 20 random runs (proposal + both baselines): A_k updated by integer
    add/subtract = A_k recomputed (EX.CHECK), and D, N_X = direct Fractions."""
    monkeypatch.setattr(EX, "CHECK", True)
    tx, w = _rand_db(1000 + seed)
    xi = random.Random(seed).choice([0.2, 0.3, 0.35, 0.4])
    fw, _ = M.mine(tx, w, xi)
    if not fw:
        pytest.skip("no FWUP")
    S = sorted(fw)[: 1 + seed % 3]
    n_checked = [0]
    orig = EX.ExactTU.update

    def update(self, k, t):
        orig(self, k, t)  # runs check() (EX.CHECK)
        X = S[0]
        tids = [j for j, u in enumerate(self.tx) if all(i in u for i in X)]
        assert (self.n(tids), self.d()) == _direct(self.tx, w, tids)
        n_checked[0] += 1

    monkeypatch.setattr(EX.ExactTU, "update", update)
    # build the exact bookkeeping from the start so that every change is checked
    monkeypatch.setattr(H.Hider, "_setup", lambda self, f=H.Hider._setup: (f(self), self._exact()))
    tb_init = B.Tables.__init__

    def init(self, *a, **k):
        tb_init(self, *a, **k)
        self._exact()

    monkeypatch.setattr(B.Tables, "__init__", init)
    H.Hider(tx, w, xi, S, fw, H.STRATEGIES[seed % 4]).run()
    B.smse(tx, w, fw, S, xi=xi, deadline_s=None)
    B.msumiu(tx, w, S, xi=xi, deadline_s=None)
    assert n_checked[0] > 0


# ------------------------------------------------------- candidate filter of pattern removal
def _old_check_removed(tb, ids):
    """Straightforward version of _check_removed (band test on every touched id), as reference."""
    ids = ids[~tb.removed[ids]]
    if tb.xi_ns is None:
        m = tb.N[ids] < tb.thr()
    else:
        m = tb.N[ids] < np.where(tb.sens[ids], tb.thr(), tb.thr_ns())
    xis = tb.xi if tb.xi_ns is None else np.where(tb.sens[ids], tb.xi, tb.xi_ns)
    for j in np.flatnonzero(np.abs(tb.N[ids] / tb.D - xis) < EX.BAND).tolist():
        p = int(ids[j])
        m[j] = tb._below_exact(p, tb.xi if tb.xi_ns is None or tb.sens[p] else tb.xi_ns, bool(m[j]))
    return set(ids[m].tolist())


def _old_pop_fallen(tb, h, th, xi):
    out = set()
    while h and (h[0][0] < th or abs(h[0][0] / tb.D - xi) < EX.BAND):
        val, p = heapq.heappop(h)
        if tb.removed[p] or tb.dirty[p] or val != tb.N[p]:
            continue
        if tb._in_band(p, xi) and not tb._below_exact(p, xi, val < th):
            continue
        out.add(p)
    return out


def _edge_values(rng, xi, D, n):
    """N near (xi - BAND) D, xi D, (xi + BAND) D, +- a few ulps, plus some random values."""
    out = []
    for _ in range(n):
        base = rng.choice(
            [(xi - EX.BAND) * D, xi * D, (xi + EX.BAND) * D, rng.uniform(0, 2 * xi * D)]
        )
        x = base
        for _ in range(rng.randint(0, 6)):
            x = np.nextafter(x, np.inf if rng.random() < 0.5 else -np.inf)
        out.append(float(x))
    return np.array(out)


@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize("two_thresholds", [False, True])
def test_cand_filter_equals_reference(seed, two_thresholds, monkeypatch):
    """_check_removed / _pop_fallen (one vector test N < (xi + BAND) D, then exact only in the
    band) = the straightforward reference on N, D within a few ulps of both band edges: same
    decisions. 'Exact' = exact value of the floats N, D, xi (oracle shared by both versions)."""
    import heapq as _hq
    from fractions import Fraction

    rng = random.Random(seed)
    tx, w = _rand_db(seed)
    pats = sorted(
        {
            tuple(sorted(rng.sample(sorted({i for t in tx for i in t}), rng.randint(1, 2))))
            for _ in range(60)
        }
    )
    xi = rng.choice([0.2, 0.3, 0.37, 0.41])
    tb = B.Tables(
        tx,
        w,
        pats,
        [k % 3 == 0 for k in range(len(pats))],
        xi=xi,
        xi_ns=(xi * 1.25 if two_thresholds else None),
    )
    monkeypatch.setattr(
        B.Tables,
        "_below_exact",
        lambda self, p, x, b: Fraction(float(self.N[p])) < Fraction(x) * Fraction(self.D),
    )
    removed = []
    monkeypatch.setattr(B.Tables, "remove_pattern", lambda self, p: removed.append(p))
    for _ in range(20):
        tb.D = rng.uniform(10.0, 1e6)
        tb.removed[:] = False
        tb.dirty[:] = False
        xis = np.where(tb.sens, tb.xi, tb.xi_ns) if two_thresholds else np.full(len(pats), xi)
        tb.N[:] = [float(v) for v in (_edge_values(rng, x, tb.D, 1)[0] for x in xis)]
        ids = np.arange(len(pats))
        removed.clear()
        tb._check_removed(ids)
        assert set(removed) == _old_check_removed(tb, ids)
        for x, sens in ((tb.xi, True), (tb.xi_ns, False)) if two_thresholds else ((xi, None),):
            sel = [p for p in range(len(pats)) if sens is None or bool(tb.sens[p]) == sens]
            h = [(float(tb.N[p]), p) for p in sel]
            _hq.heapify(h)
            h2 = list(h)
            removed.clear()
            tb._pop_fallen(h, x * tb.D, x)
            assert set(removed) == _old_pop_fallen(tb, h2, x * tb.D, x)
