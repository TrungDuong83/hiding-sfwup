"""Baselines.
(a) utility mode reproduces the worked example of the SMSE paper (Sec. 8, pp. 178-179,
    Tables 2-7) and a hand-computed MSU-MIU run on the example of the MSU-MIU paper
    (Tables 1-4, pp. 271-275; that paper contains no worked sanitization run);
(b) SMSE with the heap == SMSE with the linear argmin (same Q', same steps) on chess and retail;
(c) wus mode: every SFWUP a baseline completes has exact wus < xi at that point.
"""

import pytest

from hiding_sfwup import baselines as B
from hiding_sfwup import data as DA
from hiding_sfwup import metrics as ME
from hiding_sfwup import workloads as WL

L = "abcdef"
a, b, c, d, e, f = 1, 2, 3, 4, 5, 6
# SMSE paper Table 2 (= MSU-MIU paper Tables 1-2): 6 transactions, eu a..f = 5 3 2 1 6 10
DB = [
    {a: 2, b: 1, e: 3},
    {c: 1, d: 6},
    {b: 1, c: 2, e: 1, f: 1},
    {a: 3, b: 4, c: 2, d: 2, e: 5},
    {b: 3, c: 5},
    {a: 2, e: 7, f: 3},
]
W = {a: 5.0, b: 3.0, c: 2.0, d: 1.0, e: 6.0, f: 10.0}
HUIS = {
    (e,): 96,
    (a, e): 125,
    (b, e): 72,
    (e, f): 88,
    (a, b, e): 88,
    (a, e, f): 82,
    (a, b, c, e): 61,
    (a, b, c, d, e): 63,
}  # SMSE Table 3 / MSU-MIU Table 3


def u_of(tx, Y):
    return sum(sum(W[i] * t[i] for i in Y) for t in tx if set(Y) <= set(t))


def test_smse_paper_tables():
    assert {Y: u_of(DB, Y) for Y in HUIS} == HUIS
    tb = B.Tables(
        DB,
        W,
        sorted(HUIS),
        [Y in {(b, e), (e, f)} for Y in sorted(HUIS)],
        measure="utility",
        minutil=60,
        nshc=True,
    )
    # SMSE paper Table 4: T-table holds T1, T3, T4, T6; NSHC(T1, T3, T4, T6) = 3, 1, 5, 3
    assert (tb.sensT + 1).tolist() == [1, 3, 4, 6]
    assert tb.NSHC.tolist() == [3, 1, 5, 3]


def test_smse_paper_example():
    tr = []
    q, st = B.smse(DB, W, list(HUIS), [(b, e), (e, f)], minutil=60, measure="utility", trace=tr)
    steps = [(t["X"], t["T"] + 1, L[t["I"] - 1], t["q"], t["delete"], t["N_X"]) for t in tr]
    assert steps == [
        (
            (b, e),
            3,
            "e",
            1,
            True,
            63.0,
        ),  # T3 (NSHC 1), CVI = {e}, delete e: u(be) 72 -> 63 (Table 7)
        (
            (b, e),
            1,
            "b",
            1,
            True,
            42.0,
        ),  # T1, CVI = {b, e}, NSHC(b,T1)=1 < NSHC(e,T1)=3 -> delete b
        (
            (e, f),
            6,
            "f",
            2,
            False,
            52.0,
        ),  # {e,f}: 88 -> 72 after T3; T6: decrease f by ceil(13/10) = 2
    ]
    assert q == [
        {a: 2, e: 3},
        {c: 1, d: 6},
        {b: 1, c: 2, f: 1},
        {a: 3, b: 4, c: 2, d: 2, e: 5},
        {b: 3, c: 5},
        {a: 2, e: 7, f: 1},
    ]
    # "the SMSE algorithm generates a missing non-sensitive itemset {a, b, e}" (p. 179)
    missing = sorted(Y for Y in HUIS if Y not in {(b, e), (e, f)} and u_of(q, Y) < 60)
    assert missing == [(a, b, e)]
    assert all(u_of(q, Y) < 60 for Y in [(b, e), (e, f)])
    # Tables 6-7 after the first update: u(e) = 90, u(be) = 63, u(ef) = 72
    tb = B.Tables(
        DB,
        W,
        sorted(HUIS),
        [Y in {(b, e), (e, f)} for Y in sorted(HUIS)],
        measure="utility",
        minutil=60,
        nshc=True,
    )
    tb.apply(int(list(tb.sensT).index(2)), e, 1)
    pid = {Y: k for k, Y in enumerate(sorted(HUIS))}
    assert (tb.N[pid[(e,)]], tb.N[pid[(b, e)]], tb.N[pid[(e, f)]]) == (90.0, 63.0, 72.0)


def test_msumiu_paper_example():
    """MSU-MIU paper p. 275 example: SHUIs E, AEF, ABCE (Table 4; the table omits T4 for E,
    which contains E). Hand computation with delta = 60:
      f(A)=2 f(B)=1 f(C)=1 f(E)=3 f(F)=1 -> w(ABCE)=7 > w(AEF)=6 > w(E)=3
      ABCE: u=61, T4 only; min-utility item C (4); q* = ceil(2/2)=1 < iu 2 -> C: 2->1, u=59, stop
      AEF : u=82, T6 only; min item A (10); q* = ceil(23/5)=5 >= iu 2 -> delete A; u=0
      E   : u=96; projection by u(E,T) desc: T6 42, T4 30, T1 18, T3 6;
            T6: item E, q* = ceil(37/6)=7 >= iu 7 -> delete E; u(E)=54 < 60, stop"""
    S = [(e,), (a, e, f), (a, b, c, e)]
    tb = B.Tables(DB, W, sorted(S), [True] * 3, measure="utility", minutil=60, track_removal=False)
    pid = {Y: k for k, Y in enumerate(sorted(S))}
    assert (tb.sensT[tb.rows_of(pid[(e,)])] + 1).tolist() == [1, 3, 4, 6]
    assert (tb.sensT[tb.rows_of(pid[(a, e, f)])] + 1).tolist() == [6]
    assert (tb.sensT[tb.rows_of(pid[(a, b, c, e)])] + 1).tolist() == [4]
    tr = []
    q, st = B.msumiu(DB, W, S, minutil=60, measure="utility", trace=tr)
    assert st["order"] == [[a, b, c, e], [a, e, f], [e]]
    steps = [(tuple(t["X"]), t["T"] + 1, L[t["I"] - 1], t["q"], t["delete"]) for t in tr]
    assert steps == [
        ((a, b, c, e), 4, "c", 1, False),
        ((a, e, f), 6, "a", None, True),
        ((e,), 6, "e", None, True),
    ]
    assert q[3] == {a: 3, b: 4, c: 1, d: 2, e: 5} and q[5] == {f: 3}
    assert all(u_of(q, Y) < 60 for Y in S)


# --------------------------------------------------------------------- (b), (c)
def _cell(ds):
    rec = WL.read_workload(f"x1_{ds}")
    tx, w = DA.load_db(ds)
    return rec, tx, w, WL.fwup_of(rec), [tuple(x) for x in rec["S"]]


@pytest.mark.slow
@pytest.mark.parametrize("ds", ["chess", "retail"])
def test_smse_heap_equals_linear(ds):
    rec, tx, w, fw, S = _cell(ds)
    t1, t2 = [], []
    q1, s1 = B.smse(tx, w, fw, S, xi=rec["xi"], argmin="heap", trace=t1, deadline_s=None)
    q2, s2 = B.smse(tx, w, fw, S, xi=rec["xi"], argmin="linear", trace=t2, deadline_s=None)
    assert [(t["X"], t["T"], t["I"], t["q"]) for t in t1] == [
        (t["X"], t["T"], t["I"], t["q"]) for t in t2
    ]
    assert q1 == q2
    assert not s1["censored"] and not s2["censored"]


@pytest.mark.slow
@pytest.mark.parametrize("ds", ["chess", "retail"])
@pytest.mark.parametrize("method", ["SMSE", "MSU-MIU"])
def test_completed_sfwup_below_xi(ds, method):
    rec, tx, w, fw, S = _cell(ds)
    seen = []

    def check(X, cur):
        hf, rows = ME.hf_exact(cur, w, rec["xi"], [X])
        seen.append((X, rows[0][2]))

    if method == "SMSE":
        q, st = B.smse(tx, w, fw, S, xi=rec["xi"], on_done=check, deadline_s=None)
    else:
        q, st = B.msumiu(tx, w, S, xi=rec["xi"], on_done=check, deadline_s=None)
    assert seen and not any(ge for _, ge in seen)
    assert st["n_done"] == len(S)
