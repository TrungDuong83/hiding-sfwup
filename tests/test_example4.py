"""Example 4 of the paper (Section 4.6): the database of Table 2 at xi = 0.6 with
S = {AC, CD, DE} and SDIF. Hiding DE raises wus(AC) back above xi; the final verification pass
(Phase 3) hides it again. Locks every number stated in the example."""

import pytest

from hiding_sfwup import data as DA
from hiding_sfwup import hiding as H
from hiding_sfwup import miner as M

from examples import A, C, D, E

XI6 = 0.6
S6 = [(A, C), (C, D), (D, E)]


def wus(tx, w, X):
    tw = [DA.twu_of(t, w) for t in tx]
    return sum(v for t, v in zip(tx, tw) if all(i in t for i in X)) / sum(tw)


@pytest.fixture(scope="module")
def run():
    tx, w = DA.paper_example()
    fw, _ = M.mine(tx, w, XI6)
    h = H.Hider(tx, w, XI6, S6, fw, "SDIF", H.HideConfig(trace=True))
    q = h.run()
    return tx, w, fw, h, q


def test_fwup_and_initial_wus(run):
    tx, w, fw, _, _ = run
    assert sorted(DA.letters(p) for p in fw) == sorted(
        ["D", "A", "AD", "E", "DE", "C", "AC", "CD", "ACD"]
    )
    assert [round(wus(tx, w, X), 3) for X in S6] == [0.605, 0.605, 0.778]


def test_covers_victims_and_order(run):
    _, _, _, h, _ = run
    sc, nsc = h.m["sc_item"], h.m["nsc_item"]
    assert (sc[C], sc[D], nsc[C], nsc[D]) == (2, 2, 2, 3)
    assert [h.order_keys[X][1] for X in S6] == [C, C, D]
    assert h.order == [(A, C), (C, D), (D, E)]
    assert h.order_keys[(A, C)][0] == h.order_keys[(C, D)][0]  # RISWU tie, broken by item IDs


def test_steps(run):
    tx, w, _, h, _ = run
    cur = [dict(t) for t in tx]
    seen = []
    for tr in h.trace:
        for k, i, d in tr["changes"]:
            cur[k][i] -= d
            if cur[k][i] <= 0:
                del cur[k][i]
        seen.append((tr["X"], tr["victim"], sorted({k for k, _, _ in tr["changes"]}), cur))
        cur = [dict(t) for t in cur]
    assert len(seen) == 3
    # 1) AC, victim C: C lowered by one unit in T1, T3, T5 -> wus(AC) = 0.586; CD below xi too
    X, v, ks, q1 = seen[0]
    assert (tuple(X), v, ks) == ((A, C), C, [0, 2, 4])
    assert round(wus(q1, w, (A, C)), 3) == 0.586 and wus(q1, w, (C, D)) < XI6
    # 2) DE, victim D: D removed from T1-T4; twu(T4) 0.700 -> 0.550; AC back above xi (0.607)
    X, v, ks, q2 = seen[1]
    assert (tuple(X), v, ks) == ((D, E), D, [0, 1, 2, 3])
    assert all(D not in q2[k] for k in range(4))
    assert round(DA.twu_of(q1[3], w), 3) == 0.700 and round(DA.twu_of(q2[3], w), 3) == 0.550
    assert round(wus(q2, w, (A, C)), 3) == 0.607
    # 3) final pass, R = {AC}: C 4 -> 3 in T1, removed from T3 and T5 -> wus(AC) = 0.184
    X, v, ks, q3 = seen[2]
    assert (tuple(X), v, ks) == ((A, C), C, [0, 2, 4])
    assert (tx[0][C], q3[0][C]) == (5, 3) and C not in q3[2] and C not in q3[4]
    assert round(wus(q3, w, (A, C)), 3) == 0.184


def test_log_and_final_state(run):
    _, w, _, h, q = run
    log = [(tuple(r["X"]), r.get("pre_hidden"), r["rehide"]) for r in h.sfwup_log]
    assert log == [((A, C), None, False), ((C, D), True, False), ((D, E), None, False),
                   ((A, C), None, True)]
    assert h.stats["n_rehide"] == 1
    assert all(wus(q, w, X) < XI6 for X in S6)
