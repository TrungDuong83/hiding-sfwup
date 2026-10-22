"""The paper's running example (Tables 2-7, xi = 0.4, S = {AD, CD, EAD}): FWUP and wus,
Phase 1 (SC, NSC, Wt), RISWU, processing order, Prop. 4 allocation, final Q' and metrics."""

import math

import pytest

from hiding_sfwup import data as DA
from hiding_sfwup import hiding as H
from hiding_sfwup import metrics as ME
from hiding_sfwup import miner as M
from hiding_sfwup import phase1 as P1
from hiding_sfwup import workloads as W

import running_example as RE
from examples import S_PAPER, XI, A, B, C, D, E, paper

S = S_PAPER


@pytest.fixture(scope="module")
def r():
    return RE.compute()


@pytest.fixture(scope="module")
def ex():
    tx, w, fw, _ = paper()
    return tx, w, fw


def _prepared(tx, w, fw, strategy="SDIF"):
    h = H.Hider(tx, w, XI, S, fw, strategy)
    h._setup()
    h.m = P1.compute(h.tx, h.S, h.O)
    h.wt = h.m["wt"]
    return h


def test_fwups(ex):
    _, _, fw = ex
    want = {
        "D": 1.000,
        "A": 0.807,
        "AD": 0.807,
        "E": 0.778,
        "DE": 0.778,
        "C": 0.605,
        "AC": 0.605,
        "CD": 0.605,
        "ACD": 0.605,
        "AE": 0.586,
        "ADE": 0.586,
    }
    assert {DA.letters(p): round(v, 3) for p, v in fw.items()} == want


def test_twu_and_tidsets(r):
    assert [round(v, 3) for v in r["twu"]] == [0.725, 0.667, 0.600, 0.700, 0.767]
    assert round(r["sumtwu"], 6) == 3.458333
    got = {name: round(v, 4) for name, v, _ in r["fwup"]}
    assert got == {
        "D": 1.0,
        "A": 0.8072,
        "AD": 0.8072,
        "E": 0.7783,
        "DE": 0.7783,
        "C": 0.6048,
        "AC": 0.6048,
        "CD": 0.6048,
        "ACD": 0.6048,
        "AE": 0.5855,
        "ADE": 0.5855,
    }
    assert {n: ts for n, _, ts in r["fwup"]}["ADE"] == ["T1", "T3", "T4"]


def test_phase1_items(ex, r):
    """SC(i), NSC(i) of every item (Table 6)."""
    tx, _, fw = ex
    m = P1.compute(tx, S, fw)
    assert m["sc_item"] == {A: 2, C: 1, D: 3, E: 1}
    assert m["nsc_item"] == {A: 4, C: 3, D: 3, E: 3}
    assert r["sc_item"] == {"A": 2, "B": 0, "C": 1, "D": 3, "E": 1}
    assert r["nsc_item"] == {"A": 4, "B": 0, "C": 3, "D": 3, "E": 3}


def test_phase1_transactions(ex, r):
    """SC(T), NSC(T), Wt(T) = SC(T)/(NSC(T)+1) of every transaction (Table 7)."""
    tx, _, fw = ex
    m = P1.compute(tx, S, fw)
    assert list(m["sc_trans"]) == [3, 0, 3, 2, 2]
    assert list(m["nsc_trans"]) == [8, 3, 8, 5, 5]
    assert [float(x) for x in m["wt"]] == [3 / 9, 0.0, 3 / 9, 2 / 6, 2 / 6]
    assert r["sc_T"] == [3, 0, 3, 2, 2] and r["nsc_T"] == [8, 3, 8, 5, 5]
    assert [round(v, 4) for v in r["wt_T"]] == [0.3333, 0.0, 0.3333, 0.3333, 0.3333]


def test_riswu(r, ex):
    assert round(r["riswu"]["CD"]["D"], 4) == 0.8665 and round(r["riswu"]["AD"]["D"], 4) == 0.9
    assert round(r["riswu"]["ADE"]["A"], 4) == 0.9185 and round(r["riswu"]["ADE"]["E"], 4) == 0.4074
    h = _prepared(*ex)
    assert round(h.riswu(D, (C, D), h.tids_of((C, D))), 4) == 0.8665


def test_sdif_run(ex):
    tx, w, fw = ex
    h = H.Hider(tx, w, XI, S, fw, "SDIF", H.HideConfig(trace=True))
    q = h.run()
    # processing order AD, CD, EAD by RISWU(victim) desc; victim D for all three
    assert [DA.letters(X) for X in h.order] == ["AD", "CD", "ADE"]
    assert [h.order_keys[X][1] for X in h.order] == [D, D, D]
    assert [round(h.order_keys[X][0], 4) for X in h.order] == [0.9000, 0.8665, 0.6728]
    r1 = h.trace[0]  # round 1 (X = AD, victim D), Prop. 4
    assert DA.letters(r1["X"]) == "AD" and r1["victim"] == D and r1["iter"] == 1
    assert round(r1["alpha"], 6) == 2.347222
    assert [(k + 1, dd) for k, _, dd in r1["changes"]] == [(1, 1), (3, 1), (4, 2), (5, 3)]
    assert ME.hf_exact(q, w, XI, S)[0] == 0.0
    assert h.stats["n_rehide"] == 0


def test_prop4_cd_first(ex):
    """CD hidden first, victim D, on the original DB."""
    tx, w, fw = ex
    h = _prepared(tx, w, fw)
    X = (C, D)
    TX = h.tids_of(X)
    assert h.select_victim(X, TX) == D
    n, d = h.nd(TX)
    alpha = (n - XI * d) / (1 - XI)
    assert round(alpha, 6) == 1.180556
    ch = h.prop4_changes(D, alpha, TX, h.contributions(D, TX))
    assert [(k + 1, dd) for k, _, dd in ch] == [(1, 1), (3, 1), (5, 3)]
    for k, i, dd in ch:
        h.apply(k, i, dd)
    assert round(h.wus((A, D)), 6) == 0.219321
    assert round(h.wus((A, D, E)), 6) == 0.219321


def test_strategies(r):
    st = r["strategies"]
    assert st["SDIF"]["order"] == ["AD", "CD", "ADE"] == st["SC-only"]["order"]
    assert st["Max-RISWU"]["order"] == ["ADE", "AD", "CD"]
    assert st["Min-RISWU"]["order"] == ["AD", "CD", "ADE"]
    r1 = st["SDIF"]["rounds"][0]
    assert round(r1["alpha"], 6) == 2.347222
    assert [x["diu"] for x in r1["prop4"]] == [1, 1, 2, 3]
    assert r1["increased"] == ["AE", "E"]
    assert [len(s["rounds"]) for s in st.values()] == [1, 2, 2, 1]
    assert all(s["hf"] == 0.0 and s["n_rehide"] == 0 for s in st.values())
    assert [s["n_pre_hidden"] for s in st.values()] == [2, 1, 1, 2]
    assert st["Max-RISWU"]["qprime"] == [
        "{C:5, E:2}",
        "{B:2, D:1, E:1}",
        "{B:1, C:2, E:1}",
        "{D:2, E:1}",
        "{A:1, C:2}",
    ]
    assert st["Min-RISWU"]["rounds"][0]["increased"] == ["C", "CD"]


# Def. 3: u(Y) = sum_{T ⊇ Y} sum_{i in Y} iu(i, T) eu(i); eu A..E = .4 .6 .2 .5 .3
U_Q = {
    "D": 4.0,
    "A": 3.2,
    "E": 1.5,
    "C": 1.8,
    "AD": 6.7,
    "DE": 4.0,
    "AC": 4.2,
    "CD": 4.3,
    "ACD": 6.7,
    "AE": 4.0,
    "ADE": 6.0,
}  # sum 46.4
# Q' of the SDIF run = D deleted from T1, T3, T4, T5
FW_QP = {"A", "B", "C", "E", "AC", "AE", "BE", "CE", "ACE"}
U_QP = {
    "A": 3.2,
    "B": 1.8,
    "C": 1.8,
    "E": 1.5,
    "AC": 4.2,
    "AE": 4.0,
    "BE": 2.4,
    "CE": 2.3,
    "ACE": 4.3,
}  # sum 25.5


def test_metrics(ex):
    tx, w, fw = ex
    assert math.isclose(sum(U_Q.values()), 46.4)
    assert math.isclose(W.sum_u_def3(tx, w, list(fw)), 46.4)
    h = H.Hider(tx, w, XI, S, fw, "SDIF")
    q = h.run()
    assert q == [
        {A: 2, C: 5, E: 2},
        {B: 2, D: 1, E: 1},
        {A: 3, B: 1, C: 2, E: 1},
        {A: 2, E: 1},
        {A: 1, C: 2},
    ]
    ev = ME.evaluate(tx, q, w, XI, S, fw, 46.4)
    fq, mq = M.mine(q, w, XI, collect_u=True)
    assert {DA.letters(p) for p in fq} == FW_QP
    assert {DA.letters(p): round(v, 6) for p, v in mq.u.items()} == U_QP
    assert math.isclose(ev["ius"], 25.5 / 46.4)
    assert math.isclose(ev["mc"], 3 / 8)  # NS lost: D, DE, ACD of 8
    assert math.isclose(ev["ac"], 4 / 11)  # new: B, BE, CE, ACE
    assert ev["hf"] == 0.0 and ev["hf_mined"] == 0.0


def test_hf_exact_vs_direct(ex):
    """HF from metrics.hf_exact == direct count of wus over the DB (plain loops)."""
    tx, w, _ = ex
    q = [dict(t) for t in tx]  # unsanitized: every X still >= xi
    hf, rows = ME.hf_exact(q, w, XI, S)
    twu = [sum(w[i] * v for i, v in t.items()) / len(t) for t in q]
    direct = [sum(tw for t, tw in zip(q, twu) if set(X) <= set(t)) / sum(twu) >= XI for X in S]
    assert [r[2] for r in rows] == direct and hf == sum(direct) / len(S) == 1.0
    q2 = [dict(t) for t in tx]
    del q2[0][D]  # AD loses T1 only
    _, rows2 = ME.hf_exact(q2, w, XI, S)
    twu2 = [sum(w[i] * v for i, v in t.items()) / len(t) for t in q2]
    direct2 = [sum(tw for t, tw in zip(q2, twu2) if set(X) <= set(t)) / sum(twu2) >= XI for X in S]
    assert [r[2] for r in rows2] == direct2
