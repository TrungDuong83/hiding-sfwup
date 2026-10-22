"""RISWU variants, greedy mechanisms, the wus processing order, a lower hiding target, and the
workload files of the threshold, workload, utility-distribution and scalability experiments."""

import random

import numpy as np
import pytest

from hiding_sfwup import baselines as BL
from hiding_sfwup import data as DA
from hiding_sfwup import hiding as H
from hiding_sfwup import metrics as ME
from hiding_sfwup import miner as M
from hiding_sfwup import phase1 as P1
from hiding_sfwup import workloads as W

from examples import XI, C, D, paper


def _prepared(variant, strategy="Max-RISWU"):
    tx, w, fw, S = paper()
    h = H.Hider(tx, w, XI, S, fw, strategy, H.HideConfig(riswu_w=variant))
    h._setup()
    h.m = P1.compute(h.tx, h.S, h.O)
    h.wt = h.m["wt"]
    return h


def test_riswu_variants():
    """RISWU(D, CD) over T1, T3, T5 (u(D) = .5, .5, 1.5; twu = .725, .6, .766667; Wt = 1/3 each):
    W = Wt  -> (0.5 + 0.5 + 1.5)/3 = 0.833333
    W = twu -> (0.3625 + 0.3 + 1.15)/2.091667 = 0.866534 (= full, Wt constant here)"""
    vals, contribs = {}, {}
    for v in ("full", "no_twu", "no_wt"):
        h = _prepared(v)
        TX = h.tids_of((C, D))
        vals[v] = round(h.riswu(D, (C, D), TX), 6)
        contribs[v] = h.contributions(D, TX).tolist()
    assert vals == {"full": 0.866534, "no_twu": 0.833333, "no_wt": 0.866534}
    assert contribs["full"] == contribs["no_twu"] == contribs["no_wt"]  # only RISWU changes


@pytest.mark.parametrize("mech", ["greedy-batch", "greedy-unit"])
def test_greedy_example(mech):
    tx, w, fw, S = paper()
    h = H.Hider(tx, w, XI, S, fw, "SDIF", H.HideConfig(mechanism=mech, trace=True))
    q = h.run()
    assert ME.hf_exact(q, w, XI, S)[0] == 0.0
    assert all(dd == 1 for t in h.trace for _, _, dd in t["changes"])
    if mech == "greedy-unit":
        assert all(len(t["changes"]) == 1 for t in h.trace)


def test_pre_hidden_paper():
    tx, w, fw, S = paper()
    h = H.Hider(tx, w, XI, S, fw, "SDIF", H.HideConfig())
    h.run()
    # round 1 on AD deletes D everywhere -> CD and EAD are already hidden at their turn
    assert h.stats["n_pre_hidden"] == 2
    assert [r["pre_hidden"] for r in h.sfwup_log if "pre_hidden" in r] == [True, True]
    assert h.stats["rs_decrease"] == 0.0 and h.stats["rs_delete"] > 0
    assert 0 < h.stats["t_order_s"] and 0 < h.stats["t_victim_s"]


def test_order_wus_no_riswu(monkeypatch):
    tx, w, fw, S = paper()

    def boom(*a, **k):
        raise AssertionError("RISWU must not be used by SC-only + order=wus")

    monkeypatch.setattr(H.Hider, "riswu", boom)
    h = H.Hider(tx, w, XI, S, fw, "SC-only", H.HideConfig(order="wus"))
    q = h.run()
    wus0 = {X: ME.hf_exact(tx, w, 0.0, [X])[1][0][1] for X in S}
    assert h.order == sorted(S, key=lambda X: (-wus0[X], X))
    assert ME.hf_exact(q, w, XI, S)[0] == 0.0


def test_lazy_riswu_same_victims():
    """SC-only / SDIF victims (RISWU computed only where needed) == victims with every item
    scored."""
    rng = random.Random(7)
    for _ in range(30):
        n_items = rng.randint(3, 6)
        tx = [
            {
                i: rng.randint(1, 4)
                for i in rng.sample(range(1, n_items + 1), rng.randint(2, n_items))
            }
            for _ in range(rng.randint(4, 9))
        ]
        w = {i: float(rng.randint(1, 5)) for i in range(1, n_items + 1)}
        fw, _ = M.mine(tx, w, 0.3)
        pool = [p for p in fw if len(p) >= 2]
        if len(pool) < 2:
            continue
        S = pool[:3]
        for st in ("SDIF", "SC-only"):
            h = H.Hider(tx, w, 0.3, S, fw, st, H.HideConfig())
            h._setup()
            h.m = P1.compute(h.tx, h.S, h.O)
            h.wt = h.m["wt"]
            for X in h.S:
                TX = h.tids_of(X)
                r = {i: h.riswu(i, X, TX) for i in X}
                sc, nsc = h.m["sc_item"], h.m["nsc_item"]
                if st == "SC-only":
                    old = max(X, key=lambda i: (sc.get(i, 0), -i))
                else:
                    best = max(sc.get(i, 0) for i in X)
                    c = [i for i in X if sc.get(i, 0) == best]
                    mn = min(nsc.get(i, 0) for i in c)
                    c = [i for i in c if nsc.get(i, 0) == mn]
                    old = c[0] if len(c) == 1 else min(c, key=lambda i: (r[i], i))
                assert h.select_victim(X, TX) == old


def test_xi_target_proposal_paper():
    tx, w, fw, S = paper()
    xt = 0.9 * XI
    q = H.Hider(tx, w, xt, S, fw, "SDIF", H.HideConfig()).run()
    assert ME.hf_exact(q, w, xt, S)[0] == 0.0


def _rand_db(seed, n=60, n_items=8):
    rng = random.Random(seed)
    tx = [
        {i: rng.randint(1, 5) for i in rng.sample(range(1, n_items + 1), rng.randint(2, 6))}
        for _ in range(n)
    ]
    w = {i: float(rng.randint(1, 10)) for i in range(1, n_items + 1)}
    return tx, w


@pytest.mark.parametrize("seed", range(6))
def test_smse_two_thresholds(seed):
    """xi_ns == xi reproduces the one-threshold run; with xi_t < xi every processed X ends below
    xi_t and heap == linear argmin."""
    tx, w = _rand_db(seed)
    xi = 0.25
    fw, _ = M.mine(tx, w, xi)
    S = [p for p in sorted(fw, key=lambda p: (-fw[p], p)) if len(p) >= 2][:4]
    if not S:
        pytest.skip("no pool")
    q1, s1 = BL.smse(tx, w, fw, S, xi=xi)
    q2, s2 = BL.smse(tx, w, fw, S, xi=xi, xi_ns=xi)
    assert q1 == q2 and s1["n_steps"] == s2["n_steps"]
    xt = 0.8 * xi
    tr_h, tr_l = [], []
    qh, sh = BL.smse(tx, w, fw, S, xi=xt, xi_ns=xi, trace=tr_h)
    ql, sl = BL.smse(tx, w, fw, S, xi=xt, xi_ns=xi, argmin="linear", trace=tr_l)
    assert qh == ql
    assert [(t["X"], t["T"], t["I"], t["q"]) for t in tr_h] == [
        (t["X"], t["T"], t["I"], t["q"]) for t in tr_l
    ]
    for xl in sh["xlog"]:
        if xl["action"] == "processed":
            assert xl["wus_done"] < xt  # (it may rise again later: no final check in SMSE)
    qm, sm = BL.msumiu(tx, w, S, xi=xt)
    for xl in sm["xlog"]:
        if xl["action"] == "processed":
            assert xl["wus_done"] < xt
    assert sh["rs_delete"] + sh["rs_decrease"] > 0 or sh["n_steps"] == 0


# ------------------------------------------------------------------ workload files
@pytest.mark.slow
@pytest.mark.parametrize("ds", list(W.X3_GRID))
def test_threshold_workloads(ds):
    fixed = None
    for m in W.X3_GRID[ds]:
        rec = W.read_workload(f"x3_{ds}_{m}")
        fw = W.fwup_of(rec)
        assert rec["xi"] == DA.xi_of(ds, m)
        assert rec["n_S"] == W.n_select(rec["n_pool"])
        Sf = [tuple(x) for x in rec["S_fixed"]]
        fixed = fixed or Sf
        assert Sf == fixed and all(X in fw and fw[X] >= rec["xi"] for X in Sf)


@pytest.mark.parametrize("ds", W.X4_DS)
def test_workload_sets(ds):
    rec = W.read_workload(f"x4_{ds}")
    base = W.read_workload(f"x1_{ds}")
    assert rec["base_sha256"] == base["_sha256"]
    pool = {tuple(X) for X, _ in base["pool"]}
    assert rec["sets"] == W.x4_sets(base)  # reproducible
    for r in W.X4_RHO:
        k = max(5, int(r * base["n_pool"]))
        hi, lo = rec["sets"][f"rho{r:.2f}_ovhigh"], rec["sets"][f"rho{r:.2f}_ovlow"]
        assert len(hi) == len(lo) == k
        assert not ({tuple(x) for x in hi} & {tuple(x) for x in lo})  # rho <= 50 % -> disjoint
        for i in range(5):
            s = rec["sets"][f"rho{r:.2f}_r{i}"]
            assert len(s) == k and {tuple(x) for x in s} <= pool


@pytest.mark.parametrize("ds", W.X5_DS)
@pytest.mark.parametrize("dist", W.X5B_DISTS)
@pytest.mark.parametrize("seed", W.X5B_SEEDS)
def test_utility_tables(ds, dist, seed):
    """The generated utility tables are the documented draws (numpy default_rng(seed))."""
    w = DA.load_weights(DA.weights_path(f"variants/{ds}_{dist}_seed{seed}.txt"))
    w0 = DA.load_weights(DA.weights_path(DA.DATASETS[ds][1]))
    assert sorted(w) == sorted(w0)
    v = np.array([w[i] for i in sorted(w)])
    rng = np.random.default_rng(seed)
    if dist == "normal":
        want = np.clip(np.rint(rng.normal(5.5, 2.0, size=len(v))), 1, 10)
    elif dist == "lognormal10":
        want = np.clip(np.rint(rng.lognormal(1.0, 0.8, size=len(v))), 1, 10)
    else:
        want = rng.integers(1, 11, size=len(v)).astype(np.float64)
    assert np.array_equal(v, want)


@pytest.mark.slow
@pytest.mark.parametrize("ds", W.X6_DS)
def test_scalability_nested(ds):
    recs = {p: W.read_workload(f"x6_{ds}_{p}") for p in (10, 25, 50, 100)}
    n = recs[100]["n_transactions"]
    prev = set()
    for p in (10, 25, 50, 100):
        idx = set(W.sample_indices(n, p / 100).tolist())
        assert len(idx) == recs[p]["n_transactions"] and prev <= idx
        prev = idx
    S = [tuple(x) for x in recs[100]["S"]]
    for p in (10, 25, 50, 100):
        assert [tuple(x) for x in recs[p]["S"]] == S
        assert all(X in W.fwup_of(recs[p]) for X in S)
