"""Wall-clock budget (7200 s for every method; exceeded -> censored), feasibility, and the
bookkeeping of the baselines (per-SFWUP log, gap decomposition, CPU of the hiding phase)."""

import pytest

from hiding_sfwup import baselines as B
from hiding_sfwup import data as DA
from hiding_sfwup import hiding as H
from hiding_sfwup import metrics as ME
from hiding_sfwup import miner as M

S = [(1, 4), (3, 4), (1, 4, 5)]


def _ex():
    tx, w = DA.paper_example()
    fw, _ = M.mine(tx, w, 0.4)
    return tx, w, fw


def test_default_budget_7200():
    assert H.HideConfig().budget_s == 7200.0


@pytest.mark.parametrize("mech", ["swm", "greedy-batch", "greedy-unit"])
def test_budget_exceeded_is_censored(mech):
    tx, w, fw = _ex()
    h = H.Hider(tx, w, 0.4, S, fw, "SDIF", H.HideConfig(mechanism=mech, budget_s=1e-9))
    q = h.run()
    assert h.stats["censored"] is True
    assert ME.hf_exact(q, w, 0.4, S)[0] > 0  # nothing could be hidden -> not feasible


def test_proposal_decomposition_totals():
    tx, w, fw = _ex()
    h = H.Hider(tx, w, 0.4, S, fw, "SDIF")
    h.run()
    logs = [r for r in h.sfwup_log if r.get("gap0")]
    assert h.stats["red_delete"] == pytest.approx(sum(r["red_delete"] for r in logs))
    assert h.stats["red_decrease"] == 0.0  # spec on the example: only deletions


@pytest.mark.parametrize("method", ["SMSE", "MSU-MIU"])
def test_baseline_xlog_and_decomposition(method):
    tx, w, fw = _ex()
    if method == "SMSE":
        q, st = B.smse(tx, w, fw, S, xi=0.4)
    else:
        q, st = B.msumiu(tx, w, S, xi=0.4)
    assert [x["X"] for x in st["xlog"]] == st["order"]
    assert all(x["action"] in ("processed", "skip_removed", "skip_below") for x in st["xlog"])
    proc = [x for x in st["xlog"] if x["action"] == "processed"]
    assert proc and all(x["wus_done"] < 0.4 for x in proc)
    assert st["red_delete"] + st["red_decrease"] > 0
    assert st["cpu_hide"] is not None and st["cpu_build"] is not None
