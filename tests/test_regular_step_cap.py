"""Cap variant of the regular SWM step (cap ablation): the regular step never deletes the victim
(diu_j = min(iu_j - 1, ceil(du_j/eu))); no progress -> escalation (Sub-Algorithm 4).
The default is the specified step ("spec")."""

import pytest

from hiding_sfwup import data as DA
from hiding_sfwup import hiding as H
from hiding_sfwup import metrics as ME
from hiding_sfwup import miner as M
from hiding_sfwup import phase1 as P1

A, B, C, D, E = 1, 2, 3, 4, 5
S = [(A, D), (C, D), (A, D, E)]


def _run(st, step):
    tx, w = DA.paper_example()
    fw, _ = M.mine(tx, w, 0.4)
    h = H.Hider(tx, w, 0.4, S, fw, st, H.HideConfig(trace=True, regular_step=step))
    q = h.run()
    return h, q, w


def test_default_is_spec():
    assert H.HideConfig().regular_step == "spec"


def test_cap_paper_example_sdif(monkeypatch):
    # the trace lists the Tier-1/2 changes as (tid, item, diu); they are checked against the
    # DB diff around each escalate() call, independently of the trace code
    esc_diff = []
    orig = H.Hider.escalate

    def spy(self, X, v, TX, C):
        before = [dict(t) for t in self.tx]
        ok = orig(self, X, v, TX, C)
        esc_diff.append(
            sorted(
                (k, i, before[k][i] - self.tx[k].get(i, 0))
                for k in range(len(before))
                for i in before[k]
                if before[k][i] != self.tx[k].get(i, 0)
            )
        )
        return ok

    monkeypatch.setattr(H.Hider, "escalate", spy)
    h, q, w = _run("SDIF", "cap")
    esc = [t for t in h.trace if t.get("escalation")]
    assert [sorted(tuple(c) for c in t["changes"]) for t in esc] == esc_diff
    steps = [
        (
            DA.letters(t["X"]),
            t["iter"],
            round(t["alpha"], 6),
            [(k + 1, d) for k, _, d in t["changes"]],
            t.get("escalation"),
            round(t["wus_after"], 6),
        )
        for t in h.trace
    ]
    assert steps == [
        ("AD", 1, 2.347222, [(4, 1), (5, 2)], None, 0.774648),
        ("AD", 2, 1.847222, [(1, 1), (3, 1), (4, 1), (5, 1)], "tier1", 0.0),
    ]
    assert all(
        i == D for t in esc for _, i, _ in t["changes"]
    )  # Sub-Alg. 4 touches only the victim D
    assert h.stats["n_regular_iter"] == 1 and h.stats["n_tier1"] == 1 and h.stats["n_tier2"] == 0
    assert h.stats["n_deletions"] == 4  # all deletions come from Tier 1
    assert ME.hf_exact(q, w, 0.4, S)[0] == 0.0


@pytest.mark.parametrize("st", H.STRATEGIES)
def test_cap_regular_step_never_deletes(st):
    tx, w = DA.paper_example()
    fw, _ = M.mine(tx, w, 0.4)
    h = H.Hider(tx, w, 0.4, S, fw, st, H.HideConfig(trace=True, regular_step="cap"))
    h._setup()
    h.m = P1.compute(h.tx, h.S, h.O)
    h.wt = h.m["wt"]
    for X in S:
        TX = h.tids_of(X)
        v = h.select_victim(X, TX)
        n, d = h.nd(TX)
        ch = h.prop4_changes(v, (n - 0.4 * d) / 0.6, TX, h.contributions(v, TX))
        assert all(dd <= h.tx[k][v] - 1 for k, _, dd in ch)
    h2, q, _ = _run(st, "cap")
    assert ME.hf_exact(q, w, 0.4, S)[0] == 0.0


@pytest.mark.parametrize("step", ["spec", "cap"])
def test_gap_decomposition_sums_to_closure(step):
    """red_delete + red_decrease (own changes) == closure of N_X - xi*D (alpha_ratio * gap0)."""
    h, q, w = _run("SDIF", step)
    for r in h.sfwup_log:
        if r.get("gap0"):
            assert r["red_delete"] + r["red_decrease"] == pytest.approx(
                r["alpha_ratio"] * r["gap0"], rel=1e-12
            )
