"""Resurfacing and the final verification loop (Theorem 3) on a 4-transaction DB: with Max-RISWU
the SFWUP AD falls below xi during its turn but rises above xi again while AC and AB are hidden
(twu of transactions that do not support AD decreases, so D shrinks); without the final check
HF > 0, with it HF = 0. The step-by-step values (twu of every transaction, sumtwu, wus(AD)) are
locked as exact fractions."""

from fractions import Fraction as F

import pytest

from hiding_sfwup import data as DA
from hiding_sfwup import hiding as H
from hiding_sfwup import metrics as ME
from hiding_sfwup import miner as M
from hiding_sfwup.data import twu_of

from examples import RESURFACE_S as S
from examples import RESURFACE_TX as TX
from examples import RESURFACE_W as W
from examples import RESURFACE_XI as XI

L = {1: "A", 2: "B", 3: "C", 4: "D"}
Y = (1, 4)  # AD, the pattern that resurfaces


def name(X):
    return "".join(L[i] for i in X)


def show_t(t):
    return "{" + ", ".join(f"{L[i]}:{q}" for i, q in t.items()) + "}"


def wus_of(tx, X):
    tw = [twu_of(t, W) for t in tx]
    return sum(v for t, v in zip(tx, tw) if all(i in t for i in X)) / sum(tw)


def _run(final_check):
    fw, _ = M.mine(TX, W, XI)
    h = H.Hider(TX, W, XI, S, fw, "Max-RISWU", H.HideConfig(final_check=final_check, trace=True))
    q = h.run()
    return h, q, ME.hf_exact(q, W, XI, S)


def trace():
    """Every modification event with twu of every transaction and sumtwu after it, and the
    relation of the modified T to Y = AD (T keeps Y / T does not contain Y / T loses Y)."""
    fw, _ = M.mine(TX, W, XI)
    events = []

    class Traced(H.Hider):
        def apply(self, k, i, d):
            before = [twu_of(t, W) for t in self.tx]
            sup0 = all(j in self.tx[k] for j in Y)
            super().apply(k, i, d)
            after = [twu_of(t, W) for t in self.tx]
            sup1 = all(j in self.tx[k] for j in Y)
            case = "keeps Y" if sup0 and sup1 else "loses Y" if sup0 else "without Y"
            events.append(
                {
                    "X": name(self._curX),
                    "rehide": self._rehide,
                    "T": f"T{k + 1}",
                    "item": L[i],
                    "diu": d,
                    "twu_after": after,
                    "sumtwu_after": sum(after),
                    "delta": after[k] - before[k],
                    "case_Y": case,
                    "wus_Y_after": wus_of(self.tx, Y),
                }
            )

    h = Traced(TX, W, XI, S, fw, "Max-RISWU", H.HideConfig(final_check=True))
    orig = h.hide_one

    def hide_one(X, rehide=False):
        h._curX, h._rehide = X, rehide
        return orig(X, rehide)

    h.hide_one = hide_one
    q = h.run()
    init = [twu_of(t, W) for t in TX]
    return {
        "init_twu": init,
        "init_sumtwu": sum(init),
        "order": [name(X) for X in h.order],
        "events": events,
        "n_rehide": h.stats["n_rehide"],
        "n_final_rounds": h.stats["n_final_rounds"],
        "qprime": [show_t(t) for t in q],
        "hf": ME.hf_exact(q, W, XI, S)[0],
    }


def fr(x):
    return F(x).limit_denominator(1000)


@pytest.fixture(scope="module")
def tr():
    return trace()


def test_resurfaces_without_final_check():
    h, q, (hf, rows) = _run(False)
    assert h.stats["n_fail_inloop"] == 0
    assert all(r["wus_end"] < XI for r in h.sfwup_log)  # each X below xi when its turn ended
    assert [(tuple(X), ge) for X, _, ge in rows] == [
        ((1, 3), False),
        ((1, 4), True),
        ((1, 2), False),
    ]  # AD is back above xi
    assert hf == pytest.approx(1 / 3)
    h2, q2, (hf2, _) = _run(True)
    assert hf2 == 0.0 and h2.stats["n_rehide"] >= 1
    assert [DA.letters(X) for X in h.order] == ["AD", "AC", "AB"]


def test_initial_state(tr):
    assert [fr(v) for v in tr["init_twu"]] == [5, F(31, 2), 21, F(122, 3)]
    assert fr(tr["init_sumtwu"]) == F(493, 6)
    assert tr["order"] == ["AD", "AC", "AB"]


def test_events(tr):
    ev = tr["events"]
    got = [(e["X"], e["rehide"], e["T"], e["item"], e["diu"], e["case_Y"]) for e in ev]
    assert got == [
        ("AD", False, "T3", "D", 1, "keeps Y"),
        ("AC", False, "T2", "C", 3, "without Y"),
        ("AC", False, "T3", "C", 3, "keeps Y"),
        ("AB", False, "T1", "B", 1, "without Y"),
        ("AB", False, "T3", "B", 1, "keeps Y"),
        ("AD", True, "T3", "D", 1, "keeps Y"),
    ]
    assert [fr(e["twu_after"][2]) for e in ev] == [F(75, 4), F(75, 4), 16, 16, 20, F(31, 2)]
    assert fr(ev[1]["twu_after"][1]) == 4  # T2 loses C: 15.5 -> 4
    sums = [fr(e["sumtwu_after"]) for e in ev]
    assert sums == [F(959, 12), F(821, 12), F(197, 3), F(188, 3), F(200, 3), F(373, 6)]
    wus = [round(e["wus_Y_after"], 4) for e in ev]
    assert wus == [0.2346, 0.2741, 0.2437, 0.2553, 0.3000, 0.2493]
    assert [e["delta"] < 0 for e in ev] == [True, True, True, True, False, True]


def test_end(tr):
    assert tr["n_rehide"] == 1 and tr["n_final_rounds"] == 1 and tr["hf"] == 0.0
    assert tr["qprime"] == ["{A:2}", "{A:4}", "{A:4, D:3}", "{B:4, C:5, D:5}"]


def test_removal_share():
    fw, _ = M.mine(TX, W, XI)
    h = H.Hider(TX, W, XI, S, fw, "Max-RISWU", H.HideConfig())
    h.run()
    # deletions: T2 (15.5), T3 (18.75), T1 (5), T3 (16); decrements: T3 21 -> 18.75, 20 -> 15.5
    assert h.stats["rs_delete"] == pytest.approx(55.25, abs=1e-12)
    assert h.stats["rs_decrease"] == pytest.approx(6.75, abs=1e-12)
    assert (h.stats["n_ev_delete"], h.stats["n_ev_decrease"]) == (4, 2)
    assert h.stats["n_pre_hidden"] == 0
