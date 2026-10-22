"""Proposed algorithm: Algorithm 1 (with its procedure HideOne), Sub-Algorithms 1-4 and the final
verification loop. Line numbers in comments refer to Algorithm 1 and HideOne as in Fig. 1.

  * SWM allocation (Prop. 3-4): alpha = (N_X - xi D) / (1 - xi);
    du_j = alpha * C_j / sum_k (C_k / |S(T_k)|), diu_j = min(iu_j, ceil(du_j / eu(v))).
  * Processing order: RISWU(victim(X), X) descending, victim by the running strategy, computed
    once on the original DB in Phase 2 (lines 5-6 of Algorithm 1); ties -> item-ID tuple ascending.
  * Final verification loop (Phase 3, lines 8-11 of Algorithm 1; Theorem 3): after S is
    processed, every X with exact wus >= xi is hidden again (victim re-selected on the current
    DB, same strategy, Phase-1 SC/NSC/Wt kept, original processing order), until none is left.
  * Hidden means wus < xi strictly; |wus - xi| < 1e-9 is decided exactly (exact.py).
  * MaxIter(X) = sum of iu(victim, T) over T ⊇ X when X starts.
  * No progress (alpha <= 0 or no change) triggers the escalation of Sub-Algorithm 4 at once;
    stagnation of wus over the last k rounds is only recorded.
  * |S(T)| = number of items with iu > 0 (an item reaching 0 is deleted).
Greedy mechanisms (reduction-mechanism ablation): -1 unit per round in all (greedy-batch) or in
the single best (greedy-unit) transaction(s), at most greedy_watchdog rounds per SFWUP.
Every mechanism runs under a wall-clock budget (Phases 1-3); exceeding it marks the run censored.
"""

import math
import statistics
import time
from collections import deque
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from . import exact as EX
from . import phase1 as P1
from .data import twu_of

STRATEGIES = ("SDIF", "Max-RISWU", "Min-RISWU", "SC-only")


@dataclass
class HideConfig:
    stagnation_k: int = 5
    stagnation_epsilon: float = 1e-6
    batch: int = 10  # Tier-1 / Tier-2 batch size of Sub-Algorithm 4
    mechanism: str = "swm"  # swm | greedy-batch | greedy-unit
    riswu_w: str = "full"  # full: W = twu*Wt | no_twu: W = Wt | no_wt: W = twu
    final_check: bool = True
    greedy_watchdog: int = 50_000
    budget_s: float = 7200.0  # wall-clock budget per run; exceeded -> censored
    trace: bool = False  # record per-iteration alpha / changes (small examples)
    regular_step: str = "spec"  # spec | cap: the regular SWM step never deletes the victim
    order: str = "riswu"  # riswu: RISWU(victim, X) desc | wus: wus(X) desc, RISWU not used


class Hider:
    def __init__(self, tx, w, xi, S, O, strategy, cfg=None):
        if strategy not in STRATEGIES:
            raise ValueError(strategy)
        self.tx0 = tx
        self.w = w
        self.xi = float(xi)
        self.S = [tuple(sorted(p)) for p in S]
        self.O = [tuple(sorted(p)) for p in O]
        self.strategy = strategy
        self.cfg = cfg or HideConfig()
        self._ex = None  # exact.ExactTU, built at the first exact decision
        self.stats = {
            "n_iter": 0,
            "n_tier1": 0,
            "n_tier2": 0,
            "n_stagnation": 0,
            "n_rehide": 0,
            "n_final_rounds": 0,
            "n_fail_inloop": 0,
            "censored": False,
            "n_changes": 0,
            "n_deletions": 0,
            "n_regular_iter": 0,
            "n_escalation_iter": 0,
            # bookkeeping only -- does not change any decision
            "n_pre_hidden": 0,
            "n_cap_hit": 0,
            "n_escalate_fail": 0,
            "n_wus_eval": 0,
            "rs_delete": 0.0,
            "rs_decrease": 0.0,
            "n_ev_delete": 0,
            "n_ev_decrease": 0,
            "t_victim_s": 0.0,
            "t_order_s": 0.0,
            # entries into the exact branch / exact result != float result
            "n_guard_alpha": 0,
            "n_guard_qstar": 0,
            "n_guard_qstar_skipped": 0,
            "n_guard_skip": 0,
            "n_guard_flip": 0,
        }
        self.trace = []
        self.sfwup_log = []
        self._acc_dtwu = 0.0
        self._acc_du = 0.0
        self._du_plan = 0.0
        self._acc_red_del = 0.0
        self._acc_red_dec = 0.0

    # ------------------------------------------------------------ primitives
    def _setup(self):
        self.tx = [dict(t) for t in self.tx0]
        items = {i for p in self.S for i in p}
        self.isets = {i: set() for i in items}
        for k, t in enumerate(self.tx):
            for i in t:
                s = self.isets.get(i)
                if s is not None:
                    s.add(k)
        self.twu = np.fromiter(
            (twu_of(t, self.w) for t in self.tx), dtype=np.float64, count=len(self.tx)
        )

    def tids_of(self, X):
        sets = sorted((self.isets[i] for i in X), key=len)
        s = sets[0].intersection(*sets[1:]) if len(sets) > 1 else set(sets[0])
        return np.fromiter(sorted(s), dtype=np.int64, count=len(s))

    def _exact(self):
        if self._ex is None:
            self._ex = EX.ExactTU(self.tx, self.w)
        return self._ex

    def _exact_fn(self, TX):
        def f():
            ex = self._exact()
            return ex.n(TX), ex.d()

        return f

    def nd(self, TX):
        self.stats["n_wus_eval"] += 1
        return float(self.twu[TX].sum()), float(self.twu.sum())

    def wus_ge(self, X, TX=None):
        if TX is None:
            TX = self.tids_of(X)
        if TX.size == 0:
            return False
        n, d = self.nd(TX)
        return EX.wus_ge(n, d, self.xi, self._exact_fn(TX))

    def wus(self, X, TX=None):
        if TX is None:
            TX = self.tids_of(X)
        n, d = self.nd(TX)
        return n / d if d > 0 else 0.0

    def _W(self, TX):
        tw, wt = self.twu[TX], self.wt[TX]
        if self.cfg.riswu_w == "full":
            return tw * wt
        if self.cfg.riswu_w == "no_twu":
            return wt.astype(np.float64)
        if self.cfg.riswu_w == "no_wt":
            return tw.copy()
        raise ValueError(self.cfg.riswu_w)

    def _u(self, i, TX):
        e = self.w.get(i, 0.0)
        return np.fromiter((e * self.tx[k][i] for k in TX), dtype=np.float64, count=TX.size)

    def riswu(self, i, X, TX):
        """RISWU(i, X) = sum u(i,T) W(T,X) / sum W(T,X) over T ⊇ X."""
        W = self._W(TX)
        den = float(W.sum())
        if den <= 0:
            return 0.0
        return float((self._u(i, TX) * W).sum()) / den

    def select_victim(self, X, TX):
        """Sub-Algorithms 2-3. RISWU is computed only where the strategy uses it (SC-only never,
        SDIF only to break an SC/NSC tie)."""
        sc, nsc = self.m["sc_item"], self.m["nsc_item"]
        st = self.strategy
        if st == "Max-RISWU":
            r = {i: self.riswu(i, X, TX) for i in X}
            return max(X, key=lambda i: (r[i], -i))
        if st == "Min-RISWU":
            r = {i: self.riswu(i, X, TX) for i in X}
            return min(X, key=lambda i: (r[i], i))
        if st == "SC-only":
            return max(X, key=lambda i: (sc.get(i, 0), -i))
        # SDIF: max SC, then min NSC, then Min-RISWU, then item ID
        best = max(sc.get(i, 0) for i in X)
        c = [i for i in X if sc.get(i, 0) == best]
        mn = min(nsc.get(i, 0) for i in c)
        c = [i for i in c if nsc.get(i, 0) == mn]
        if len(c) == 1:
            return c[0]
        r = {i: self.riswu(i, X, TX) for i in c}
        return min(c, key=lambda i: (r[i], i))

    def contributions(self, v, TX):
        """C_j = u(v, T_j) * twu(T_j) * Wt(T_j), T_j ⊇ X."""
        C = self._u(v, TX) * self.twu[TX] * self.wt[TX]
        return C

    def prop4_changes(self, v, alpha, TX, C):
        """Prop. 4 allocation of alpha over T_X -> [(tid, victim, diu)]."""
        S_len = np.fromiter((len(self.tx[k]) for k in TX), dtype=np.float64, count=TX.size)
        w_total = math.fsum((C / S_len).tolist())
        if w_total <= 0:
            return []
        e = self.w.get(v, 0.0)
        if e <= 0:
            return []
        self._du_plan = alpha * math.fsum(C.tolist()) / w_total
        out = []
        cap = 1 if self.cfg.regular_step == "cap" else 0  # cap variant: keep >= 1 unit
        for k, c in zip(TX.tolist(), C.tolist()):
            du = alpha * c / w_total
            d = min(self.tx[k][v] - cap, math.ceil(du / e))
            if d > 0:
                out.append((k, v, d))
        return out

    def prop4_exact(self, v, TX):
        """Prop. 3 + Prop. 4 in Fractions, used only inside the exact band.
        -> (alpha, changes, du_plan); changes = [] when alpha <= 0 (-> Sub-Algorithm 4)."""
        ex = self._exact()
        n, d = ex.n(TX), ex.d()
        xi = EX.frac(self.xi)
        alpha = (n - xi * d) / (1 - xi)
        if alpha <= 0:
            return alpha, [], 0.0
        sc, nsc = self.m["sc_trans"], self.m["nsc_trans"]
        tids = TX.tolist()
        e = ex.eu(v)
        C = {k: e * self.tx[k][v] * ex.twu(k) * Fraction(int(sc[k]), int(nsc[k]) + 1) for k in tids}
        w_total = sum((C[k] / len(self.tx[k]) for k in tids), Fraction(0))
        if w_total <= 0 or e <= 0:
            return alpha, [], 0.0
        du_plan = float(alpha * sum(C.values(), Fraction(0)) / w_total)
        out = []
        cap = 1 if self.cfg.regular_step == "cap" else 0
        for k in tids:
            dd = min(self.tx[k][v] - cap, math.ceil(alpha * C[k] / w_total / e))
            if dd > 0:
                out.append((k, v, dd))
        return alpha, out, du_plan

    def greedy_changes(self, v, TX, C):
        cand = sorted(
            ((-c, k) for k, c in zip(TX.tolist(), C.tolist()) if self.tx[k].get(v, 0) > 0)
        )
        if not cand:
            return []
        if self.cfg.mechanism == "greedy-unit":
            return [(cand[0][1], v, 1)]
        return [(k, v, 1) for _, k in cand]

    def apply(self, k, i, d):
        t = self.tx[k]
        if i not in t:
            return
        removed = min(d, t[i])
        t[i] -= d
        deleted = t[i] <= 0
        if deleted:
            del t[i]
            self.stats["n_deletions"] += 1
            s = self.isets.get(i)
            if s is not None:
                s.discard(k)
        old = self.twu[k]
        self.twu[k] = twu_of(t, self.w)
        if self._ex is not None:
            self._ex.update(k, t)
        self.stats["n_changes"] += 1
        # every modification hits a T ⊇ X (X = SFWUP being hidden, i = its victim):
        # deletion -> T leaves T_X, N(X) loses twu(T); decrement -> N(X) loses twu before - after
        if deleted:
            self.stats["rs_delete"] += float(old)
            self.stats["n_ev_delete"] += 1
        else:
            self.stats["rs_decrease"] += float(old - self.twu[k])
            self.stats["n_ev_decrease"] += 1
        self._acc_dtwu += old - self.twu[k]  # overshoot bookkeeping
        self._acc_du += removed * self.w.get(i, 0.0)
        # decrease of the gap N_X - xi*D of the SFWUP being hidden
        if deleted:
            self._acc_red_del += old + self.xi * (self.twu[k] - old)
        else:
            self._acc_red_dec += (1.0 - self.xi) * (old - self.twu[k])

    def _stagnated(self, hist):
        if len(hist) < self.cfg.stagnation_k:
            return False
        h = list(hist)
        deltas = [abs(h[j] - h[j - 1]) for j in range(1, len(h))]
        return bool(deltas) and statistics.median(deltas) < self.cfg.stagnation_epsilon

    def escalate(self, X, v, TX, C):
        """Sub-Algorithm 4. Tier 1: delete v from <= batch T ⊇ X with iu(v) = 1 (contribution
        desc, tid asc); Tier 2: delete v entirely from <= batch T ⊇ X (same order).
        The applied changes (tid, item, diu) are kept in self._esc_changes."""
        cmap = dict(zip(TX.tolist(), C.tolist()))
        self._esc_changes = []
        t1 = sorted((-cmap.get(k, 0.0), k) for k in TX.tolist() if self.tx[k].get(v) == 1)
        if t1:
            for _, k in t1[: self.cfg.batch]:
                self._esc_changes.append((k, v, 1))
                self.apply(k, v, 1)
            self.stats["n_tier1"] += 1
            return True
        t2 = sorted((-cmap.get(k, 0.0), k) for k in TX.tolist() if self.tx[k].get(v, 0) > 0)
        if t2:
            for _, k in t2[: self.cfg.batch]:
                self._esc_changes.append((k, v, self.tx[k][v]))
                self.apply(k, v, self.tx[k][v])
            self.stats["n_tier2"] += 1
            return True
        return False

    # ------------------------------------------------------------ procedure HideOne
    def hide_one(self, X, rehide=False):
        cfg = self.cfg
        TX = self.tids_of(X)
        rec = {
            "X": list(X),
            "rehide": rehide,
            "wus0": self.wus(X, TX),
            "iters": 0,
            "victim": None,
            "phi0": 0,
            "ok": True,
            "escalated": False,
        }
        if not self.wus_ge(X, TX):  # HideOne line 1
            # already below xi at its turn; the flag is meaningful only in the main pass
            # (a re-hide call is still counted in n_rehide)
            rec["pre_hidden"] = not rehide
            if not rehide:
                self.stats["n_pre_hidden"] += 1
            self.sfwup_log.append(rec)
            return True
        tv = time.perf_counter()
        v = self.select_victim(X, TX)  # HideOne line 2 (once per call)
        self.stats["t_victim_s"] += time.perf_counter() - tv
        phi0 = int(sum(self.tx[k][v] for k in TX.tolist()))
        max_iter = phi0 if cfg.mechanism == "swm" else cfg.greedy_watchdog
        rec.update(victim=v, phi0=phi0)
        hist = deque(maxlen=cfg.stagnation_k)
        it = 0
        self._acc_dtwu = 0.0
        self._acc_du = 0.0
        self._acc_red_del = 0.0
        self._acc_red_dec = 0.0
        alpha0 = gap0 = None
        du_plan = 0.0
        while True:
            TX = self.tids_of(X)
            if TX.size == 0:  # wus(X) = 0: the while test of line 4 fails
                break
            n, d = self.nd(TX)
            if not EX.wus_ge(n, d, self.xi, self._exact_fn(TX)):  # HideOne line 4 (strict)
                break
            it += 1
            self.stats["n_iter"] += 1
            if it > max_iter:  # MaxIter(X) reached
                rec["ok"] = False
                rec["cap_hit"] = True
                self.stats["n_cap_hit"] += 1
                break
            if self.deadline is not None and time.time() > self.deadline:
                self.stats["censored"] = True
                rec["ok"] = False
                break
            hist.append(n / d)
            C = self.contributions(v, TX)
            changes = []
            alpha = None
            if cfg.mechanism == "swm":
                alpha = (n - self.xi * d) / (1.0 - self.xi)  # Prop. 3, HideOne line 8
                if alpha0 is None:
                    alpha0, gap0 = alpha, n - self.xi * d
                du_it = 0.0
                if alpha > 0:
                    self._du_plan = 0.0
                    changes = self.prop4_changes(v, alpha, TX, C)
                    du_it = self._du_plan
                if abs(n / d - self.xi) < EX.BAND:  # inside the band alpha and Prop. 4
                    self.stats["n_guard_alpha"] += 1  # are decided exactly
                    alpha_x, changes_x, du_x = self.prop4_exact(v, TX)
                    if changes_x != changes:
                        self.stats["n_guard_flip"] += 1
                        alpha, changes, du_it = float(alpha_x), changes_x, du_x
                du_plan += du_it
            else:
                changes = self.greedy_changes(v, TX, C)
            if cfg.trace:
                self.trace.append(
                    {
                        "X": list(X),
                        "iter": it,
                        "victim": v,
                        "N": n,
                        "D": d,
                        "alpha": alpha,
                        "changes": [(k, i, dd) for k, i, dd in changes],
                    }
                )
            if changes:  # HideOne lines 10-12: apply, continue
                for k, i, dd in changes:
                    self.apply(k, i, dd)
                self.stats["n_regular_iter"] += 1
                if cfg.trace:
                    self.trace[-1]["wus_after"] = self.wus(X)
                continue
            self.stats["n_escalation_iter"] += 1
            if self._stagnated(hist):  # stagnation (recorded only)
                self.stats["n_stagnation"] += 1
            rec["escalated"] = True  # HideOne line 14: Sub-Algorithm 4
            t1 = self.stats["n_tier1"]
            if not self.escalate(X, v, TX, C):
                rec["ok"] = False
                rec["escalate_fail"] = True  # Sub-Algorithm 4 failed with T_X != ∅
                self.stats["n_escalate_fail"] += 1
                break
            if cfg.trace:
                self.trace[-1].update(
                    escalation="tier1" if self.stats["n_tier1"] > t1 else "tier2",
                    changes=list(self._esc_changes),
                    wus_after=self.wus(X),
                )
        rec["iters"] = it
        rec["wus_end"] = self.wus(X)
        TXe = self.tids_of(X)
        n1, d1 = self.nd(TXe) if TXe.size else (0.0, float(self.twu.sum()))
        rec.update(
            alpha_ratio=((gap0 - (n1 - self.xi * d1)) / gap0) if gap0 else None,
            gap0=gap0,
            red_delete=self._acc_red_del,
            red_decrease=self._acc_red_dec,
            alpha0=alpha0,
            dtwu_total=self._acc_dtwu,
            du_total=self._acc_du,
            du_plan_total=du_plan,
            overshoot_twu=(self._acc_dtwu / alpha0) if alpha0 else None,
        )
        if not rec["ok"]:
            self.stats["n_fail_inloop"] += 1
        self.sfwup_log.append(rec)
        return rec["ok"]

    # ------------------------------------------------------------ driver
    def processing_order(self):
        """RISWU(victim(X), X) descending on the original DB (called before any modification);
        ties -> item-ID tuple ascending. cfg.order == "wus": wus(X) descending on the original
        DB, same tie-break, RISWU not used (key = (wus, victim))."""
        keys = {}
        for X in self.S:
            TX = self.tids_of(X)
            v = self.select_victim(X, TX)
            if self.cfg.order == "wus":
                keys[X] = (self.wus(X, TX), v)
            else:
                keys[X] = (self.riswu(v, X, TX), v)
        order = sorted(self.S, key=lambda X: (-keys[X][0], X))
        return order, {X: keys[X] for X in order}

    def run(self):
        self.deadline = time.time() + self.cfg.budget_s if self.cfg.budget_s > 0 else None
        w0, c0 = time.perf_counter(), time.process_time()
        self._setup()
        self.m = P1.compute(self.tx, self.S, self.O)
        self.wt = self.m["wt"]
        w1, c1 = time.perf_counter(), time.process_time()
        self.order, self.order_keys = self.processing_order()
        self.stats["t_order_s"] = time.perf_counter() - w1
        for X in self.order:
            if self.stats["censored"]:
                break
            self.hide_one(X)  # Algorithm 1 line 7
        if self.cfg.final_check:  # final verification loop (lines 8-11; Theorem 3)
            while not self.stats["censored"]:
                again = [X for X in self.order if self.wus_ge(X)]
                if not again:
                    break
                self.stats["n_final_rounds"] += 1
                before = self.stats["n_changes"]
                for X in again:
                    self.hide_one(X, rehide=True)
                    self.stats["n_rehide"] += 1
                if self.stats["n_changes"] == before:  # cannot happen for swm (safety)
                    self.stats["final_check_stuck"] = True
                    break
        w2, c2 = time.perf_counter(), time.process_time()
        logs = [r for r in self.sfwup_log if r.get("gap0")]
        self.stats["red_delete"] = sum(r["red_delete"] for r in logs)
        self.stats["red_decrease"] = sum(r["red_decrease"] for r in logs)
        self.times = {
            "rt_phase1": w1 - w0,
            "rt_hide": w2 - w1,
            "rt_total": w2 - w0,
            "cpu_phase1": c1 - c0,
            "cpu_hide": c2 - c1,
            "cpu_total": c2 - c0,
        }
        return self.tx
