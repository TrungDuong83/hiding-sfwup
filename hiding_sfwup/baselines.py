"""Baselines re-implemented from the original papers, on the wus measure.

  SMSE     [SMSE]    Liu, Wen, Zuo, Applied Intelligence 50 (2020), Algorithm 3 (p. 178)
  MSU-MIU  [MSU-MIU] Lin et al., Eng. Appl. Artif. Intell. 55 (2016), Algorithm 2 (p. 277)

The core (`Tables`) is parameterised by the measure:
  measure="wus"     N_Y = sum_{T ⊇ Y} twu(T); active iff N_Y >= xi*D (D global)
  measure="utility" N_Y = u(Y) = sum_{T ⊇ Y} u(Y, T); active iff N_Y >= minutil
The utility mode reproduces the worked examples of the two papers (tests).

Adaptation choices: MSU-MIU weight w(X) = sum_{i in X} f(i); an item is decreased by
q* = floor(alpha*|S(T)|/eu(I)) + 1 (the smallest q with q*eu/|S(T)| > alpha); SMSE takes
argmin (NSHC(T), tid) from a lazy heap (equivalent to the linear argmin, tested); patterns that
fall below xi*D when D increases are found through a lazy min-heap over N_Y of the active patterns.
RT = table construction (two scans) + hiding; a wall-clock budget marks the run censored.
Decisions with |N_Y/D - xi| < 1e-9 are taken exactly (exact.py), as in the proposed method.
"""

import heapq
import math
import time

import numpy as np

from . import exact as EX
from .data import twu_of


class Timeout(Exception):
    pass


class Tables:
    """FWUP-table (HUI-table of [SMSE]) + T-table restricted to transactions that
    support at least one SFWUP (Def. 12 of [SMSE]).

    Pattern-level arrays (index p): N[p], removed[p], sens[p], items[p].
    T-table rows r (one per sensitive transaction, ascending tid) in CSR form:
    rptr[r]:rptr[r+1] slice of `rpat` (pattern ids supported by T at build
    time) with the parallel flag `ralive` (T still ⊇ pattern). Transposed CSR
    pptr/prow/pslot gives, for every pattern, its (row, slot) pairs.
    """

    def __init__(
        self,
        tx,
        w,
        patterns,
        sens_mask,
        measure="wus",
        xi=None,
        minutil=None,
        track_removal=True,
        nshc=False,
        xi_ns=None,
    ):
        self.measure = measure
        self.w = w
        self.xi = xi
        # equal-margin runs: the SFWUP are hidden below xi (= xi_target) while the NS patterns
        # stay defined at the original threshold xi_ns. None -> one threshold.
        self.xi_ns = None if (xi_ns is None or xi_ns == xi) else xi_ns
        self.minutil = minutil
        self.track_removal = track_removal
        self.do_nshc = nshc
        n = len(tx)
        # ---- scan 1: I-lists, TU, |S|, twu, D; item postings for the pattern items
        self.tx = [dict(t) for t in tx]
        pat_items = sorted({i for p in patterns for i in p})
        ipos = {i: k for k, i in enumerate(pat_items)}
        post = [[] for _ in pat_items]
        twu = np.empty(n, dtype=np.float64)
        for k, t in enumerate(self.tx):
            twu[k] = twu_of(t, w)
            for i in t:
                r = ipos.get(i)
                if r is not None:
                    post[r].append(k)
        self.twu = twu
        self.D = float(twu.sum())
        post = [np.asarray(a, dtype=np.int64) for a in post]
        # ---- scan 2: relation T <-> patterns (tidsets of every pattern)
        P = len(patterns)
        self.items = [tuple(sorted(p)) for p in patterns]
        self.sens = np.asarray(sens_mask, dtype=bool)
        # initial numerators + sensitive transactions (union of SFWUP tidsets)
        self.N = np.zeros(P, dtype=np.float64)

        def tidset(p):
            arrs = sorted((post[ipos[i]] for i in p), key=len)
            ts = arrs[0]
            for a in arrs[1:]:
                ts = np.intersect1d(ts, a, assume_unique=True)
            return ts

        sens_ts = []
        for p in range(P):
            if self.sens[p]:
                sens_ts.append(tidset(self.items[p]))
        sensT = np.unique(np.concatenate(sens_ts)) if sens_ts else np.empty(0, np.int64)
        self.sensT = sensT
        # CSR rows: patterns supported by each sensitive transaction (scan 2)
        pairs_r, pairs_p = [], []
        for p in range(P):
            ts = tidset(self.items[p])
            if measure == "wus":
                self.N[p] = float(twu[ts].sum())
            else:
                self.N[p] = math.fsum(
                    sum(w[i] * self.tx[k][i] for i in self.items[p]) for k in ts.tolist()
                )
            rows = np.searchsorted(sensT, ts)
            ok = rows < sensT.size
            ok[ok] = sensT[rows[ok]] == ts[ok]
            pairs_r.append(rows[ok].astype(np.int32))
            pairs_p.append(np.full(int(ok.sum()), p, dtype=np.int32))
        del sens_ts
        pr = np.concatenate(pairs_r) if pairs_r else np.empty(0, np.int32)
        pp = np.concatenate(pairs_p) if pairs_p else np.empty(0, np.int32)
        del pairs_r, pairs_p
        order = np.lexsort((pp, pr))
        pr, pp = pr[order], pp[order]
        R = sensT.size
        self.rptr = np.zeros(R + 1, dtype=np.int64)
        np.add.at(self.rptr, pr + 1, 1)
        self.rptr = np.cumsum(self.rptr)
        self.rpat = pp
        self.ralive = np.ones(pp.size, dtype=bool)
        # transposed CSR (pattern -> slots)
        order2 = np.lexsort((pr, pp))
        self.pslot = order2.astype(np.int64)  # slot indices into rpat
        cnt = np.bincount(pp, minlength=P)
        self.pptr = np.concatenate([[0], np.cumsum(cnt)])
        # item -> pattern membership masks
        self.has = {i: np.zeros(P, dtype=bool) for i in pat_items}
        for p, its in enumerate(self.items):
            for i in its:
                self.has[i][p] = True
        self.removed = np.zeros(P, dtype=bool)
        # NSHC(T) = # active NS patterns supported by T
        if nshc:
            ns_slot = ~self.sens[self.rpat]
            self.NSHC = (
                np.add.reduceat(ns_slot.astype(np.int64), self.rptr[:-1])
                if R
                else np.zeros(0, np.int64)
            )
            if R:
                empty = self.rptr[:-1] == self.rptr[1:]
                self.NSHC[empty] = 0
        self.nshc_hook = None
        # lazy min-heap over N of the active patterns + dirty set
        self.dirty = np.zeros(P, dtype=bool)
        self.dirty_ids = []
        self.n_dirty = 0
        self.heap_ns = []
        self._rebuild_heap()
        self._ex = None  # exact.ExactTU, built at the first exact decision
        self.red_delete = 0.0  # decrease of N_X - xi*D of the SFWUP being hidden
        self.red_decrease = 0.0
        self.rs_delete = 0.0  # decrease of N(X) = sum twu(T), T ⊇ X
        self.rs_decrease = 0.0
        self.n_ev_delete = 0
        self.n_ev_decrease = 0
        self.n_d_increase = 0
        self.n_heap_rebuild = 0
        self.n_guard_qstar = 0  # entries into the exact branch / exact != float
        self.n_guard_qstar_skipped = 0  # q* in the band but min(iu, q*) = iu for every q*
        self.n_guard_skip = 0
        self.n_guard_flip = 0

    # -------------------------------------------------------------- thresholds
    def thr(self):
        return self.xi * self.D if self.measure == "wus" else self.minutil

    def below(self, p):
        return self.N[p] < self.thr()

    def _exact(self):
        if self._ex is None:
            self._ex = EX.ExactTU(self.tx, self.w)
        return self._ex

    def _in_band(self, p, xi):
        return self.measure == "wus" and self.D > 0 and abs(self.N[p] / self.D - xi) < EX.BAND

    def _below_exact(self, p, xi, b):
        """N_p < xi D in Fractions (pattern p inside the exact band); b = float decision.
        N_p over every T ⊇ p (NS patterns also live in non-sensitive transactions)."""
        self.n_guard_skip += 1
        ex = self._exact()
        if self.sens[p]:  # every T ⊇ X is a T-table row
            ks = self.sensT[self.rows_of(p)]
        else:
            its = self.items[p]
            ks = [k for k, t in enumerate(self.tx) if all(i in t for i in its)]
        n, d = ex.n(ks), ex.d()
        bx = not (d > 0 and n / d >= EX.frac(xi))
        if bx != b:
            self.n_guard_flip += 1
        return bx

    def below_turn(self, p):
        """Every 'N_X < xi D' decision on an SFWUP (skip at its turn, SMSE l. 5, MSU-MIU stop
        after a deletion) re-decided exactly inside the band (|N_X/D - xi| < BAND), as the stop
        test (exact_resync) is; outside the band identical to below()."""
        b = self.below(p)
        if not self._in_band(p, self.xi):
            return b
        return self._below_exact(p, self.xi, b)

    def thr_ns(self):
        return self.xi_ns * self.D

    def _rebuild_heap(self):
        act = np.flatnonzero(~self.removed)
        if self.xi_ns is not None:  # two classes, one heap each (keys = N)
            ns = act[~self.sens[act]]
            act = act[self.sens[act]]
            self.heap_ns = list(zip(self.N[ns].tolist(), ns.tolist()))
            heapq.heapify(self.heap_ns)
        self.heap = list(zip(self.N[act].tolist(), act.tolist()))
        heapq.heapify(self.heap)
        self.dirty[:] = False
        self.dirty_ids = []
        self.n_dirty = 0

    def _mark_dirty(self, ids):
        new = ids[~self.dirty[ids]]
        if new.size:
            self.dirty[new] = True
            self.dirty_ids.append(new)
            self.n_dirty += new.size

    def remove_pattern(self, p):
        """[SMSE] p. 175: a pattern below the threshold leaves the HUI-table and the
        SID/NSID of every transaction (never re-added)."""
        if self.removed[p]:
            return
        self.removed[p] = True
        if self.do_nshc and not self.sens[p]:
            slots = self.pslot[self.pptr[p] : self.pptr[p + 1]]
            slots = slots[self.ralive[slots]]
            if slots.size:
                rows = np.searchsorted(self.rptr, slots, side="right") - 1
                np.subtract.at(self.NSHC, rows, 1)
                if self.nshc_hook is not None:
                    self.nshc_hook(rows)

    def _check_removed(self, ids):
        if not self.track_removal or ids.size == 0:
            return
        ids = ids[~self.removed[ids]]
        if self.measure != "wus" or self.D <= 0:
            if self.xi_ns is None:
                m = self.N[ids] < self.thr()
            else:
                m = self.N[ids] < np.where(self.sens[ids], self.thr(), self.thr_ns())
        else:
            # one vector test cand = N < (xi + BAND) D; inside cand, N < (xi - BAND) D falls for
            # sure (float and exact agree); only the rest of cand is band-checked (same test as
            # below_turn) and decided exactly; outside cand the pattern stays (N >= xi D)
            D = self.D
            xis = self.xi if self.xi_ns is None else np.where(self.sens[ids], self.xi, self.xi_ns)
            m = self.N[ids] < (xis + EX.BAND) * D
            cand = np.flatnonzero(m)
            if cand.size:
                xc = xis if self.xi_ns is None else xis[cand]
                Nc = self.N[ids[cand]]
                for j in cand[Nc >= (xc - EX.BAND) * D].tolist():
                    p = int(ids[j])
                    xi = self.xi if self.xi_ns is None or self.sens[p] else self.xi_ns
                    b = bool(self.N[p] < xi * D)  # float decision
                    m[j] = self._below_exact(p, xi, b) if self._in_band(p, xi) else b
        fall = ids[m]
        for p in fall.tolist():
            self.remove_pattern(p)

    def _pop_fallen(self, h, th, xi):
        """Pop the clean patterns whose key N is below th = xi D. Keys inside the
        exact band are popped too and decided exactly (same test as below_turn); a popped
        pattern still >= xi stays active and is marked dirty (re-checked on later D increases).
        The loop test N < (xi + BAND) D is evaluated first, so the band test runs only near xi."""
        band = self.measure == "wus" and self.D > 0
        hi = (xi + EX.BAND) * self.D if band else th
        while h and h[0][0] < hi:
            top = h[0][0]
            if not (top < th or (band and abs(top / self.D - xi) < EX.BAND)):
                break
            val, p = heapq.heappop(h)
            if self.removed[p] or self.dirty[p] or val != self.N[p]:
                continue
            if band and self._in_band(p, xi):
                if not self._below_exact(p, xi, val < th):
                    self._mark_dirty(np.array([p], dtype=np.int64))
                    continue
            self.remove_pattern(p)

    def _on_d_increase(self):
        """D grew, so ξ·D grew; untouched patterns may fall. Dirty patterns
        (N changed since the heap was built) are checked directly; clean ones are
        popped from the heap while their (exact) key is below ξ·D."""
        self.n_d_increase += 1
        th = self.thr()
        if self.dirty_ids:
            ids = np.concatenate(self.dirty_ids)
            self.dirty_ids = [ids]
            self._check_removed(ids)
        self._pop_fallen(self.heap, th, self.xi)
        if self.xi_ns is not None:
            self._pop_fallen(self.heap_ns, self.thr_ns(), self.xi_ns)
        h = self.heap
        if self.n_dirty > max(
            4096, (len(h) + len(self.heap_ns)) // 4 if self.xi_ns is not None else len(h) // 4
        ):
            self._rebuild_heap()
            self.n_heap_rebuild += 1

    # -------------------------------------------------------------- APPLY
    def row_slice(self, r):
        return self.rptr[r], self.rptr[r + 1]

    def u_pattern_T(self, p, k):
        t = self.tx[k]
        return sum(self.w[i] * t[i] for i in self.items[p])

    def apply(self, r, I, q):
        """APPLY(T, I, q): delete I from T when
        q >= iu(I,T), otherwise iu(I,T) -= q; update N, D, NSHC, removals."""
        k = int(self.sensT[r])
        t = self.tx[k]
        a, b = self.row_slice(r)
        ids = self.rpat[a:b]
        alive = self.ralive[a:b]
        hasI = self.has.get(I)
        withI = alive & hasI[ids] if hasI is not None else np.zeros(b - a, dtype=bool)
        twu_old = self.twu[k]
        delete = q >= t[I]
        if self.measure == "wus":
            if delete:
                lose = withI
                lose_ids = ids[lose]
                act_lose = lose_ids[~self.removed[lose_ids]]
                self.N[act_lose] -= twu_old
                if self.do_nshc:
                    dec = int((~self.sens[act_lose]).sum())
                    if dec:
                        self.NSHC[r] -= dec
                        if self.nshc_hook is not None:
                            self.nshc_hook(np.array([r]))
                self.ralive[a:b][lose] = False
                del t[I]
            else:
                t[I] -= q
            twu_new = twu_of(t, self.w)
            self.twu[k] = twu_new
            delta = twu_new - twu_old
            if delete:
                self.red_delete += twu_old + self.xi * delta
                self.rs_delete += float(twu_old)  # T ⊇ X loses I ∈ X -> leaves T_X
                self.n_ev_delete += 1
            else:
                self.red_decrease += -(1.0 - self.xi) * delta
                self.rs_decrease += float(-delta)
                self.n_ev_decrease += 1
            self.D += delta
            still = self.ralive[a:b]
            st_ids = ids[still]
            st_ids = st_ids[~self.removed[st_ids]]
            self.N[st_ids] += delta
            touched = np.concatenate([act_lose, st_ids]) if delete else st_ids
        else:  # utility
            if delete:
                lose_ids = ids[withI]
                act_lose = lose_ids[~self.removed[lose_ids]]
                for p in act_lose.tolist():
                    self.N[p] -= self.u_pattern_T(p, k)
                if self.do_nshc:
                    dec = int((~self.sens[act_lose]).sum())
                    if dec:
                        self.NSHC[r] -= dec
                        if self.nshc_hook is not None:
                            self.nshc_hook(np.array([r]))
                self.ralive[a:b][withI] = False
                del t[I]
                touched = act_lose
            else:
                t[I] -= q
                dec_ids = ids[withI]
                dec_ids = dec_ids[~self.removed[dec_ids]]
                self.N[dec_ids] -= q * self.w[I]
                touched = dec_ids
            self.twu[k] = twu_of(t, self.w)
            delta = 0.0
        if self._ex is not None:
            self._ex.update(k, t)
        if touched.size:
            self._mark_dirty(touched)
        self._check_removed(touched)
        if self.measure == "wus" and delta > 0 and self.track_removal:
            self._on_d_increase()
        return delete

    # -------------------------------------------------------------- helpers
    def wus_now(self, p):
        """Current wus of pattern p from the live T-table (valid also for removed p)."""
        if self.measure != "wus":
            return None
        ks = self.sensT[self.rows_of(p)]
        d = float(self.twu.sum())
        return float(self.twu[ks].sum()) / d if d > 0 else 0.0

    def rows_of(self, p):
        """Rows (sensitive transactions) still supporting pattern p."""
        slots = self.pslot[self.pptr[p] : self.pptr[p + 1]]
        slots = slots[self.ralive[slots]]
        return np.searchsorted(self.rptr, slots, side="right") - 1

    def alive_in_row(self, p, r):
        a, b = self.row_slice(r)
        ids = self.rpat[a:b]
        j = np.searchsorted(ids, p)
        return j < ids.size and ids[j] == p and bool(self.ralive[a + j])

    def q_star(self, p, r, I):
        """wus: q* = floor(alpha*|S(T)|/eu(I)) + 1, alpha = (N_X - xi D)/(1 - xi)
        (smallest q with q*eu/|S| > alpha); utility: ceil(diffu/eu(I)),
        diffu = u(X) - minutil + 1 ([SMSE] eq. 5, Alg. 3 l. 17)."""
        k = int(self.sensT[r])
        e = self.w[I]
        if self.measure == "wus":
            alpha = (self.N[p] - self.xi * self.D) / (1.0 - self.xi)
            x = alpha * len(self.tx[k]) / e
            q = int(math.floor(x)) + 1
            delta = EX.BAND * max(1.0, abs(x))
            if abs(x - round(x)) <= delta:  # floor near an integer
                # q* is used only through min(iu, q*): if the smallest q* possible in the band
                # already reaches iu(I, T), every q* in the band deletes I from T -> keep float
                if int(math.floor(x - delta)) + 1 >= self.tx[k][I]:
                    self.n_guard_qstar_skipped += 1
                    return q
                self.n_guard_qstar += 1  # -> exact, both directions
                ex = self._exact()
                ks = self.sensT[self.rows_of(p)]
                xi = EX.frac(self.xi)
                ax = (ex.n(ks) - xi * ex.d()) / (1 - xi)
                xx = ax * len(self.tx[k]) / ex.eu(I)
                qx = xx.numerator // xx.denominator + 1
                if qx != q:
                    self.n_guard_flip += 1
                    q = qx
            return q
        diffu = self.N[p] - self.minutil + 1
        return int(math.ceil(diffu / e))

    def exact_resync(self, p):
        """Recompute N_p and D from scratch (fsum / pairwise) and decide
        'p still at or above threshold' exactly."""
        rows = self.rows_of(p)
        ks = self.sensT[rows]
        if self.measure == "wus":
            self.D = float(self.twu.sum())
            n = math.fsum(self.twu[ks].tolist()) if ks.size else 0.0
            if not ks.size:
                self.N[p] = 0.0
                return False
            self.N[p] = n
            return EX.wus_ge(n, self.D, self.xi, lambda: (self._exact().n(ks), self._exact().d()))
        n = math.fsum(self.u_pattern_T(p, int(k)) for k in ks.tolist())
        self.N[p] = n
        return n >= self.minutil


# ============================================================== SMSE
def smse(
    tx,
    w,
    O,
    S,
    xi=None,
    minutil=None,
    measure="wus",
    deadline_s=7200.0,
    argmin="heap",
    trace=None,
    on_done=None,
    xi_ns=None,
):
    """[SMSE] Algorithm 3 on wus. O: all FWUP (incl. S)."""
    t0w, t0c = time.perf_counter(), time.process_time()
    deadline = time.time() + deadline_s if deadline_s else None
    S = [tuple(sorted(p)) for p in S]
    Sset = set(S)
    pats = sorted({tuple(sorted(p)) for p in O} | Sset)
    pid = {p: k for k, p in enumerate(pats)}
    tb = Tables(
        tx,
        w,
        pats,
        [p in Sset for p in pats],
        measure=measure,
        xi=xi,
        minutil=minutil,
        track_removal=True,
        nshc=True,
        xi_ns=xi_ns,
    )
    t_build = time.perf_counter() - t0w
    c_build = time.process_time() - t0c
    order = sorted(S, key=lambda X: (-len(X), X))  # l. 2 (+ tie: item IDs)
    stats = {
        "n_steps": 0,
        "n_delete": 0,
        "n_decrease": 0,
        "n_resync": 0,
        "censored": False,
        "n_done": 0,
        "unfinished": [],
        "xlog": [],
    }
    cur = {"heap": None, "p": None}

    def hook(rows):
        h = cur["heap"]
        if h is None:
            return
        for r in rows.tolist():
            if tb.alive_in_row(cur["p"], r):
                heapq.heappush(h, (int(tb.NSHC[r]), int(tb.sensT[r]), r))

    tb.nshc_hook = hook if argmin == "heap" else None
    try:
        for idx, X in enumerate(order):
            p = pid[X]
            xl = {"X": list(X), "wus_turn": tb.wus_now(p), "steps0": stats["n_steps"]}
            stats["xlog"].append(xl)
            if tb.removed[p] or tb.below_turn(p):  # l. 4 (hidden as a side effect)
                xl["action"] = "skip_removed" if tb.removed[p] else "skip_below"
                stats["n_done"] += 1
                continue
            xl["action"] = "processed"
            rows = tb.rows_of(p)
            if argmin == "heap":
                H = [(int(tb.NSHC[r]), int(tb.sensT[r]), int(r)) for r in rows.tolist()]
                heapq.heapify(H)
                cur["heap"], cur["p"] = H, p
            while True:
                while not tb.removed[p] and not tb.below_turn(p):  # l. 5
                    if deadline is not None and time.time() >= deadline:
                        raise Timeout(idx)
                    r = _pick_T(tb, p, cur["heap"], argmin)  # l. 6-7
                    k = int(tb.sensT[r])
                    a, b = tb.row_slice(r)
                    ids = tb.rpat[a:b]
                    act = tb.ralive[a:b] & ~tb.removed[ids]
                    s_ids, ns_ids = ids[act & tb.sens[ids]], ids[act & ~tb.sens[ids]]
                    shc = {i: int(tb.has[i][s_ids].sum()) for i in X}  # l. 8
                    best = max(shc.values())
                    cvi = [i for i in X if shc[i] == best]
                    if len(cvi) == 1:  # l. 9-10
                        I = cvi[0]
                    else:  # l. 12 (+ tie: item ID)
                        I = min(cvi, key=lambda i: (int(tb.has[i][ns_ids].sum()), i))
                    q = tb.q_star(p, r, I)  # l. 14-17
                    iu = tb.tx[k][I]
                    deleted = tb.apply(r, I, min(q, iu))
                    stats["n_steps"] += 1
                    stats["n_delete" if deleted else "n_decrease"] += 1
                    if trace is not None:
                        trace.append(
                            {
                                "X": X,
                                "T": k,
                                "I": I,
                                "q": min(q, iu),
                                "delete": deleted,
                                "N_X": float(tb.N[p]),
                            }
                        )
                    if argmin == "heap" and not deleted:  # push back (iu decreased)
                        heapq.heappush(cur["heap"], (int(tb.NSHC[r]), k, r))
                if not tb.exact_resync(p):  # l. 15 exact check
                    break
                stats["n_resync"] += 1  # float drift: continue
                tb.removed[p] = False
            stats["n_done"] += 1
            cur["heap"] = None
            xl["steps"] = stats["n_steps"] - xl.pop("steps0")
            xl["wus_done"] = tb.wus_now(p)
            if on_done is not None:
                on_done(X, tb.tx)
    except Timeout as e:
        stats["censored"] = True
        stats["unfinished"] = [list(X) for X in order[e.args[0] :]]
    t1w, t1c = time.perf_counter(), time.process_time()
    stats.update(
        rt_build=t_build,
        rt_total=t1w - t0w,
        rt_hide=t1w - t0w - t_build,
        cpu_total=t1c - t0c,
        cpu_build=c_build,
        cpu_hide=t1c - t0c - c_build,
        order=[list(X) for X in order],
        red_delete=tb.red_delete,
        red_decrease=tb.red_decrease,
        n_d_increase=tb.n_d_increase,
        n_heap_rebuild=tb.n_heap_rebuild,
        n_guard_qstar=tb.n_guard_qstar,
        n_guard_qstar_skipped=tb.n_guard_qstar_skipped,
        n_guard_skip=tb.n_guard_skip,
        n_guard_flip=tb.n_guard_flip,
        rs_delete=tb.rs_delete,
        rs_decrease=tb.rs_decrease,
        n_ev_delete=tb.n_ev_delete,
        n_ev_decrease=tb.n_ev_decrease,
    )
    for xl in stats["xlog"]:
        xl.pop("steps0", None)
    return tb.tx, stats


def _pick_T(tb, p, H, argmin):
    """argmin_{T ⊇ X} (NSHC(T), tid) -- lazy heap or linear scan."""
    if argmin == "heap":
        while H:
            key, tid, r = heapq.heappop(H)
            if key != tb.NSHC[r] or not tb.alive_in_row(p, r):
                continue
            return r
        raise RuntimeError("SMSE: X above threshold but no supporting transaction")
    rows = tb.rows_of(p)
    if rows.size == 0:
        raise RuntimeError("SMSE: X above threshold but no supporting transaction")
    j = np.lexsort((tb.sensT[rows], tb.NSHC[rows]))[0]
    return int(rows[j])


# ============================================================== MSU-MIU
def msumiu(
    tx, w, S, xi=None, minutil=None, measure="wus", deadline_s=7200.0, trace=None, on_done=None
):
    """[MSU-MIU] Algorithm 2 on wus. MSU-MIU does not use NS."""
    t0w, t0c = time.perf_counter(), time.process_time()
    deadline = time.time() + deadline_s if deadline_s else None
    S = sorted({tuple(sorted(p)) for p in S})
    pid = {p: k for k, p in enumerate(S)}
    tb = Tables(
        tx,
        w,
        S,
        [True] * len(S),
        measure=measure,
        xi=xi,
        minutil=minutil,
        track_removal=False,
        nshc=False,
    )
    t_build = time.perf_counter() - t0w
    c_build = time.process_time() - t0c
    f = {}
    for X in S:
        for i in X:
            f[i] = f.get(i, 0) + 1
    wX = {X: sum(f[i] for i in X) for X in S}
    order = sorted(S, key=lambda X: (-wX[X], -len(X), X))  # l. 2-5
    stats = {
        "n_steps": 0,
        "n_delete": 0,
        "n_decrease": 0,
        "n_resync": 0,
        "censored": False,
        "n_done": 0,
        "unfinished": [],
        "xlog": [],
    }
    try:
        for idx, X in enumerate(order):
            p = pid[X]
            xl = {"X": list(X), "wus_turn": tb.wus_now(p), "steps0": stats["n_steps"]}
            stats["xlog"].append(xl)
            if tb.below_turn(p):  # l. 6-8
                xl["action"] = "skip_below"
                stats["n_done"] += 1
                continue
            xl["action"] = "processed"
            rows = tb.rows_of(p)  # l. 10 projection (once)
            ks = tb.sensT[rows]
            uXT = np.array([tb.u_pattern_T(p, int(k)) for k in ks.tolist()], dtype=np.float64)
            L = rows[np.lexsort((ks, -uXT))].tolist()  # l. 11
            pos = 0
            while True:
                while pos < len(L):  # l. 12-13
                    if deadline is not None and time.time() >= deadline:
                        raise Timeout(idx)
                    r = L[pos]
                    if not tb.alive_in_row(p, r):
                        pos += 1
                        continue
                    k = int(tb.sensT[r])
                    t = tb.tx[k]
                    I = min(X, key=lambda i: (w[i] * t[i], i))  # l. 14
                    q = tb.q_star(p, r, I)
                    stats["n_steps"] += 1
                    if q >= t[I]:  # l. 15-20 delete
                        tb.apply(r, I, t[I])
                        stats["n_delete"] += 1
                        if trace is not None:
                            trace.append({"X": X, "T": k, "I": I, "q": None, "delete": True})
                        pos += 1
                        if tb.below_turn(p):
                            break
                    else:  # l. 21-27 decrease, stop X
                        tb.apply(r, I, q)
                        stats["n_decrease"] += 1
                        if trace is not None:
                            trace.append({"X": X, "T": k, "I": I, "q": q, "delete": False})
                        break
                if not tb.exact_resync(p):  # assert exact wus < xi
                    break
                stats["n_resync"] += 1
                if pos >= len(L):
                    raise RuntimeError("MSU-MIU: projection exhausted but X not hidden")
            stats["n_done"] += 1
            xl["steps"] = stats["n_steps"] - xl.pop("steps0")
            xl["wus_done"] = tb.wus_now(p)
            if on_done is not None:
                on_done(X, tb.tx)
    except Timeout as e:
        stats["censored"] = True
        stats["unfinished"] = [list(X) for X in order[e.args[0] :]]
    t1w, t1c = time.perf_counter(), time.process_time()
    stats.update(
        rt_build=t_build,
        rt_total=t1w - t0w,
        rt_hide=t1w - t0w - t_build,
        cpu_total=t1c - t0c,
        cpu_build=c_build,
        cpu_hide=t1c - t0c - c_build,
        order=[list(X) for X in order],
        red_delete=tb.red_delete,
        red_decrease=tb.red_decrease,
        n_guard_qstar=tb.n_guard_qstar,
        n_guard_qstar_skipped=tb.n_guard_qstar_skipped,
        n_guard_skip=tb.n_guard_skip,
        n_guard_flip=tb.n_guard_flip,
        rs_delete=tb.rs_delete,
        rs_decrease=tb.rs_decrease,
        n_ev_delete=tb.n_ev_delete,
        n_ev_decrease=tb.n_ev_decrease,
    )
    for xl in stats["xlog"]:
        xl.pop("steps0", None)
    return tb.tx, stats
