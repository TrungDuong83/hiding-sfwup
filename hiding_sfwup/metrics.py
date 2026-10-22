"""Evaluation of a sanitized DB Q'.

HF  = #{X in S : wus_Q'(X) >= xi} / |S|               exact, not via the miner
MC  = |NS \\ NS'| / |NS|,  NS = FWUP(Q) \\ S, NS' = FWUP(Q') \\ S
AC  = |FWUP(Q') \\ FWUP(Q)| / |FWUP(Q)|
IUS = sum_{Y in FWUP(Q')} u(Y) / sum_{Y in FWUP(Q)} u(Y), u(Y) per Definition 3
      (= sum_{T ⊇ Y} sum_{i in Y} u(i, T)) on the respective DB; not capped at 1
DUS = min(sum TU(Q') / sum TU(Q), 1);  TMR = #modified T / |Q|;
DDI = #T that lost an item / |Q|
"""

import gzip
import pickle
import time

from . import data as DA
from . import exact as EX
from . import miner as M


def hf_exact(txq, w, xi, S):
    """-> (HF, [(X, wus, wus >= xi)])"""
    iidx = DA.item_index(txq)
    twu = DA.twu_array(txq, w)
    D = float(twu.sum())
    wf = None
    rows = []
    for X in S:
        ts = DA.tidset(X, iidx)
        n = float(twu[ts].sum()) if ts.size else 0.0

        def ex(ts=ts):
            nonlocal wf
            if wf is None:
                wf = EX.exact_weights(w)
            return EX.n_exact(txq, wf, ts), EX.d_exact(txq, wf)

        ge = ts.size > 0 and EX.wus_ge(n, D, xi, ex)
        rows.append((list(X), n / D if D > 0 else 0.0, bool(ge)))
    hf = sum(1 for r in rows if r[2]) / len(S) if S else 0.0
    return hf, rows


def evaluate(tx0, txq, w, xi, S, fwup0, sum_u0):
    t0 = time.perf_counter()
    S = [tuple(sorted(p)) for p in S]
    Sset = set(S)
    hf, hf_rows = hf_exact(txq, w, xi, S)
    fwq, mq = M.mine(txq, w, xi, collect_u=True)
    O = set(fwup0)
    Oq = set(fwq)
    NS = O - Sset
    NSq = Oq - Sset
    mc = len(NS - NSq) / len(NS) if NS else 0.0
    ac = len(Oq - O) / len(O) if O else 0.0
    sum_uq = float(sum(mq.u.values()))
    ius = sum_uq / sum_u0 if sum_u0 > 0 else 1.0
    tu0 = sum(DA.tu_of(t, w) for t in tx0)
    tuq = sum(DA.tu_of(t, w) for t in txq)
    dus = min(tuq / tu0, 1.0) if tu0 > 0 else 1.0
    n = len(tx0)
    tmr = sum(1 for a, b in zip(tx0, txq) if a != b) / n if n else 0.0
    ddi = sum(1 for a, b in zip(tx0, txq) if len(a) > len(b)) / n if n else 0.0
    hf_mined = len(Sset & Oq) / len(S) if S else 0.0  # cross-check: equals the exact HF
    return {
        "hf": hf,
        "mc": mc,
        "ac": ac,
        "ius": ius,
        "dus": dus,
        "tmr": tmr,
        "ddi": ddi,
        "hf_mined": hf_mined,
        "hf_rows": hf_rows,
        "n_fwup_q": len(O),
        "n_fwup_qprime": len(Oq),
        "n_ns": len(NS),
        "n_ns_lost": len(NS - NSq),
        "n_artificial": len(Oq - O),
        "sum_u_q": sum_u0,
        "sum_u_qprime": sum_uq,
        "fwup_qprime_sha256": M.fwup_hash(fwq),
        "eval_s": time.perf_counter() - t0,
        "miner_borderline_qprime": len(mq.borderline),
    }


def qprime_diff(tx0, txq):
    return {k: b for k, (a, b) in enumerate(zip(tx0, txq)) if a != b}


def save_qprime(path, tx0, txq):
    """Q' is stored as the diff {tid: transaction} against Q (gzip + pickle)."""
    with gzip.open(path, "wb") as f:
        pickle.dump(qprime_diff(tx0, txq), f, protocol=4)


def load_qprime(path, tx0):
    with gzip.open(path, "rb") as f:
        diff = pickle.load(f)
    return [diff.get(k, t) for k, t in enumerate(tx0)]
