"""Phase 1 of Algorithm 1 (lines 1-4): SC(i), NSC(i) of items, SC(T), NSC(T) of transactions and
Wt(T) = SC(T) / (NSC(T) + 1).

Memory-lean: each pattern's tidset is built once as a boolean mask (AND of per-item masks; only
items occurring in some pattern get a mask), added to the per-transaction counter and dropped.
Peak extra memory = (#items in FWUP + 2) x |Q| bytes plus the counters.
"""

from collections import defaultdict

import numpy as np


def compute(tx, S, O):
    """tx: DB; S: SFWUP tuples; O: all FWUP tuples (S ⊆ O).
    -> dict(sc_item, nsc_item, sc_trans, nsc_trans, wt); *_trans and wt are numpy arrays
    indexed by transaction index."""
    n = len(tx)
    S_set = {tuple(sorted(p)) for p in S}
    NS = sorted({tuple(sorted(p)) for p in O} - S_set)
    S_list = sorted(S_set)
    items = sorted({i for p in S_list for i in p} | {i for p in NS for i in p})
    item_pos = {i: k for k, i in enumerate(items)}
    masks = np.zeros((len(items), n), dtype=bool)
    for k, t in enumerate(tx):
        for i in t:
            r = item_pos.get(i)
            if r is not None:
                masks[r, k] = True

    def count(patterns, cnt, item_cnt):
        m = np.empty(n, dtype=bool)
        for p in patterns:
            for i in p:
                item_cnt[i] += 1
            np.copyto(m, masks[item_pos[p[0]]])
            for i in p[1:]:
                np.logical_and(m, masks[item_pos[i]], out=m)
            cnt += m

    sc_item, nsc_item = defaultdict(int), defaultdict(int)
    sc_trans = np.zeros(n, dtype=np.int64)
    nsc_trans = np.zeros(n, dtype=np.int64)
    count(S_list, sc_trans, sc_item)
    count(NS, nsc_trans, nsc_item)
    wt = sc_trans / (nsc_trans + 1)
    return {
        "sc_item": dict(sc_item),
        "nsc_item": dict(nsc_item),
        "sc_trans": sc_trans,
        "nsc_trans": nsc_trans,
        "wt": wt,
    }
