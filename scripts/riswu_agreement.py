"""RISWU variants, measured on the original DB of each main-comparison workload (Phase-1 Wt):
  (i)  Spearman rank correlation between twu(T) and Wt(T) over T ⊇ X, per SFWUP (average ranks
       for ties; None when either side is constant);
  (ii) share of SFWUP for which the variant (W = Wt "no_twu", W = twu "no_wt") selects the same
       victim as the full RISWU (W = twu*Wt), for Max-RISWU and Min-RISWU.

  env PYTHONHASHSEED=0 python scripts/riswu_agreement.py
-> results/riswu_variants/victim_agreement.json
"""

import json
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hiding_sfwup import data as DA  # noqa: E402
from hiding_sfwup import hiding as H  # noqa: E402
from hiding_sfwup import phase1 as P1  # noqa: E402
from hiding_sfwup import workloads as W  # noqa: E402


def ranks(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=np.float64)
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman(a, b):
    if len(a) < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return None
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def main():
    out = {}
    for ds in W.X1_DS:
        rec = W.read_workload(f"x1_{ds}")
        tx, w = DA.load_db(ds)
        S = [tuple(x) for x in rec["S"]]
        hs = {}
        for st in ("Max-RISWU", "Min-RISWU"):
            for v in ("full", "no_twu", "no_wt"):
                h = H.Hider(tx, w, rec["xi"], S, W.fwup_of(rec), st, H.HideConfig(riswu_w=v))
                if not hs:
                    h._setup()
                    h.m = P1.compute(h.tx, h.S, h.O)
                    base = h
                else:
                    h.tx, h.isets, h.twu, h.m = base.tx, base.isets, base.twu, base.m
                h.wt = base.m["wt"]
                hs[(st, v)] = h
        rows = []
        for X in S:
            TX = base.tids_of(X)
            rho = spearman(base.twu[TX], base.wt[TX])
            vic = {key: h.select_victim(X, TX) for key, h in hs.items()}
            rows.append(
                {
                    "X": list(X),
                    "TX": int(TX.size),
                    "spearman": rho,
                    "victims": {f"{a}|{b}": int(v) for (a, b), v in vic.items()},
                }
            )
        rhos = [r["spearman"] for r in rows if r["spearman"] is not None]
        same = {}
        for st in ("Max-RISWU", "Min-RISWU"):
            for v in ("no_twu", "no_wt"):
                same[f"{st}|{v}"] = sum(
                    1 for r in rows if r["victims"][f"{st}|{v}"] == r["victims"][f"{st}|full"]
                ) / len(rows)
        out[ds] = {
            "rows": rows,
            "spearman_min": min(rhos) if rhos else None,
            "spearman_median": statistics.median(rhos) if rhos else None,
            "spearman_max": max(rhos) if rhos else None,
            "n_undefined": len(rows) - len(rhos),
            "same_victim_share": same,
        }
        print(ds, flush=True)
    path = os.path.join(DA.REPO, "results", "riswu_variants", "victim_agreement.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
