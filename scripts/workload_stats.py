"""Dataset and workload statistics (independent of the hiding method), from the DBs and the
workload files: |D|, avg |T|, #distinct items, density = avg |T| / #items, xi, |FWUP|,
|NSFWUP| = |FWUP \\ S|, #candidates (pool), |S|, average length of FWUP and of SFWUP, share of
transactions containing >= 1 SFWUP, sum of |T_X| over X in S.

  env PYTHONHASHSEED=0 python scripts/workload_stats.py [datasets threshold scalability utility]
-> results/tables/{dataset_stats, threshold_workload, scalability_workload,
                  utility_tables}.{csv,json}
"""

import csv
import json
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hiding_sfwup import data as DA  # noqa: E402
from hiding_sfwup import workloads as W  # noqa: E402

OUT = os.path.join(DA.REPO, "results", "tables")


def db_stats(tx):
    lens = [len(t) for t in tx]
    items = {i for t in tx for i in t}
    return {
        "n_tx": len(tx),
        "avg_len": statistics.fmean(lens),
        "n_items": len(items),
        "density": statistics.fmean(lens) / len(items),
    }


def s_cover(tx, S):
    iidx = DA.item_index(tx)
    cover = np.zeros(len(tx), dtype=bool)
    tot = 0
    for X in S:
        ts = DA.tidset(X, iidx)
        cover[ts] = True
        tot += int(ts.size)
    return float(cover.mean()), tot


def cell(rec, tx, sset="S", mode=None):
    S = [tuple(x) for x in rec[sset]]
    fw = [tuple(p) for p, _ in rec["fwup"]]
    share, tot = s_cover(tx, S)
    Ss = set(S)
    return {
        "dataset": rec["dataset"],
        "mult": rec["mult"],
        "xi": rec["xi"],
        "mode": mode or sset,
        "n_fwup": len(fw),
        "n_nsfwup": len(set(fw) - Ss),
        "n_candidates": rec["n_pool"],
        "n_S": len(S),
        "avg_len_fwup": statistics.fmean(len(p) for p in fw),
        "avg_len_sfwup": statistics.fmean(len(p) for p in S),
        "max_len_fwup": max(len(p) for p in fw),
        "share_tx_with_sfwup": share,
        "sum_TX": tot,
    }


def write(rows, name):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{name}.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(OUT, f"{name}.json"), "w") as f:
        json.dump(rows, f, indent=1)


def datasets():
    rows = []
    for ds in W.X1_DS:
        tx, _ = DA.load_db(ds)
        rows.append(dict(db_stats(tx), **cell(W.read_workload(f"x1_{ds}"), tx)))
        print(ds, flush=True)
    write(rows, "dataset_stats")


def threshold():
    rows = []
    for ds, grid in W.X3_GRID.items():
        tx, _ = DA.load_db(ds)
        st = db_stats(tx)
        for mm in grid:
            rec = W.read_workload(f"x3_{ds}_{mm}")
            for sset, mode in (("S", "reselect"), ("S_fixed", "fixed")):
                rows.append(dict(st, **cell(rec, tx, sset, mode)))
        print(ds, flush=True)
    write(rows, "threshold_workload")


def scalability():
    rows = []
    for ds in W.X6B_DS:
        full, _ = DA.load_db(ds)
        for pm in W.X6B_PERMS:
            for pc in W.X6B_PCT:
                rec = W.read_workload(f"x6b_{ds}_p{pm}_{pc}")
                idx = W.sample_indices(len(full), pc / 100, pm)
                tx = [full[k] for k in idx.tolist()]
                for sset in (("S", "S_x1") if rec.get("S_x1_valid_all") else ("S",)):
                    rows.append(dict(db_stats(tx), perm=pm, pct=pc, **cell(rec, tx, sset)))
        print(ds, flush=True)
    write(rows, "scalability_workload")


def utility():
    rows = []
    for ds in W.X5_DS:
        tx, _ = DA.load_db(ds)
        st = db_stats(tx)
        for dist in W.X5B_DISTS:
            for sd in W.X5B_SEEDS:
                rec = W.read_workload(f"x5b_{ds}_{dist}_s{sd}")
                for sset in ("S", "S_x1fixed"):
                    if rec.get(sset):
                        rows.append(
                            dict(
                                st,
                                dist=dist,
                                seed=sd,
                                weights_freq=json.dumps(rec.get("weights_freq")),
                                n_S_reselect=rec["n_S"],
                                n_S_x1fixed=rec["n_S_x1fixed"],
                                n_S_x1_dropped=rec.get("n_S_x1_dropped"),
                                **cell(rec, tx, sset),
                            )
                        )
        print(ds, flush=True)
    write(rows, "utility_tables")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["datasets", "threshold", "scalability", "utility"]
    for name in todo:
        {
            "datasets": datasets,
            "threshold": threshold,
            "scalability": scalability,
            "utility": utility,
        }[name]()
