"""SFWUP selection and the workload files (FWUP(Q, xi) with wus, S, xi) of every experiment.

  pool  = {X in FWUP : |X| >= 2, wus(X) > xi}
  ov(X) = #{Y in FWUP : Y != X, Y ∩ X != ∅}
  rank  = ov descending, ties -> item-ID tuple ascending
  |S|   = min(max(10, floor(0.10 |pool|)), 40); |pool| < 10 -> the whole pool

Workload file data/workloads/<id>.json.gz: dataset, mult, xi, weights_file, n_transactions, D,
fwup ([pattern, wus] pairs), n_fwup, pool (ranked [pattern, ov] pairs), S (ranked),
sum_u_def3 (IUS denominator), fwup_sha256, mine_s (timing), plus experiment-specific fields.
The SHA-256 of the uncompressed JSON text is checked against data/checksums.txt on reading.
Files shipped with the repository also hold an `xc` record: the result of a cross-check of the
FWUP set against an independent miner (not part of this package).

Regenerate (overwrites the files and their checksum entries; `mine_s` differs between runs):
  env PYTHONHASHSEED=0 python -m hiding_sfwup.workloads <main_comparison | threshold | workload |
      utility_distribution | scalability>
"""

import gzip
import hashlib
import json
import os
import sys
import time

import numpy as np

from . import data as DA
from . import miner as M

WORKLOAD_DIR = os.path.join(DA.DATA_DIR, "workloads")
MINER_LABEL = "hiding_sfwup.miner (no length cap)"
X1_DS = ["chess", "mushroom", "accident", "retail", "kosarak", "bms-pos", "chainstore"]


def pool_of(fwup, xi):
    return sorted(p for p, w in fwup.items() if len(p) >= 2 and w > xi)


def ov_scores(fwup, targets):
    """ov(X) for X in targets over the full FWUP set; packed bitsets over patterns."""
    pats = sorted(fwup)
    P = len(pats)
    items = sorted({i for p in pats for i in p})
    rows = {i: np.zeros(P, dtype=bool) for i in items}
    for k, p in enumerate(pats):
        for i in p:
            rows[i][k] = True
    packed = {i: np.packbits(r) for i, r in rows.items()}
    del rows
    pop = np.array([bin(x).count("1") for x in range(256)], dtype=np.int64)
    out = {}
    for X in targets:
        acc = packed[X[0]].copy()
        for i in X[1:]:
            np.bitwise_or(acc, packed[i], out=acc)
        out[X] = int(pop[acc].sum()) - 1  # minus X itself (X ∈ FWUP and X ∩ X ≠ ∅)
    return out


def rank_by_ov(pool, ov, reverse=True):
    if reverse:
        return sorted(pool, key=lambda X: (-ov[X], X))
    return sorted(pool, key=lambda X: (ov[X], X))


def n_select(npool):
    if npool < 10:
        return npool
    return min(max(10, int(0.10 * npool)), 40)


def select_main(fwup, xi):
    pool = pool_of(fwup, xi)
    ov = ov_scores(fwup, pool)
    ranked = rank_by_ov(pool, ov)
    return ranked[: n_select(len(pool))], ranked, ov


def sum_u_def3(tx, w, patterns):
    """sum_Y u(Y), u(Y) = sum_{T ⊇ Y} sum_{i in Y} u(i, T) (Definition 3)."""
    iidx = DA.item_index(tx)
    tot = 0.0
    for p in patterns:
        ts = DA.tidset(p, iidx)
        tot += float(sum(w[i] * tx[k][i] for k in ts.tolist() for i in p))
    return tot


# ------------------------------------------------------------------ file I/O
def _gzip_bytes(raw):
    """Deterministic gzip (no file name, mtime 0), the same as `gzip -n -9`."""
    import io

    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=buf, mtime=0) as g:
        g.write(raw)
    return buf.getvalue()


def update_checksum(path, raw, stored):
    """Set the data/checksums.txt entry of `path` (content SHA-256, stored-file SHA-256)."""
    rel = os.path.relpath(os.path.abspath(path), DA.REPO)
    line = f"{hashlib.sha256(raw).hexdigest()}  {hashlib.sha256(stored).hexdigest()}  {rel}\n"
    with open(DA.CHECKSUMS) as f:
        lines = f.readlines()
    out = [x for x in lines if x.startswith("#") or not x.strip() or x.split()[2] != rel]
    head = [x for x in out if x.startswith("#")]
    body = sorted(
        [x for x in out if not x.startswith("#") and x.strip()] + [line], key=lambda x: x.split()[2]
    )
    with open(DA.CHECKSUMS, "w") as f:
        f.writelines(head + body)
    DA._SUMS = None


def write_workload(name, rec):
    os.makedirs(WORKLOAD_DIR, exist_ok=True)
    path = os.path.join(WORKLOAD_DIR, f"{name}.json.gz")
    txt = json.dumps(rec, indent=1, sort_keys=True)
    raw = txt.encode()
    stored = _gzip_bytes(raw)
    with open(path, "wb") as f:
        f.write(stored)
    update_checksum(path, raw, stored)
    return path, hashlib.sha256(raw).hexdigest()


def read_workload(name):
    path = os.path.join(WORKLOAD_DIR, f"{name}.json.gz")
    raw = DA.read_verified(path)
    rec = json.loads(raw.decode())
    rec["_sha256"] = hashlib.sha256(raw).hexdigest()
    rec["_path"] = path
    return rec


def fwup_of(rec):
    return {tuple(p): w for p, w in rec["fwup"]}


def calibrate_cell(ds, mult, tx=None, w=None, weights_file=None, selector=None, extra=None):
    """Mine FWUP(Q, xi) (no length cap), pool, ov, S. Returns the record (not written)."""
    xi = DA.xi_of(ds, mult)
    if tx is None:
        tx, w = DA.load_db(ds, weights_file)
    t0 = time.perf_counter()
    res, m = M.mine(tx, w, xi, collect_u=True)
    mine_s = time.perf_counter() - t0
    if selector is None:
        S, ranked, ov = select_main(res, xi)
    else:
        S, ranked, ov = selector(res, xi)
    rec = {
        "dataset": ds,
        "mult": float(mult),
        "xi": xi,
        "weights_file": (
            os.path.relpath(weights_file, DA.REPO) if weights_file else DA.DATASETS[ds][1]
        ),
        "n_transactions": len(tx),
        "D": m.D,
        "fwup": [[list(p), res[p]] for p in sorted(res)],
        "n_fwup": len(res),
        "fwup_sha256": M.fwup_hash(res),
        "max_len": max((len(p) for p in res), default=0),
        "pool": [[list(X), ov[X]] for X in ranked],
        "n_pool": len(ranked),
        "S": [list(X) for X in S],
        "n_S": len(S),
        "sum_u_def3": float(sum(m.u.values())),
        "borderline": [[list(b[0]), b[1], b[2], bool(b[3])] for b in m.borderline],
        "mine_s": mine_s,
        "miner": MINER_LABEL,
    }
    if extra:
        rec.update(extra)
    return rec, res


# ------------------------------------------------------------------ main comparison
def main_comparison(log=print):
    out = []
    for ds in X1_DS:
        name = f"x1_{ds}"
        log(f"[workload] {name}: mining ...")
        rec, res = calibrate_cell(ds, 1.0)
        path, h = write_workload(name, rec)
        log(
            f"[workload] {name}: |FWUP|={rec['n_fwup']} pool={rec['n_pool']} |S|={rec['n_S']} "
            f"mine={rec['mine_s']:.2f}s sha256={h[:16]}"
        )
        out.append(rec)
    return out


# ------------------------------------------------------------------ threshold sensitivity
X3_GRID = {
    "chess": [0.7, 0.8, 0.9, 1.1],
    "retail": [0.6, 0.8, 1.2, 1.4],
    "kosarak": [0.8, 1.2, 1.4, 1.6],
    "bms-pos": [0.6, 0.8, 1.2, 1.4],
}


def threshold(log=print, only=None):
    """Per cell: S re-selected at that xi (field S). Fixed mode: S_fixed = main selection at the
    largest xi of grid ∪ {1.0}, verified to be FWUP at every xi of the grid."""
    for ds, grid in X3_GRID.items():
        if only and ds not in only:
            continue
        tx, w = DA.load_db(ds)
        mmax = max(grid + [1.0])
        recs = {}
        for m in grid:
            log(f"[workload] x3_{ds}_{m}: mining ...")
            recs[m] = calibrate_cell(ds, m, tx, w)
        if mmax == 1.0:
            base = read_workload(f"x1_{ds}")
            S_fixed = [tuple(x) for x in base["S"]]
        else:
            S_fixed = [tuple(x) for x in recs[mmax][0]["S"]]
        for m in grid:
            rec, res = recs[m]
            missing = [list(X) for X in S_fixed if X not in res]
            if missing:
                raise RuntimeError(f"threshold {ds} {m}: S_fixed patterns not FWUP: {missing}")
            rec["S_fixed"] = [list(X) for X in S_fixed]
            rec["S_fixed_from"] = f"main selection at xi x{mmax}"
            name = f"x3_{ds}_{m}"
            path, h = write_workload(name, rec)
            log(
                f"[workload] {name}: xi={rec['xi']:.6g} |FWUP|={rec['n_fwup']} "
                f"pool={rec['n_pool']} |S|={rec['n_S']} |S_fixed|={len(S_fixed)} "
                f"mine={rec['mine_s']:.2f}s sha256={h[:16]}"
            )


# ------------------------------------------------------------------ workload (SFWUP choice x size)
X4_DS = ["chess", "mushroom", "kosarak", "retail"]
X4_RHO = [0.10, 0.25, 0.50]


def x4_sets(rec):
    """|S| = max(5, floor(rho*pool)); one ranking (ov desc, IDs asc); ov-high = its first |S|,
    ov-low = its last |S| (disjoint when 2|S| <= pool); random r0..r4 =
    numpy default_rng(seed).choice(pool sorted by item IDs, |S|, replace=False)."""
    pool = [tuple(X) for X, _ in rec["pool"]]
    ov = {tuple(X): o for X, o in rec["pool"]}
    by_id = sorted(pool)
    hi = rank_by_ov(pool, ov, reverse=True)
    lo = hi[::-1]
    out = {}
    for rho in X4_RHO:
        k = max(5, int(rho * len(pool)))
        out[f"rho{rho:.2f}_ovhigh"] = [list(X) for X in hi[:k]]
        out[f"rho{rho:.2f}_ovlow"] = [list(X) for X in lo[:k]]
        for seed in range(5):
            idx = np.random.default_rng(seed).choice(len(by_id), size=k, replace=False)
            out[f"rho{rho:.2f}_r{seed}"] = [list(by_id[i]) for i in sorted(idx.tolist())]
    return out


def workload(log=print):
    for ds in X4_DS:
        base = read_workload(f"x1_{ds}")
        sets = x4_sets(base)
        rec = {
            "dataset": ds,
            "base_calib": f"x1_{ds}",
            "base_sha256": base["_sha256"],
            "n_pool": base["n_pool"],
            "sets": sets,
            "sizes": {k: len(v) for k, v in sets.items()},
            "rule": "|S| = max(5, floor(rho*pool)); ovhigh/ovlow = first/last |S| of the A3 "
            "ranking (ov desc, IDs asc); r<seed> = numpy.random.default_rng(seed)"
            ".choice(pool sorted by item IDs, |S|, replace=False)",
        }
        path, h = write_workload(f"x4_{ds}", rec)
        log(
            f"[workload] x4_{ds}: pool={base['n_pool']} sizes rho .10/.25/.50 = "
            f"{[rec['sizes'][f'rho{r:.2f}_ovhigh'] for r in X4_RHO]} sha256={h[:16]}"
        )


# ------------------------------------------------------------------ utility distribution
X5_DS = ["chess", "kosarak"]
X5_DISTS = ("normal", "lognormal10")
X5B_DISTS = ("uniform", "normal", "lognormal10")
X5B_SEEDS = (0, 1, 2, 3, 4)


def make_weights(ds, dist, seed=0):
    """normal: rint(N(5.5, 2.0^2)) clipped to [1, 10]; lognormal10: rint(lognormal(mu=1.0,
    sigma=0.8)) clipped to [1, 10]; uniform: integers on {1, ..., 10}. One draw per item, items in
    ascending ID order, numpy default_rng(seed)."""
    w = DA.load_weights(os.path.join(DA.UTILITY_DIR, DA.DATASETS[ds][1]))
    items = sorted(w)
    rng = np.random.default_rng(seed)
    if dist == "normal":
        v = np.clip(np.rint(rng.normal(5.5, 2.0, size=len(items))), 1, 10)
    elif dist == "lognormal10":
        v = np.clip(np.rint(rng.lognormal(1.0, 0.8, size=len(items))), 1, 10)
    elif dist == "uniform":
        v = rng.integers(1, 11, size=len(items)).astype(np.float64)
    else:
        raise ValueError(dist)
    os.makedirs(DA.VARIANT_DIR, exist_ok=True)
    path = os.path.join(DA.VARIANT_DIR, f"{ds}_{dist}_seed{seed}.txt")
    raw = "".join(f"{i}:{int(x)}\n" for i, x in zip(items, v.tolist())).encode()
    with open(path, "wb") as f:
        f.write(raw)
    update_checksum(path, raw, raw)
    return path, v


def _x5_cells(log, dists):
    for ds in X5_DS:
        for dist in dists:
            path, v = make_weights(ds, dist)
            log(
                f"[workload] x5_{ds}_{dist}: weights mean={v.mean():.3f} sd={v.std():.3f} "
                f"min={v.min():.0f} max={v.max():.0f}"
            )
            rec, res = calibrate_cell(
                ds,
                1.0,
                weights_file=path,
                extra={
                    "weights_dist": dist,
                    "weights_seed": 0,
                    "weights_stats": {
                        "mean": float(v.mean()),
                        "sd": float(v.std()),
                        "min": float(v.min()),
                        "max": float(v.max()),
                    },
                },
            )
            name = f"x5_{ds}_{dist}"
            _, h = write_workload(name, rec)
            log(
                f"[workload] {name}: |FWUP|={rec['n_fwup']} pool={rec['n_pool']} |S|={rec['n_S']} "
                f"sha256={h[:16]}"
            )


def _x5b_cells(log, only=None):
    """Weight tables {uniform, normal, lognormal10} x seeds 0-4 for chess, kosarak.
    Re-selected mode: S chosen on each table (field S). Fixed mode: S of the main comparison
    minus the X that are no longer FWUP under the new table (field S_x1fixed)."""
    for ds in X5_DS:
        if only and ds not in only:
            continue
        x1 = read_workload(f"x1_{ds}")
        S1 = [tuple(x) for x in x1["S"]]
        tx = DA.load_db(ds)[0]
        for dist in X5B_DISTS:
            for sd in X5B_SEEDS:
                path = os.path.join(DA.VARIANT_DIR, f"{ds}_{dist}_seed{sd}.txt")
                old = open(path).read() if os.path.exists(path) else None
                path, v = make_weights(ds, dist, sd)
                if old is not None and open(path).read() != old:
                    raise RuntimeError(f"regenerated weights differ from the existing file {path}")
                w = DA.load_weights(path)
                freq = {str(k): int(c) for k, c in zip(*np.unique(v, return_counts=True))}
                rec, res = calibrate_cell(
                    ds,
                    1.0,
                    tx=tx,
                    w=w,
                    weights_file=path,
                    extra={
                        "weights_dist": dist,
                        "weights_seed": sd,
                        "weights_stats": {
                            "mean": float(v.mean()),
                            "sd": float(v.std()),
                            "min": float(v.min()),
                            "max": float(v.max()),
                        },
                        "weights_freq": freq,
                    },
                )
                keep = [X for X in S1 if X in res]
                rec["S_x1fixed"] = [list(X) for X in keep]
                rec["n_S_x1fixed"] = len(keep)
                rec["S_x1_dropped"] = [[list(X), None] for X in S1 if X not in res]
                rec["n_S_x1_dropped"] = len(S1) - len(keep)
                rec["S_x1fixed_rule"] = (
                    "S of x1_<ds> minus the X with wus < xi under this "
                    "weight table; cell skipped if < 5"
                )
                if dist in ("normal", "lognormal10") and sd == 0:
                    o = read_workload(f"x5_{ds}_{dist}")
                    rec["same_as_x5_seed0"] = bool(
                        o["fwup_sha256"] == rec["fwup_sha256"] and o["S"] == rec["S"]
                    )
                name = f"x5b_{ds}_{dist}_s{sd}"
                _, h = write_workload(name, rec)
                log(
                    f"[workload] {name}: eu {freq} |FWUP|={rec['n_fwup']} pool={rec['n_pool']} "
                    f"|S|={rec['n_S']} |S_x1fixed|={len(keep)} (dropped {rec['n_S_x1_dropped']}) "
                    f"mine={rec['mine_s']:.1f}s sha256={h[:16]}"
                )


def utility_distribution(log=print, only=None):
    _x5_cells(log, X5_DISTS)
    _x5b_cells(log, only)


# ------------------------------------------------------------------ scalability
X6_DS = ["kosarak", "chainstore"]
X6_FRAC = [0.10, 0.25, 0.50, 1.00]
X6B_DS = ["kosarak", "chainstore"]
X6B_PCT = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
X6B_PERMS = [0, 1, 2]


def sample_indices(n, frac, seed=0):
    """Nested samples: prefix of one permutation (numpy default_rng(seed)), restored to the
    original transaction order."""
    perm = np.random.default_rng(seed).permutation(n)
    return np.sort(perm[: int(frac * n)])


def _sample_record(ds, xi, idx, seed, frac, res, info, ranked, ov, S, pool_rule, mine_s):
    return {
        "dataset": ds,
        "mult": 1.0,
        "xi": xi,
        "weights_file": DA.DATASETS[ds][1],
        "sample": {
            "seed": seed,
            "frac": frac,
            "n": int(idx.size),
            "indices_sha256": hashlib.sha256(idx.astype(np.int64).tobytes()).hexdigest(),
        },
        "n_transactions": int(idx.size),
        "D": info["D"],
        "fwup": [[list(p), res[p]] for p in sorted(res)],
        "n_fwup": len(res),
        "fwup_sha256": M.fwup_hash(res),
        "max_len": max((len(p) for p in res), default=0),
        "pool": [[list(X), ov[X]] for X in ranked],
        "n_pool": len(ranked),
        "pool_rule": pool_rule,
        "S": [list(X) for X in S],
        "n_S": len(S),
        "sum_u_def3": info["sum_u"],
        "borderline": info["borderline"],
        "mine_s": mine_s,
        "miner": MINER_LABEL,
    }


def _x6_cells(log):
    for ds in X6_DS:
        tx, w = DA.load_db(ds)
        xi = DA.XI[ds]
        cells = {}
        for fr in X6_FRAC:
            idx = sample_indices(len(tx), fr)
            sub = [tx[k] for k in idx.tolist()]
            log(f"[workload] x6_{ds}_{fr}: |Q|={len(sub)} mining ...")
            t0 = time.perf_counter()
            res, m = M.mine(sub, w, xi, collect_u=True)
            cells[fr] = (idx, res, m, time.perf_counter() - t0)
        full = cells[1.00][1]
        pool = [X for X in pool_of(full, xi) if all(X in cells[fr][1] for fr in X6_FRAC)]
        ov = ov_scores(full, pool)
        ranked = rank_by_ov(pool, ov)
        S = ranked[: n_select(len(pool))]
        for fr in X6_FRAC:
            idx, res, m, ms = cells[fr]
            info = {
                "D": m.D,
                "sum_u": float(sum(m.u.values())),
                "borderline": [[list(b[0]), b[1], b[2], bool(b[3])] for b in m.borderline],
            }
            rec = _sample_record(
                ds,
                xi,
                idx,
                0,
                fr,
                res,
                info,
                ranked,
                ov,
                S,
                "pool(100%) restricted to FWUP of all 4 samples; " "ov on the 100% FWUP set",
                ms,
            )
            name = f"x6_{ds}_{int(fr * 100)}"
            _, h = write_workload(name, rec)
            log(
                f"[workload] {name}: |FWUP|={rec['n_fwup']} pool(all 4)={len(pool)} |S|={len(S)} "
                f"mine={ms:.2f}s sha256={h[:16]}"
            )


def _x6b_cells(log, only=None):
    """Nested samples pct% (frac = pct/100) of permutations default_rng(0), (1), (2); xi of the
    main comparison. Candidates = pool(100 %) restricted to the X that are FWUP in every sample
    of every permutation; ov on the 100 % FWUP set; S = top |S| (as above). Also records
    whether S of the main comparison is FWUP in every sample (S_x1, S_x1_valid_all)."""
    for ds in X6B_DS:
        if only and ds not in only:
            continue
        tx, w = DA.load_db(ds)
        xi = DA.XI[ds]
        x1 = read_workload(f"x1_{ds}")
        S1 = [tuple(x) for x in x1["S"]]
        cells = {}
        full = None
        for pm in X6B_PERMS:
            for pc in X6B_PCT:
                fr = pc / 100
                idx = sample_indices(len(tx), fr, pm)
                if pc == 100 and full is not None:
                    cells[(pm, pc)] = (idx,) + full
                    continue
                sub = [tx[k] for k in idx.tolist()] if pc < 100 else tx
                t0 = time.perf_counter()
                res, m = M.mine(sub, w, xi, collect_u=True)
                info = {
                    "D": m.D,
                    "sum_u": float(sum(m.u.values())),
                    "borderline": [[list(b[0]), b[1], b[2], bool(b[3])] for b in m.borderline],
                }
                del m
                cells[(pm, pc)] = (idx, res, info, time.perf_counter() - t0)
                if pc == 100:
                    full = cells[(pm, pc)][1:]
                log(
                    f"[workload] x6b {ds} p{pm} {pc}%: |Q|={len(sub)} |FWUP|={len(res)} "
                    f"mine={cells[(pm, pc)][3]:.1f}s"
                )
        fw_full = full[0]
        pool = [X for X in pool_of(fw_full, xi) if all(X in c[1] for c in cells.values())]
        ov = ov_scores(fw_full, pool)
        ranked = rank_by_ov(pool, ov)
        S = ranked[: n_select(len(pool))]
        inval = [
            [list(X), pm, pc] for X in S1 for (pm, pc), c in sorted(cells.items()) if X not in c[1]
        ]
        old = read_workload(f"x6_{ds}_100")
        for (pm, pc), (idx, res, info, ms) in sorted(cells.items()):
            rec = _sample_record(
                ds,
                xi,
                idx,
                pm,
                pc / 100,
                res,
                info,
                ranked,
                ov,
                S,
                "pool(100%) restricted to FWUP of all 10 samples x 3 "
                "permutations; ov on the 100% FWUP set",
                ms,
            )
            rec.update(
                {
                    "S_equals_x6": [list(X) for X in S] == old["S"],
                    "S_x1": [list(X) for X in S1],
                    "S_x1_valid_all": not inval,
                    "S_x1_invalid": inval,
                }
            )
            write_workload(f"x6b_{ds}_p{pm}_{pc}", rec)
        log(
            f"[workload] x6b {ds}: candidates={len(pool)} |S|={len(S)} "
            f"S_x1 valid in all 30 samples={not inval} (invalid pairs {len(inval)})"
        )


def scalability(log=print, only=None):
    _x6_cells(log)
    _x6b_cells(log, only)


COMMANDS = {
    "main_comparison": main_comparison,
    "threshold": threshold,
    "workload": workload,
    "utility_distribution": utility_distribution,
    "scalability": scalability,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        raise SystemExit(f"usage: python -m hiding_sfwup.workloads {{{' | '.join(COMMANDS)}}}")
    COMMANDS[sys.argv[1]]()
