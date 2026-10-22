"""Tables and figures of the paper, from the published results (no experiment is run).

Reads results/<experiment>/summary.csv, the analysis files in results/ (re-exposure, RISWU victim
agreement, workload statistics) and data/datasets/ (dataset statistics only), and writes
  results/paper/tables/<T_name>.csv     one long-format CSV per table
  results/paper/figures/<F_name>.{pdf,eps,png}   vector PDF / EPS and 1000 dpi PNG
  results/paper/figures/fig1_algorithm.dot       Graphviz source of Fig. 1
The content of every table is described in results/paper/README_tables.md.

Rules for every table: feasible = 1 iff HF = 0 (at xi; equal margin: at xi_t) and the run is not
censored (7 200 s budget). Infeasible runs keep their values and are marked † in the column
`mark`. best_<metric> = 1 marks the best value among the feasible runs of the same cell (lower is
better for HF, AC, MC, TMR, DDI and times; higher for IUS and DUS); tie_<metric> = 1 when more
than one feasible run of the cell reaches that best value.

  python scripts/paper_assets.py [--runs] [--tables-only | --figures-only]
  --runs  read runs/<experiment>/summary.csv (your re-run) instead of results/
"""

import collections
import csv
import gzip
import json
import math
import os
import statistics
import sys

import numpy as np
from scipy import stats

import common as V

OUT = os.path.join(V.RESULTS, "paper")
TAB = os.path.join(OUT, "tables")
FIG = os.path.join(OUT, "figures")
DATA = os.path.join(V.REPO, "data", "datasets")

OURS = ["SC-only", "SDIF", "Max-RISWU", "Min-RISWU"]
BASE = ["SMSE", "MSU-MIU"]
METHODS = OURS + BASE
ABL = [
    "greedy-batch",
    "greedy-unit",
    "SWM without removal",
    "Max-RISWU without twu",
    "Min-RISWU without twu",
    "Max-RISWU without Wt",
    "Min-RISWU without Wt",
    "SC-only, ordered by wus",
]
MORDER = METHODS + ABL
NAME = {
    "chess": "Chess",
    "mushroom": "Mushroom",
    "accident": "Accident",
    "retail": "Retail",
    "kosarak": "Kosarak",
    "bms-pos": "BMS-POS",
    "chainstore": "Chainstore",
}
FILES = {
    "chess": "chess_fimi_quantities.txt.gz",
    "mushroom": "mushroom_quantities.txt.gz",
    "accident": "accident_quantities.txt.gz",
    "retail": "retail_quantities.txt.gz",
    "kosarak": "kosarak_quantities.txt.gz",
    "bms-pos": "bms-pos_quantities.txt.gz",
    "chainstore": "chainstore_quantities.txt.gz",
}
DIST = {"uniform": "uniform", "normal": "normal", "lognormal10": "log-normal"}
SELECTION = {"ovhigh": "high overlap", "ovlow": "low overlap", "random": "random"}
LOWER = {
    "hf": True,
    "hf_target": True,
    "ac": True,
    "mc": True,
    "ius": False,
    "dus": False,
    "tmr": True,
    "ddi": True,
    "sanitization_time_s": True,
    "hiding_loop_time_s": True,
    "cpu_time_s": True,
}
Q7 = ["hf", "ac", "mc", "ius", "dus", "tmr", "ddi"]


# ------------------------------------------------------------------ unit accessors
def san_time(u):
    """Sanitization time (Phase 1 + hiding loop, wall clock). Some early baseline runs only
    have the older column name."""
    return u["rt_wall_total"] if u.get("rt_wall_total") is not None else u.get("rt_total")


def hide_time(u):
    return u["rt_wall_hide"] if u.get("rt_wall_hide") is not None else u.get("rt_hide")


def hf_target(u):
    """HF at xi_t; early baseline runs (xi_t = xi) only have HF at xi."""
    if u.get("hf_target") is not None:
        return u["hf_target"]
    if u.get("xi_target") not in (None, u["xi"]):
        raise RuntimeError(f"no HF at xi_t for {u['key']}")
    return u["hf"]


def is_feasible(u):
    f = hf_target(u) == 0 and not u.get("censored")
    if bool(f) != bool(u["feasible"]):
        raise RuntimeError(f"feasible flag differs from HF / censored for {u['key']}")
    return f


def label(u):
    """Method name as used in the paper."""
    v = u.get("variant") or ""
    if "greedy-batch" in v:
        return "greedy-batch"
    if "greedy-unit" in v:
        return "greedy-unit"
    if u.get("regular_step") == "cap":
        return "SWM without removal"
    if v == "no_twu":
        return f"{u['method']} without twu"
    if v == "no_wt":
        return f"{u['method']} without Wt"
    if v == "wusorder":
        return "SC-only, ordered by wus"
    return u["method"]


def base_row(u):
    return {
        "method": label(u),
        "n_S": u["n_S"],
        "feasible": int(is_feasible(u)),
        "censored": int(bool(u.get("censored"))),
        "mark": "" if is_feasible(u) else "†",
    }


def quality(u, keys=Q7):
    return {k: u.get(k) for k in keys}


def mean(v):
    v = [x for x in v if x is not None]
    return statistics.fmean(v) if v else None


def mmm(prefix, vals):
    v = [x for x in vals if x is not None]
    if not v:
        return {f"{prefix}_mean": None, f"{prefix}_min": None, f"{prefix}_max": None}
    return {f"{prefix}_mean": statistics.fmean(v), f"{prefix}_min": min(v), f"{prefix}_max": max(v)}


def mark_best(rows, metrics):
    """best_<m> / tie_<m> among the feasible rows of one cell."""
    F = [r for r in rows if r["feasible"]]
    for k in metrics:
        vals = [r[k] for r in F if r.get(k) is not None]
        best = (min(vals) if LOWER[k] else max(vals)) if vals else None
        n_best = sum(1 for r in F if best is not None and r.get(k) == best)
        for r in rows:
            hit = r["feasible"] and best is not None and r.get(k) == best
            r[f"best_{k}"] = int(hit)
            r[f"tie_{k}"] = int(hit and n_best > 1)
    return rows


def morder(name):
    return MORDER.index(name) if name in MORDER else len(MORDER)


def ds_key(ds):
    return V.DS.index(ds) if ds in V.DS else 99


# ------------------------------------------------------------------ writing
def fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return repr(round(float(v), 6))
    return str(v)


def write_csv(name, rows, cols=None):
    os.makedirs(TAB, exist_ok=True)
    if cols is None:
        cols = []
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
    with open(os.path.join(TAB, f"{name}.csv"), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([fmt(r.get(c)) for c in cols])
    print(f"  {name}.csv  ({len(rows)} rows)")


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------ T_datasets, T_main_config
def scan_dataset(ds):
    """#transactions, #distinct items, average / maximum length (items per transaction after
    parsing: a repeated item keeps its last quantity) and #transactions with a repeated item."""
    n = tot = mx = dup = 0
    items = set()
    with gzip.open(os.path.join(DATA, FILES[ds]), "rt", encoding="utf-8") as f:
        for line in f:
            ids = [int(p.split(":")[0]) for p in line.split()]
            s = set(ids)
            n += 1
            tot += len(s)
            mx = max(mx, len(s))
            dup += len(s) != len(ids)
            items |= s
    return {"n_trans": n, "n_items": len(items), "sum_len": tot, "max_len": mx, "n_dup": dup}


def t_datasets():
    stats_ = {r["dataset"]: r for r in read_csv(os.path.join(V.TABLES, "dataset_stats.csv"))}
    rows = []
    for ds in V.DS:
        s = scan_dataset(ds)
        avg = s["sum_len"] / s["n_trans"]
        dens = 100 * avg / s["n_items"]
        ref = stats_[ds]
        if int(ref["n_tx"]) != s["n_trans"] or int(ref["n_items"]) != s["n_items"]:
            raise RuntimeError(f"{ds}: data/ and dataset_stats.csv disagree")
        rows.append(
            {
                "dataset": NAME[ds],
                "n_trans": s["n_trans"],
                "n_items": s["n_items"],
                "avg_len": f"{avg:.2f}",
                "max_len": s["max_len"],
                "density_pct": f"{dens:.3f}",
                "type": "dense" if ds in V.DENSE else "sparse",
                "avg_len_unrounded": avg,
                "density_pct_unrounded": dens,
                "n_trans_with_repeated_item": s["n_dup"],
            }
        )
    write_csv("T_datasets", rows)


def t_main_config():
    rows = []
    for r in read_csv(os.path.join(V.TABLES, "dataset_stats.csv")):
        rows.append(
            {
                "dataset": NAME[r["dataset"]],
                "xi": float(r["xi"]),
                "n_fwup": int(r["n_fwup"]),
                "n_candidates": int(r["n_candidates"]),
                "n_S": int(r["n_S"]),
                "mean_len_fwup": float(r["avg_len_fwup"]),
                "mean_len_S": float(r["avg_len_sfwup"]),
                "trans_supporting_S_pct": 100 * float(r["share_tx_with_sfwup"]),
            }
        )
    write_csv("T_main_config", rows)


# ------------------------------------------------------------------ T_design
def role(u):
    lab = label(u)
    if lab in BASE:
        return "baselines"
    if lab.startswith("greedy"):
        return "greedy"
    if lab == "SWM without removal":
        return "without_removal"
    return "ours"


def t_design():
    D = [
        (
            "main comparison",
            "main_comparison",
            None,
            "all 7",
            "none (main threshold xi)",
            "main selection",
            "1",
        ),
        (
            "ablation: RISWU variants",
            "riswu_variants",
            None,
            "all 7",
            "RISWU weight W = Wt (without twu) or W = twu (without Wt); Max-RISWU, Min-RISWU",
            "main selection",
            "1",
        ),
        (
            "ablation: wus ordering",
            "ordering_control",
            None,
            "all 7",
            "SC-only victim, S ordered by wus(X) instead of RISWU",
            "main selection",
            "1",
        ),
        (
            "ablation: greedy modification",
            "reduction_mechanism",
            None,
            "all 7",
            "greedy-batch, greedy-unit (victim SDIF)",
            "main selection",
            "1",
        ),
        (
            "ablation: SWM without removal",
            "cap_ablation",
            None,
            "Chess, Mushroom, Retail, Chainstore",
            "regular SWM step never deletes the victim",
            "main selection",
            "1",
        ),
        (
            "threshold",
            "threshold",
            None,
            "Chess, Retail, Kosarak, BMS-POS",
            "xi multiplier (4 per dataset)",
            "re-selected at each xi / fixed (main S still frequent)",
            "1",
        ),
        (
            "workload",
            "workload",
            None,
            "Chess, Mushroom, Retail, Kosarak",
            "selection (high overlap, low overlap, random) x proportion (10, 25, 50 %)",
            "by selection rule; random: 5 seeds",
            "5 seeds (random)",
        ),
        (
            "utility distribution",
            "utility_distribution",
            None,
            "Chess, Kosarak",
            "external utilities: uniform, normal, log-normal",
            "re-selected on each table / fixed (main S)",
            "5 draws",
        ),
        (
            "scalability",
            "scalability",
            ("x6b",),
            "Kosarak, Chainstore",
            "sample size 10, 20, ..., 100 % (nested samples)",
            "main rule on each sample",
            "3 orders",
        ),
        (
            "scalability, early run (4 sizes)",
            "scalability",
            ("x6",),
            "Kosarak, Chainstore",
            "sample size 10, 25, 50, 100 %",
            "main rule on each sample",
            "1 order",
        ),
        (
            "lower threshold",
            None,
            None,
            "all 7 (feasible Q')",
            "attacker threshold (1 - delta) xi, delta = 5, 10, 20 %",
            "as in the source experiment",
            "-",
        ),
        (
            "equal margin",
            "equal_margin",
            None,
            "all 7",
            "every method hides below xi_t = (1 - delta) xi, delta = 5, 10, 20 %",
            "main selection",
            "1",
        ),
    ]
    rows = []
    tot = collections.Counter()
    per_base = collections.Counter()
    for name, exp, plans, dss, factor, sel, seeds in D:
        U = V.units(exp, plans) if exp else []
        c = collections.Counter(role(u) for u in U)
        m = collections.Counter(label(u) for u in U)
        for u in U:
            if role(u) == "baselines":
                per_base[(label(u), exp == "main_comparison")] += 1
        tot.update(c)
        per = sorted({n for n in m.values()}) if m else []
        rows.append(
            {
                "experiment": name,
                "datasets": dss,
                "varied_factor": factor,
                "selection_of_S": sel,
                "seeds_or_orders": seeds,
                "runs_per_method": "/".join(str(x) for x in per) if per else "0 (analysis)",
                "runs_by_method": "; ".join(f"{k}: {m[k]}" for k in sorted(m, key=morder)),
                "runs_ours": c["ours"],
                "runs_baselines": c["baselines"],
                "runs_greedy": c["greedy"],
                "runs_without_removal": c["without_removal"],
                "total_runs": len(U),
            }
        )
    ddi_n = len(V.all_units())
    rows.append(
        {
            "experiment": "total",
            "runs_ours": tot["ours"],
            "runs_baselines": tot["baselines"],
            "runs_greedy": tot["greedy"],
            "runs_without_removal": tot["without_removal"],
            "total_runs": sum(tot.values()),
            "runs_by_method": "; ".join(
                f"{b} {'main' if mn else 'other'}: {per_base[(b, mn)]}"
                for b in BASE
                for mn in (True, False)
            ),
        }
    )
    rows.append(
        {
            "experiment": "total used for the DDI correlation (ours + baselines + greedy)",
            "total_runs": ddi_n,
        }
    )
    write_csv("T_design", rows)
    return tot, ddi_n


# ------------------------------------------------------------------ main comparison
def main_units():
    return V.units("main_comparison")


def t_main_quality():
    rows = []
    for ds in V.DS:
        cell = []
        for u in main_units():
            if u["dataset"] != ds:
                continue
            cell.append({"dataset": NAME[ds], **base_row(u), **quality(u)})
        cell.sort(key=lambda r: morder(r["method"]))
        rows += mark_best(cell, Q7)
    cols = ["dataset", "method", "n_S"] + Q7 + ["feasible", "censored", "mark"]
    cols += [f"best_{k}" for k in Q7] + [f"tie_{k}" for k in Q7]
    write_csv("T_main_quality", rows, cols)

    aux = []
    for ds in V.DS:
        U = sorted(
            (u for u in main_units() if u["dataset"] == ds), key=lambda u: morder(u["method"])
        )
        ours = [u for u in U if u["method"] in OURS]
        for u in U:
            r = {"dataset": NAME[ds], **base_row(u)}
            rs = u.get("removal_share")
            nd, nc = u.get("n_delete"), u.get("n_decrease")
            r.update(
                n_regular_rounds=u.get("n_regular_rounds"),
                pct_pre_hidden=u.get("pct_pre_hidden"),
                removal_share=rs,
                reduction_share_lower_only=None if rs is None else 1 - rs,
                modification_share_lower_only=(
                    (nc / (nd + nc)) if nd is not None and nd + nc else None
                ),
                abs_tmr_minus_ddi=abs(u["tmr"] - u["ddi"]),
                tmr=u["tmr"],
                ddi=u["ddi"],
                mc=u["mc"],
            )
            if u["method"] in OURS:
                lower = sum(1 for o in ours if o["mc"] < u["mc"])
                same = sum(1 for o in ours if o["mc"] == u["mc"]) - 1
                r.update(
                    mc_rank_among_ours=lower + 1,
                    mc_rank_tied_with=same,
                    mc_lowest_strict=int(lower == 0 and same == 0),
                    mc_highest_strict=int(all(o["mc"] < u["mc"] for o in ours if o is not u)),
                )
            aux.append(r)
    write_csv("T_main_aux", aux)


def t_resources():
    rows = []
    for ds in V.DS:
        for u in sorted(
            (u for u in main_units() if u["dataset"] == ds), key=lambda u: morder(u["method"])
        ):
            r = {"dataset": NAME[ds], **base_row(u)}
            r.update(
                sanitization_time_s=san_time(u),
                hiding_loop_time_s=hide_time(u),
                cpu_time_s=u["cpu_total"],
                cpu_per_wall_hiding_loop=u["rt_cpu_hide"] / u["rt_wall_hide"],
                peak_resident_memory_mb=u["peak_rss_hide_mb"],
                additional_memory_mb=u["delta_rss_hide_mb"],
                riswu_share_of_hiding_loop=u.get("t_riswu_share") if u["method"] in OURS else None,
            )
            rows.append(r)
    write_csv("T_resources", rows)


# ------------------------------------------------------------------ ablation
def t_ablation():
    x1 = {(u["dataset"], u["method"]): u for u in main_units()}
    agree = json.load(open(os.path.join(V.RESULTS, "riswu_variants", "victim_agreement.json")))
    rows = []
    q6 = ["ac", "mc", "ius", "dus", "tmr", "ddi"]
    for ds in V.DS:
        for m in OURS:
            u = x1[(ds, m)]
            rows.append(
                {
                    "panel": "a",
                    "row_type": "strategy",
                    "dataset": NAME[ds],
                    **base_row(u),
                    **quality(u, ["mc", "tmr", "ddi", "ius"]),
                    "sanitization_time_s": san_time(u),
                }
            )
        for u in sorted(
            (u for u in V.units("riswu_variants") if u["dataset"] == ds),
            key=lambda u: (u["method"] != "Max-RISWU", u["variant"]),
        ):
            full = x1[(ds, u["method"])]
            d = {k: abs(u[k] - full[k]) for k in q6}
            rows.append(
                {
                    "panel": "a",
                    "row_type": "riswu_variant",
                    "dataset": NAME[ds],
                    **base_row(u),
                    "same_victim_pct": 100
                    * agree[ds]["same_victim_share"][f"{u['method']}|{u['variant']}"],
                    **quality(u, ["mc", "tmr", "ddi", "ius"]),
                    "sanitization_time_s": san_time(u),
                    **{f"abs_delta_{k}": d[k] for k in ("mc", "tmr", "ddi", "ius")},
                    "max_abs_delta": max(d.values()),
                    "identical_quality": int(max(d.values()) == 0),
                }
            )
        for u in V.units("ordering_control"):
            if u["dataset"] != ds:
                continue
            rows.append(
                {
                    "panel": "a",
                    "row_type": "wus_ordering",
                    "dataset": NAME[ds],
                    **base_row(u),
                    **quality(u, ["mc", "tmr", "ddi", "ius"]),
                    "sanitization_time_s": san_time(u),
                    "delta_mc": u["mc"] - x1[(ds, "SC-only")]["mc"],
                }
            )
        a = agree[ds]
        rows.append(
            {
                "panel": "a",
                "row_type": "spearman_twu_wt",
                "dataset": NAME[ds],
                "spearman_median": a["spearman_median"],
                "spearman_min": a["spearman_min"],
                "spearman_max": a["spearman_max"],
                "n_S": len(a["rows"]),
            }
        )
    for ds in V.DS:
        cell = [x1[(ds, "SDIF")]] + [
            u for u in V.units("reduction_mechanism") if u["dataset"] == ds
        ]
        out = []
        for u in cell:
            r = {
                "panel": "b",
                "row_type": "modification",
                "dataset": NAME[ds],
                **base_row(u),
                **quality(u, ["hf", "mc", "tmr", "ddi", "ius"]),
                "sanitization_time_s": san_time(u),
                "iterations": u.get("n_iter"),
                "removal_share": u.get("removal_share"),
            }
            if r["method"] == "SDIF":
                r["method"] = "SWM (SDIF)"
            out.append(r)
        rows += mark_best(out, ["mc", "tmr", "ddi", "ius", "sanitization_time_s"])
    for u in sorted(
        V.units("cap_ablation"), key=lambda u: (ds_key(u["dataset"]), morder(u["method"]))
    ):
        main = x1[(u["dataset"], u["method"])]
        rows.append(
            {
                "panel": "b",
                "row_type": "without_removal",
                "dataset": NAME[u["dataset"]],
                **base_row(u),
                "strategy": u["method"],
                **quality(u, ["hf", "mc", "tmr", "ddi", "ius"]),
                "sanitization_time_s": san_time(u),
                "time_ratio_to_main": san_time(u) / san_time(main),
                "tier1_calls": u.get("n_tier1"),
                "n_rehide": u.get("n_rehide"),
                "main_mc": main["mc"],
                "main_tmr": main["tmr"],
                "main_ddi": main["ddi"],
            }
        )
    write_csv("T_ablation", rows)


# ------------------------------------------------------------------ threshold
def t_threshold():
    rows = []
    MODE = {"reselect": "re-selected S", "fixed": "fixed S"}
    dss = ["chess", "retail", "kosarak", "bms-pos"]
    for r in sorted(
        read_csv(os.path.join(V.TABLES, "threshold_workload.csv")),
        key=lambda r: (dss.index(r["dataset"]), float(r["mult"]), r["mode"] != "fixed"),
    ):
        rows.append(
            {
                "panel": "task",
                "dataset": NAME[r["dataset"]],
                "multiplier": float(r["mult"]),
                "mode": MODE[r["mode"]],
                "xi": float(r["xi"]),
                "n_fwup": int(r["n_fwup"]),
                "n_nsfwup": int(r["n_nsfwup"]),
                "n_candidates": int(r["n_candidates"]),
                "n_S": int(r["n_S"]),
                "mean_len_fwup": float(r["avg_len_fwup"]),
                "mean_len_S": float(r["avg_len_sfwup"]),
                "trans_supporting_S_pct": 100 * float(r["share_tx_with_sfwup"]),
            }
        )
    cells = collections.defaultdict(list)
    for u in V.units("threshold"):
        cells[(u["dataset"], u["mult"], u["variant"])].append(u)
    keys = ["hf", "mc", "ac", "ius", "dus", "tmr", "ddi"]
    for c in sorted(cells, key=lambda c: (dss.index(c[0]), c[1], c[2] != "fixed")):
        out = []
        for u in sorted(cells[c], key=lambda u: morder(u["method"])):
            out.append(
                {
                    "panel": "results",
                    "dataset": NAME[c[0]],
                    "multiplier": c[1],
                    "mode": MODE[c[2]],
                    "xi": u["xi"],
                    **base_row(u),
                    **quality(u, keys),
                    "sanitization_time_s": san_time(u),
                }
            )
        rows += mark_best(out, keys + ["sanitization_time_s"])
    write_csv("T_threshold", rows)


# ------------------------------------------------------------------ workload
def t_workload():
    groups = collections.defaultdict(list)
    for u in V.units("workload"):
        rho, kind = u["variant"].split("_")
        sel = "random" if kind[0] == "r" and kind[1:].isdigit() else kind
        seed = int(kind[1:]) if sel == "random" else None
        groups[(u["dataset"], sel, rho, seed)].append(u)
    # best MC per cell; a cell is one S (each random seed is its own cell)
    best = collections.Counter()
    tie = collections.Counter()
    ncell = collections.Counter()
    for (ds, sel, rho, seed), L in groups.items():
        R = mark_best(
            [{"m": u["method"], "feasible": is_feasible(u), "mc": u["mc"]} for u in L], ["mc"]
        )
        for r in R:
            ncell[(ds, sel, rho, r["m"])] += 1
            best[(ds, sel, rho, r["m"])] += r["best_mc"]
            tie[(ds, sel, rho, r["m"])] += r["tie_mc"]
    agg = collections.defaultdict(list)
    for (ds, sel, rho, seed), L in groups.items():
        for u in L:
            agg[(ds, sel, rho, u["method"])].append(u)
    dss = ["chess", "mushroom", "retail", "kosarak"]
    sels = ["ovhigh", "ovlow", "random"]
    rows = []
    keys = ["hf", "mc", "ac", "ius", "tmr", "ddi"]
    for k in sorted(agg, key=lambda k: (dss.index(k[0]), sels.index(k[1]), k[2], morder(k[3]))):
        L = agg[k]
        r = {
            "dataset": NAME[k[0]],
            "selection": SELECTION[k[1]],
            "proportion_pct": round(100 * float(k[2][3:])),
            "method": k[3],
            "n_runs": len(L),
            "n_feasible": sum(is_feasible(u) for u in L),
            "n_S_min": min(u["n_S"] for u in L),
            "n_S_max": max(u["n_S"] for u in L),
        }
        r["mark"] = "" if r["n_feasible"] == r["n_runs"] else "†"
        for m in keys:
            r.update(mmm(m, [u[m] for u in L]))
        r.update(mmm("sanitization_time_s", [san_time(u) for u in L]))
        r.update(mmm("removal_share", [u.get("removal_share") for u in L]))
        r.update(best_mc_cells=best[k], tie_mc_cells=tie[k], n_cells=ncell[k])
        rows.append(r)
    write_csv("T_workload", rows)


# ------------------------------------------------------------------ utility distribution
def utility_units():
    x5 = {
        (u["dataset"], u["calib"].split("_", 2)[2], u["method"]): u
        for u in V.units("utility_distribution", ("x5",))
    }
    x5b = {
        (u["calib"], u["method"], u["variant"]): u
        for u in V.units("utility_distribution", ("x5b",))
    }

    def unit(ds, dist, sd, mode, m):
        if mode == "reselect" and sd == 0 and dist in ("normal", "lognormal10") and m in OURS:
            return x5.get((ds, dist, m))
        return x5b.get((f"x5b_{ds}_{dist}_s{sd}", m, mode))

    out = {}
    for ds in ("chess", "kosarak"):
        for dist in DIST:
            for mode in ("reselect", "x1fixed"):
                for m in METHODS:
                    out[(ds, dist, mode, m)] = [unit(ds, dist, sd, mode, m) for sd in range(5)]
    return out


def t_utility():
    UU = utility_units()
    MODE = {"reselect": "S re-selected", "x1fixed": "S fixed"}
    keys = ["hf", "mc", "ac", "ius", "tmr", "ddi"]
    rows = []
    x1 = {(u["dataset"], u["method"]): u for u in main_units()}
    for ds in ("chess", "kosarak"):
        for m in METHODS:
            u = x1[(ds, m)]
            r = {
                "dataset": NAME[ds],
                "distribution": "original utilities",
                "mode": "main comparison",
                "method": m,
                "n_runs": 1,
                "n_feasible": int(is_feasible(u)),
                "mark": "" if is_feasible(u) else "†",
                "n_S_min": u["n_S"],
                "n_S_max": u["n_S"],
            }
            for k in keys:
                r.update(mmm(k, [u[k]]))
            r.update(mmm("sanitization_time_s", [san_time(u)]))
            r.update(mmm("removal_share", [u.get("removal_share")]))
            rows.append(r)
        for dist in DIST:
            for mode in MODE:
                for m in METHODS:
                    L = UU[(ds, dist, mode, m)]
                    if any(u is None for u in L):
                        raise RuntimeError(f"missing utility unit {ds} {dist} {mode} {m}")
                    nf = sum(is_feasible(u) for u in L)
                    r = {
                        "dataset": NAME[ds],
                        "distribution": DIST[dist],
                        "mode": MODE[mode],
                        "method": m,
                        "n_runs": len(L),
                        "n_feasible": nf,
                        "mark": "" if nf == len(L) else "†",
                        "n_S_min": min(u["n_S"] for u in L),
                        "n_S_max": max(u["n_S"] for u in L),
                    }
                    for k in keys:
                        r.update(mmm(k, [u[k] for u in L]))
                    r.update(mmm("sanitization_time_s", [san_time(u) for u in L]))
                    r.update(mmm("removal_share", [u.get("removal_share") for u in L]))
                    rows.append(r)
    write_csv("T_utility", rows)


# ------------------------------------------------------------------ scalability
def scal_units():
    U = V.units("scalability", ("x6b",))
    for u in U:
        c = u["calib"].split("_")  # <plan>_<ds>_p<order>_<pct>
        u["_order"], u["_pct"] = int(c[2][1:]), int(c[3])
    return U


def scal_workload():
    wl = read_csv(os.path.join(V.TABLES, "scalability_workload.csv"))
    return {(r["dataset"], int(r["perm"]), int(r["pct"])): r for r in wl if r["mode"] == "S"}


SCAL_MEASURES = [
    ("sanitization_time_s", san_time),
    ("cpu_time_s", lambda u: u["cpu_total"]),
    ("hiding_loop_time_s", hide_time),
    ("peak_memory_mb", lambda u: u["peak_rss_hide_mb"]),
    ("additional_memory_mb", lambda u: u["delta_rss_hide_mb"]),
]


def loglog(x, y):
    """OLS of log10 y on log10 x over the pairs with y > 0 -> slope, 95 % CI (t), R², n."""
    p = [(a, b) for a, b in zip(x, y) if b is not None and b > 0]
    if len(p) < 3:
        return None
    lx = np.log10([a for a, _ in p])
    ly = np.log10([b for _, b in p])
    res = stats.linregress(lx, ly)
    t = stats.t.ppf(0.975, len(p) - 2)
    return {
        "slope": res.slope,
        "ci95_low": res.slope - t * res.stderr,
        "ci95_high": res.slope + t * res.stderr,
        "r2": res.rvalue**2,
        "n": len(p),
    }


def t_scalability():
    U = scal_units()
    wl = scal_workload()
    rows = []
    for ds in ("kosarak", "chainstore"):
        for pc in range(10, 101, 10):
            W = [wl[(ds, o, pc)] for o in range(3)]
            for m in METHODS:
                L = [u for u in U if u["dataset"] == ds and u["method"] == m and u["_pct"] == pc]
                nf = sum(is_feasible(u) for u in L)
                r = {
                    "dataset": NAME[ds],
                    "size_pct": pc,
                    "method": m,
                    "n_runs": len(L),
                    "n_feasible": nf,
                    "mark": "" if nf == len(L) else "†",
                    "Q_size": int(W[0]["n_tx"]) if len({w["n_tx"] for w in W}) == 1 else None,
                }
                r.update(
                    mmm(
                        "trans_supporting_S_pct", [100 * float(w["share_tx_with_sfwup"]) for w in W]
                    )
                )
                for name, f in SCAL_MEASURES:
                    r.update(mmm(name, [f(u) for u in L]))
                rows.append(r)
    write_csv("T_scalability", rows)

    srows = []
    for m in METHODS:
        for ds in ("kosarak", "chainstore"):
            L = [u for u in U if u["dataset"] == ds and u["method"] == m]
            for name, f in SCAL_MEASURES:
                for scope in ("all runs", "feasible runs"):
                    S = L if scope == "all runs" else [u for u in L if is_feasible(u)]
                    x = [int(wl[(ds, u["_order"], u["_pct"])]["n_tx"]) for u in S]
                    fit = loglog(x, [f(u) for u in S])
                    srows.append(
                        {
                            "method": m,
                            "dataset": NAME[ds],
                            "measure": name,
                            "scope": scope,
                            "n_runs_in_scope": len(S),
                            **(fit or {"n": 0}),
                        }
                    )
    write_csv(
        "T_scalability_slopes",
        srows,
        [
            "method",
            "dataset",
            "measure",
            "scope",
            "slope",
            "ci95_low",
            "ci95_high",
            "r2",
            "n",
            "n_runs_in_scope",
        ],
    )

    erows = []
    for u in sorted(
        V.units("scalability", ("x6",)),
        key=lambda u: (
            ds_key(u["dataset"]),
            int(u["calib"].rsplit("_", 1)[1]),
            morder(u["method"]),
        ),
    ):
        erows.append(
            {
                "dataset": NAME[u["dataset"]],
                "size_pct": int(u["calib"].rsplit("_", 1)[1]),
                **base_row(u),
                **quality(u),
                "sanitization_time_s": san_time(u),
                "hiding_loop_time_s": hide_time(u),
                "cpu_time_s": u.get("cpu_total"),
                "peak_memory_mb": u.get("peak_rss_hide_mb"),
                "additional_memory_mb": u.get("delta_rss_hide_mb"),
            }
        )
    write_csv("T_scalability_early", erows)


# ------------------------------------------------------------------ attack
def reexposure():
    with open(os.path.join(V.SOURCE["dir"], "threshold_lowering", "reexposure.jsonl")) as f:
        return [json.loads(line) for line in f]


DELTAS = ("0.05", "0.10", "0.20")


def t_attack():
    R = reexposure()
    by = {
        (u["_exp"], u["key"]): u
        for e in ("main_comparison", "cap_ablation", "equal_margin")
        for u in V.units(e)
    }
    rows = []
    groups = [
        ("our method (4 strategies)", "main_comparison", lambda u: u["method"] in OURS),
        *[(m, "main_comparison", (lambda m: lambda u: u["method"] == m)(m)) for m in OURS],
        ("SMSE", "main_comparison", lambda u: u["method"] == "SMSE"),
        ("MSU-MIU", "main_comparison", lambda u: u["method"] == "MSU-MIU"),
        ("SWM without removal", "cap_ablation", lambda u: True),
    ]
    for name, exp, keep in groups:
        G = [r for r in R if r["experiment"] == exp and keep(by[(exp, r["key"])])]
        feas = [u for u in V.units(exp) if is_feasible(u) and keep(u)]
        if {r["key"] for r in G} != {u["key"] for u in feas}:
            raise RuntimeError(f"re-exposure file does not cover the feasible Q' of {name}")
        med = [statistics.median(r["wus_ratio_S"]) for r in G]
        for d in DELTAS:
            r_ = [r["deltas"][d]["reexposed"] for r in G]
            rows.append(
                {
                    "panel": "a",
                    "method": name,
                    "delta_pct": round(100 * float(d)),
                    "n": len(G),
                    "r_mean": mean(r_),
                    "r_min": min(r_),
                    "r_max": max(r_),
                    "m_mean": mean([r["deltas"][d]["n_new"] for r in G]),
                    "q_mean": mean([r["deltas"][d]["sfwup_in_new"] for r in G]),
                    "median_wus_over_xi_min": min(med),
                    "median_wus_over_xi_max": max(med),
                }
            )
    r20 = {
        r["key"]: r["deltas"]["0.20"]["reexposed"] for r in R if r["experiment"] == "equal_margin"
    }
    cells = collections.defaultdict(list)
    for u in V.units("equal_margin"):
        cells[(u["dataset"], u["delta"])].append(u)
    keys = ["hf_target", "hf", "mc", "ac", "ius", "dus", "tmr", "ddi"]
    em_order = OURS + BASE + ["greedy-batch", "greedy-unit"]
    for c in sorted(cells, key=lambda c: (ds_key(c[0]), c[1])):
        out = []
        for u in sorted(cells[c], key=lambda u: em_order.index(label(u))):
            out.append(
                {
                    "panel": "b",
                    "dataset": NAME[c[0]],
                    "delta_pct": round(100 * c[1]),
                    **base_row(u),
                    **quality(u, keys),
                    "sanitization_time_s": san_time(u),
                    "r_at_delta20": r20.get(u["key"]),
                }
            )
        rows += mark_best(out, ["mc", "ac", "ius", "dus", "tmr", "ddi", "sanitization_time_s"])
    # counts: best of ours (lowest MC over the 4 strategies) vs each comparator, cells where both
    # are feasible
    for comp in ("MSU-MIU", "greedy-batch", "SMSE", "greedy-unit"):
        n = k = 0
        for c, L in cells.items():
            ours = [
                u["mc"] for u in L if u["method"] in OURS and label(u) in OURS and is_feasible(u)
            ]
            other = [u for u in L if label(u) == comp and is_feasible(u)]
            if ours and other:
                n += 1
                k += min(ours) <= other[0]["mc"]
        rows.append(
            {
                "panel": "b_counts",
                "method": f"best of ours <= {comp}",
                "n_cells_both_feasible": n,
                "n_cells_true": k,
            }
        )
    sha = {}
    for e in ("main_comparison", "equal_margin"):
        for r in read_csv(os.path.join(V.SOURCE["dir"], e, "qprime_sha256.csv")):
            sha[(e, r["key"])] = r["qprime_sha256"]
    x1 = {(u["dataset"], u["method"]): u for u in main_units()}
    em_ours = [u for u in V.units("equal_margin") if label(u) in OURS]
    same = sum(
        sha[("equal_margin", u["key"])]
        == sha[("main_comparison", x1[(u["dataset"], u["method"])]["key"])]
        for u in em_ours
    )
    rows.append(
        {
            "panel": "b_counts",
            "method": "our runs with Q' identical to the main comparison",
            "n_cells_both_feasible": len(em_ours),
            "n_cells_true": same,
        }
    )
    same_q = sum(
        all(u[k] == x1[(u["dataset"], u["method"])][k] for k in ("mc", "ac")) and is_feasible(u)
        for u in em_ours
    )
    rows.append(
        {
            "panel": "b_counts",
            "method": "our runs with MC and AC identical to the main comparison",
            "n_cells_both_feasible": len(em_ours),
            "n_cells_true": same_q,
        }
    )
    write_csv("T_attack", rows)


# ------------------------------------------------------------------ DDI correlation
def t_ddi():
    U = V.all_units()
    rows = []
    for name, L in (
        ("all runs", U),
        ("our method", [u for u in U if role(u) == "ours"]),
        ("baselines", [u for u in U if role(u) == "baselines"]),
    ):
        d = [u["ddi"] for u in L]
        r = {"group": name, "n": len(L)}
        for k in ("mc", "ius", "tmr"):
            r[f"spearman_ddi_{k}"] = float(stats.spearmanr(d, [u[k] for u in L]).statistic)
        rows.append(r)
    write_csv("T_ddi", rows)


TABLES = [
    t_datasets,
    t_main_config,
    t_design,
    t_main_quality,
    t_resources,
    t_ablation,
    t_threshold,
    t_workload,
    t_utility,
    t_scalability,
    t_attack,
    t_ddi,
]


# ================================================================== figures
def figures():
    import paper_figures

    paper_figures.main(FIG)


def main(argv):
    if "--runs" in argv:
        V.use_runs_dir()
    if "--figures-only" not in argv:
        print("tables ->", os.path.relpath(TAB, V.REPO))
        for fn in TABLES:
            fn()
    if "--tables-only" not in argv:
        figures()


if __name__ == "__main__":
    main(sys.argv[1:])
