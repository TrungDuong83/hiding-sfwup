"""Shared helpers of the table / figure scripts: read the per-unit result rows of every experiment
from results/<experiment>/summary.csv (or from a re-run in runs/<experiment>/summary.csv)."""

import csv
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results")
TABLES = os.path.join(RESULTS, "tables")
FIGURES = os.path.join(RESULTS, "figures")
DS = ["chess", "mushroom", "accident", "retail", "kosarak", "bms-pos", "chainstore"]
DENSE = {"chess", "mushroom", "accident"}
PROP = ("SDIF", "Max-RISWU", "Min-RISWU", "SC-only")
BASE = ("SMSE", "MSU-MIU")
METHODS = PROP + BASE
# experiments whose units enter the pooled analyses (the cap variant is reported separately)
SPEC_EXPS = (
    "main_comparison",
    "reduction_mechanism",
    "riswu_variants",
    "ordering_control",
    "threshold",
    "workload",
    "utility_distribution",
    "scalability",
    "equal_margin",
)
_INT = re.compile(r"-?\d+$")


def _value(s):
    if s == "":
        return None
    if s in ("True", "False"):
        return s == "True"
    if _INT.match(s):
        return int(s)
    try:
        return float(s)
    except ValueError:
        return s


def load_rows(path, exp):
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["status"] != "OK":
                continue
            u = {k: _value(v) for k, v in r.items()}
            for k in ("key", "variant", "calib", "method", "dataset", "workload_id", "plan"):
                u[k] = r[k]  # identifiers stay strings
            # equal-margin variants are named d<percent>[_mechanism]
            m = re.match(r"d(\d\d)", u["variant"]) if u["plan"] == "x7b" else None
            u["delta"] = int(m.group(1)) / 100 if m else None
            u["_exp"] = exp
            out.append(u)
    return out


_CACHE = {}
SOURCE = {"dir": RESULTS}


def units(exp, plans=None):
    """Result rows of one experiment (optionally only the given plans)."""
    if exp not in _CACHE:
        _CACHE[exp] = load_rows(os.path.join(SOURCE["dir"], exp, "summary.csv"), exp)
    rows = _CACHE[exp]
    return [u for u in rows if plans is None or u["plan"] in plans]


def all_units(exps=SPEC_EXPS):
    return [u for e in exps for u in units(e)]


def family(u):
    v = u.get("variant", "") or ""
    if "greedy-unit" in v:
        return "greedy-unit"
    if "greedy-batch" in v:
        return "greedy-batch"
    return "proposal" if u["method"] in PROP else u["method"]


def method_label(u):
    f = family(u)
    return f if f.startswith("greedy") else u["method"]


def use_runs_dir():
    """Read runs/<experiment>/summary.csv (a re-run) instead of results/."""
    SOURCE["dir"] = os.path.join(REPO, "runs")
    _CACHE.clear()
