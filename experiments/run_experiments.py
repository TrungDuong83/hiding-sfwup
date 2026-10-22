"""Run one experiment of the paper (all its units, or a subset), resumably.

  env PYTHONHASHSEED=0 python experiments/run_experiments.py <experiment> [--out DIR]
      [--only SUBSTR[,SUBSTR...]] [--keys KEY[,KEY...]] [--list]

<experiment> is the name of a file in experiments/configs/ (type "runs"). Results go to
runs/<experiment>/ (units/*.json, qprime/*.pkl.gz, summary.csv); compare them with the published
results using scripts/compare_results.py.
  --only  keep the units whose key contains one of the substrings
  --keys  keep exactly these unit keys (plan order)
  --list  print the unit keys and exit
"""

import json
import os
import sys

if os.environ.get("PYTHONHASHSEED") != "0":
    raise SystemExit(
        "PYTHONHASHSEED must be 0: run with  env PYTHONHASHSEED=0 python "
        "experiments/run_experiments.py ..."
    )

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from hiding_sfwup import runner as R  # noqa: E402
from hiding_sfwup import workloads as W  # noqa: E402

CONFIG_DIR = os.path.join(REPO, "experiments", "configs")
# seeds that are fixed in the code; the configuration files must state the same values
SEEDS = {
    "workload": {"random_S": list(range(5))},
    "utility_distribution": {"weights": list(W.X5B_SEEDS)},
    "scalability": {"permutation": list(W.X6B_PERMS)},
}


def load_config(name):
    path = os.path.join(CONFIG_DIR, f"{name}.json")
    if not os.path.exists(path):
        names = sorted(f[:-5] for f in os.listdir(CONFIG_DIR) if f.endswith(".json"))
        raise SystemExit(f"unknown experiment {name!r}; one of: {', '.join(names)}")
    cfg = json.load(open(path))
    if cfg["type"] != "runs":
        raise SystemExit(f"{name} is an analysis of other experiments; run: {cfg['run']}")
    if cfg["plans"] != R.EXPERIMENTS[name]:
        raise SystemExit(f"{path}: plans {cfg['plans']} != {R.EXPERIMENTS[name]}")
    if cfg["budget_s"] != R.BASELINE_BUDGET_S:
        raise SystemExit(f"{path}: budget_s must be {R.BASELINE_BUDGET_S} (the published setting)")
    for key, seeds in SEEDS.get(name, {}).items():
        if f"seed = {seeds[0]}..{seeds[-1]}" not in cfg["seeds"].get(key, ""):
            raise SystemExit(f"{path}: seeds[{key!r}] does not match the code ({seeds})")
    return cfg


def main(argv):
    if not argv or argv[0].startswith("-"):
        raise SystemExit(__doc__)
    name = argv[0]
    cfg = load_config(name)
    units = [u for p in cfg["plans"] for u in R.plan_units(p)]
    if len(units) != cfg["n_units"]:
        raise SystemExit(f"{name}: {len(units)} units, configuration says {cfg['n_units']}")
    if "--only" in argv:
        pats = argv[argv.index("--only") + 1].split(",")
        units = [u for u in units if any(p in R.unit_key(u) for p in pats)]
    if "--keys" in argv:
        ks = set(argv[argv.index("--keys") + 1].split(","))
        units = [u for u in units if R.unit_key(u) in ks]
    if "--list" in argv:
        print("\n".join(R.unit_key(u) for u in units))
        return 0
    out = argv[argv.index("--out") + 1] if "--out" in argv else os.path.join(R.RUNS_DIR, name)
    return R.drive(units, out)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
