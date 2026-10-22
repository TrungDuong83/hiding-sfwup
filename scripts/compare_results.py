"""Compare a re-run with the published results: every non-RT column of summary.csv and the
SHA-256 of Q' (canonical form) of every unit present in the re-run.

  python scripts/compare_results.py <experiment> [--run DIR]     (default DIR = runs/<experiment>)

Exit status 0 when every compared unit is identical. Columns that depend on the machine or on
the run (time, CPU, memory, threads, provenance) are not compared; the n_guard_* columns are not
in the published files. Some published units were written by an earlier runner that did not yet
record every column (empty cells, e.g. 48 rows of threshold and scalability); empty published
cells are not compared and such units are counted separately.
"""

import csv
import gzip
import hashlib
import json
import os
import pickle
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOLATILE_PREFIX = (
    "rt_",
    "cpu_",
    "eval_s",
    "rss_",
    "peak_rss",
    "delta_rss",
    "t_riswu_share",
    "threads_after",
    "children_after",
    "code_tag",
    "git_hash",
    "n_guard_",
)


def qprime_sha256(path):
    """SHA-256 of Q' in canonical form: the saved diff {tid: {item: iu}} sorted by tid, item."""
    with gzip.open(path, "rb") as f:
        diff = pickle.load(f)
    canon = [
        [int(k), sorted((int(i), int(q)) for i, q in t.items())] for k, t in sorted(diff.items())
    ]
    return hashlib.sha256(json.dumps(canon, separators=(",", ":")).encode()).hexdigest()


def compare(exp, run_dir):
    pub = {
        r["key"]: r for r in csv.DictReader(open(os.path.join(REPO, "results", exp, "summary.csv")))
    }
    sha = {
        r["key"]: r["qprime_sha256"]
        for r in csv.DictReader(open(os.path.join(REPO, "results", exp, "qprime_sha256.csv")))
    }
    new = {r["key"]: r for r in csv.DictReader(open(os.path.join(run_dir, "summary.csv")))}
    bad, partial = {}, 0
    for key, r in sorted(new.items()):
        if key not in pub:
            bad[key] = ["not in the published results"]
            continue
        p = pub[key]
        cols = [c for c in p if not c.startswith(VOLATILE_PREFIX)]
        partial += any(p[c] == "" and r.get(c, "") != "" for c in cols)
        diff = [c for c in cols if p[c] != "" and r.get(c) != p[c]]
        qp = os.path.join(run_dir, "qprime", f"{key}.pkl.gz")
        if not os.path.exists(qp):
            diff.append("qprime (missing)")
        elif qprime_sha256(qp) != sha[key]:
            diff.append("qprime_sha256")
        if diff:
            bad[key] = diff
    return len(new), bad, partial


def main(argv):
    if not argv:
        raise SystemExit(__doc__)
    exp = argv[0]
    run_dir = argv[argv.index("--run") + 1] if "--run" in argv else os.path.join(REPO, "runs", exp)
    n, bad, partial = compare(exp, run_dir)
    for key, d in bad.items():
        print(f"DIFF {key}: {', '.join(d)}")
    print(
        f"{exp}: {n - len(bad)}/{n} units identical (non-RT columns + SHA-256 of Q')"
        + (f"; {partial} published with fewer columns" if partial else "")
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
