"""Threshold lowering: re-exposure of the hidden SFWUP when an adversary mines Q' at a lower
threshold. For every feasible unit of the given experiments (re-run with
experiments/run_experiments.py, so that runs/<experiment>/qprime/ holds Q'), Q' is mined once at
xi_min = (1 - 0.20) xi; the sets for delta' in {5, 10, 20 %} and for xi itself are obtained by
filtering that result (|wus - t| < 1e-9 decided exactly). Per unit and delta':
  reexposed    = #{X in S : wus_Q'(X) >= (1 - delta') xi} / |S|   (exact, not via the miner)
  n_new        = |FWUP(Q', xi') \\ FWUP(Q', xi)|
  sfwup_in_new = |S ∩ new| / |new|
and the distribution of wus_Q'(X) / xi over X in S. Equal-margin units (hidden below xi_t) use the
same delta' below the original xi. One child process per unit (900 s, 50 GB).

  env PYTHONHASHSEED=0 python scripts/reexposure.py <experiment> [<experiment> ...]
-> runs/threshold_lowering/reexposure.jsonl (published: results/threshold_lowering/)
"""

import glob
import json
import os
import resource
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
OUT = os.path.join(REPO, "runs", "threshold_lowering")
DELTAS = (0.05, 0.10, 0.20)
LIMIT_S = 900


def child(unit_path, out_path):
    resource.setrlimit(resource.RLIMIT_AS, (50 * 1024**3,) * 2)
    from hiding_sfwup import data as DA
    from hiding_sfwup import exact as EX
    from hiding_sfwup import metrics as ME
    from hiding_sfwup import miner as M
    from hiding_sfwup import workloads as W

    r = json.load(open(unit_path))
    rec = W.read_workload(r["calib"])
    tx, w = DA.load_db(rec["dataset"])
    q = ME.load_qprime(os.path.join(REPO, r["qprime"]), tx)
    xi = r["xi"]
    S = [tuple(x) for x in r["S"]]
    t0 = time.perf_counter()
    xmin = (1 - max(DELTAS)) * xi
    fw, m = M.mine(q, w, xmin)
    mine_s = time.perf_counter() - t0
    wf = EX.exact_weights(w)
    dex = None

    def ge(Y, wv, t):
        nonlocal dex
        if abs(wv - t) >= EX.BAND:
            return wv >= t
        if dex is None:
            dex = EX.d_exact(q, wf)
        ts = DA.tidset(Y, DA.item_index(q))
        return EX.n_exact(q, wf, ts) / dex >= EX.frac(t)

    at_xi = {Y for Y, wv in fw.items() if ge(Y, wv, xi)}
    hf, rows = ME.hf_exact(q, w, xi, S)
    wus_S = {tuple(X): wv for X, wv, _ in rows}
    res = {
        "key": r["key"],
        "dataset": r["dataset"],
        "method": r["method"],
        "xi": xi,
        "n_S": len(S),
        "n_fwup_qprime_xi": len(at_xi),
        "mine_s": mine_s,
        "xi_min": xmin,
        "n_fwup_qprime_xi_min": len(fw),
        "wus_ratio_S": sorted(wus_S[X] / xi for X in S),
        "deltas": {},
    }
    for d in DELTAS:
        t = (1 - d) * xi
        at_t = {Y for Y, wv in fw.items() if ge(Y, wv, t)}
        new = at_t - at_xi
        reexp = [X for X in S if wus_S[X] >= t and ge(X, wus_S[X], t)]
        res["deltas"][f"{d:.2f}"] = {
            "threshold": t,
            "reexposed": len(reexp) / len(S),
            "n_reexposed": len(reexp),
            "n_fwup": len(at_t),
            "n_new": len(new),
            "sfwup_in_new": (len(set(S) & new) / len(new)) if new else 0.0,
        }
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)


def main(exps):
    os.makedirs(os.path.join(OUT, "units"), exist_ok=True)
    log = open(os.path.join(OUT, "reexposure.log"), "a")
    for exp in exps:
        for f in sorted(glob.glob(os.path.join(REPO, "runs", exp, "units", "*.json"))):
            r = json.load(open(f))
            if r.get("status") != "OK" or not r.get("feasible"):
                continue
            op = os.path.join(OUT, "units", f"{exp}__{r['key']}.json")
            if os.path.exists(op):
                continue
            log.write(f"{time.strftime('%F %T')} [start] {exp}/{r['key']}\n")
            log.flush()
            try:
                subprocess.run(
                    [sys.executable, __file__, "child", f, op],
                    cwd=REPO,
                    timeout=LIMIT_S,
                    check=True,
                    env=dict(os.environ, PYTHONHASHSEED="0", PYTHONPATH=REPO),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                log.write(f"{time.strftime('%F %T')} [done] {exp}/{r['key']}\n")
            except subprocess.TimeoutExpired:
                json.dump(
                    {"key": r["key"], "status": "NOT_ENUMERABLE", "reason": f"> {LIMIT_S} s"},
                    open(op, "w"),
                )
                log.write(f"{time.strftime('%F %T')} [not enumerable] {exp}/{r['key']}\n")
            except subprocess.CalledProcessError as e:
                json.dump(
                    {"key": r["key"], "status": "ERROR", "reason": e.stderr.decode()[-800:]},
                    open(op, "w"),
                )
                log.write(f"{time.strftime('%F %T')} [error] {exp}/{r['key']}\n")
            log.flush()
    with open(os.path.join(OUT, "reexposure.jsonl"), "w") as out:
        for f in sorted(glob.glob(os.path.join(OUT, "units", "*.json"))):
            exp, key = os.path.basename(f)[:-5].split("__", 1)
            r = json.load(open(f))
            if "deltas" in r:
                out.write(json.dumps(dict({"experiment": exp}, **r), sort_keys=True) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child(sys.argv[2], sys.argv[3])
    elif len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        raise SystemExit(__doc__)
