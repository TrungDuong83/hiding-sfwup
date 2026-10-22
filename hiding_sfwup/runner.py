"""Experiment runner. One unit (workload x method x variant) = one child process; units run
strictly sequentially; a unit whose result JSON already exists and is valid (status OK, same spec
hash) is skipped, so an interrupted run resumes. The command-line entry point is
experiments/run_experiments.py; the child is `python -m hiding_sfwup.runner unit <spec> <out>`.

Output per unit: <out>/units/<key>.json (every metric, configuration, workload SHA-256);
Q' as a compressed diff against Q in <out>/qprime/<key>.pkl.gz; one row per unit in
<out>/summary.csv.
"""

import csv
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time

from . import data as DA

RUNS_DIR = os.path.join(DA.REPO, "runs")
PROPOSAL = ("SDIF", "Max-RISWU", "Min-RISWU", "SC-only")
BASELINE = ("SMSE", "MSU-MIU")
METHODS = PROPOSAL + BASELINE
BASELINE_BUDGET_S = 7200.0
EVAL_ALLOWANCE_S = 3600.0
RLIMIT_AS_BYTES = 50 * 1024**3
X1_DS = ["chess", "mushroom", "accident", "retail", "kosarak", "bms-pos", "chainstore"]

# experiment -> unit groups ("plans"). The plan names are stored in every result row (column
# `plan`) and prefix the workload identifiers, so they are kept as they were in the published runs.
EXPERIMENTS = {
    "main_comparison": ["x1"],
    "cap_ablation": ["x1_cap"],
    "reduction_mechanism": ["x2"],
    "riswu_variants": ["x2b"],
    "ordering_control": ["x2c"],
    "threshold": ["x3_prop", "x3_base", "x3_base_fixed"],
    "workload": ["x4", "x4_base"],
    "utility_distribution": ["x5", "x5b"],
    "scalability": ["x6", "x6b"],
    "equal_margin": ["x7b"],
}

CSV_COLS = [
    "key",
    "plan",
    "dataset",
    "mult",
    "xi",
    "xi_target",
    "workload_id",
    "method",
    "variant",
    "mechanism",
    "regular_step",
    "seed",
    "code_tag",
    "n_S",
    "n_fwup",
    "n_nsfwup",
    "status",
    "feasible",
    "censored",
    "hf",
    "hf_xi",
    "hf_target",
    "mc",
    "ac",
    "ius",
    "dus",
    "tmr",
    "ddi",
    "hf_mined",
    "rt_wall_hide",
    "rt_cpu_hide",
    "rt_wall_total",
    "rt_total",
    "rt_hide",
    "rt_phase1",
    "cpu_total",
    "cpu_hide",
    "cpu_phase1",
    "eval_s",
    "rss_before_hide_mb",
    "peak_rss_hide_mb",
    "delta_rss_hide_mb",
    "peak_rss_load_hide_mb",
    "peak_rss_mb",
    "n_iter",
    "n_regular_rounds",
    "n_regular_iter",
    "n_escalation_iter",
    "n_tier1",
    "n_tier2",
    "n_items_removed",
    "n_deletions",
    "n_cap_hit",
    "removal_share",
    "removal_share_alpha",
    "red_delete",
    "red_decrease",
    "share_delete",
    "n_rehide",
    "n_final_rounds",
    "t_riswu_share",
    "n_pre_hidden",
    "pct_pre_hidden",
    "n_skip_removed",
    "n_skip_removed_above",
    "n_wus_eval",
    "n_steps",
    "n_delete",
    "n_decrease",
    "threads_after_hide",
    "children_after_hide",
    "qc1",
    "calib",
    "calib_sha256",
    "git_hash",
    "n_guard_alpha",
    "n_guard_qstar",
    "n_guard_qstar_skipped",
    "n_guard_skip",
    "n_guard_flip",
]


def git_hash():
    g = os.path.join(DA.REPO, ".git")
    try:
        head = open(os.path.join(g, "HEAD")).read().strip()
        if head.startswith("ref: "):
            ref = head[5:]
            p = os.path.join(g, ref)
            if os.path.exists(p):
                return open(p).read().strip()
            for line in open(os.path.join(g, "packed-refs")):
                if line.strip().endswith(" " + ref):
                    return line.split()[0]
        return head
    except OSError:
        return "unknown"


def peak_rss_mb():
    """Peak RSS of this process (VmHWM). ru_maxrss is not used: on Linux it keeps the parent's
    high-water mark across fork+exec."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return None


def reset_peak_rss():
    """Reset VmHWM to the current RSS (Linux >= 4.0, /proc/self/clear_refs '5') so that VmHWM
    read after hiding is the peak of the hiding phase only."""
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


def code_tag():
    """git describe of the working tree (read-only); '-dirty' when uncommitted changes."""
    try:
        r = subprocess.run(
            ["git", "describe", "--tags", "--dirty", "--always"],
            cwd=DA.REPO,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def proc_threads():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("Threads:"):
                return int(line.split()[1])
    return None


def spec_hash(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def unit_key(spec):
    v = spec.get("variant") or "std"
    return f"{spec['calib']}__{spec['method']}__{v}"


# ------------------------------------------------------------------ unit lists
def _u(plan, calib, method, variant="std", cfg=None, **kw):
    d = {"plan": plan, "calib": calib, "method": method, "variant": variant, "cfg": cfg or {}}
    d.update(kw)
    return d


def plan_units(plan):
    """Units of one plan, in run order."""
    from . import workloads as W

    if plan == "x1":  # main comparison
        return [_u(plan, f"x1_{ds}", m) for ds in X1_DS for m in METHODS]
    if plan == "x1_cap":  # cap variant of the regular step (datasets where it is feasible)
        return [
            _u(plan, f"x1_{ds}", m, "cap", {"regular_step": "cap"})
            for ds in ("chess", "mushroom", "retail", "chainstore")
            for m in PROPOSAL
        ]
    if plan == "x2":  # reduction mechanism: greedy, victim SDIF
        return [
            _u(plan, f"x1_{ds}", "SDIF", mech, {"mechanism": mech})
            for ds in X1_DS
            for mech in ("greedy-batch", "greedy-unit")
        ]
    if plan == "x2b":  # RISWU without twu / without Wt
        return [
            _u(plan, f"x1_{ds}", m, v, {"riswu_w": v})
            for ds in X1_DS
            for m in ("Max-RISWU", "Min-RISWU")
            for v in ("no_twu", "no_wt")
        ]
    if plan == "x2c":  # SC-only victim + wus(X)-descending order, no RISWU
        return [_u(plan, f"x1_{ds}", "SC-only", "wusorder", {"order": "wus"}) for ds in X1_DS]
    if plan in ("x3_prop", "x3_base"):  # threshold grid: proposals, then baselines (re-selected S)
        return [
            dict(u, plan=plan)
            for u in plan_units("x3")
            if (u["method"] in PROPOSAL) == (plan == "x3_prop")
        ]
    if plan == "x3":
        out = []
        for ds, grid in W.X3_GRID.items():
            for mm in grid:
                c = f"x3_{ds}_{mm}"
                out += [_u(plan, c, m, "reselect") for m in METHODS]
                out += [_u(plan, c, m, "fixed", sset="S_fixed") for m in PROPOSAL]
        return out
    if plan == "x3_base_fixed":  # threshold grid: baselines on the fixed S
        return [
            _u(plan, f"x3_{ds}_{mm}", m, "fixed", sset="S_fixed")
            for ds, grid in W.X3_GRID.items()
            for mm in grid
            for m in BASELINE
        ]
    if plan in ("x4", "x4_base"):  # SFWUP choice x |S|; proposals, then baselines
        keys = [
            f"rho{r:.2f}_{k}"
            for r in W.X4_RHO
            for k in ["ovhigh", "ovlow"] + [f"r{i}" for i in range(5)]
        ]
        ms = PROPOSAL if plan == "x4" else BASELINE
        return [
            _u(plan, f"x1_{ds}", m, k, S_from=[f"x4_{ds}", k])
            for ds in W.X4_DS
            for k in keys
            for m in ms
        ]
    if plan == "x5":  # utility distribution, seed 0 (proposals)
        return [
            _u(plan, f"x5_{ds}_{d}", m)
            for ds in ("chess", "kosarak")
            for d in ("normal", "lognormal10")
            for m in PROPOSAL
        ]
    if plan == "x5b":  # 3 distributions x 5 seeds x 2 modes x 6 methods
        out = []
        for ds in W.X5_DS:
            for dist in W.X5B_DISTS:
                for sd in W.X5B_SEEDS:
                    c = f"x5b_{ds}_{dist}_s{sd}"
                    for m in METHODS:
                        # seed-0 normal/lognormal10 re-selected proposals = the x5 units
                        if not (sd == 0 and dist in ("normal", "lognormal10") and m in PROPOSAL):
                            out.append(_u(plan, c, m, "reselect", seed=sd))
                    if W.read_workload(c).get("n_S_x1fixed", 0) >= 5:
                        out += [
                            _u(plan, c, m, "x1fixed", sset="S_x1fixed", seed=sd) for m in METHODS
                        ]
        return out
    if plan == "x6":  # scalability, one permutation, 4 sizes
        return [
            _u(plan, f"x6_{ds}_{p}", m)
            for ds in ("kosarak", "chainstore")
            for p in (10, 25, 50, 100)
            for m in ("SDIF", "SMSE", "MSU-MIU")
        ]
    if plan == "x6b":  # scalability, 10 nested sizes x 3 permutations x 6 methods
        return [
            _u(plan, f"x6b_{ds}_p{pm}_{pc}", m, seed=pm)
            for ds in W.X6B_DS
            for pm in W.X6B_PERMS
            for pc in W.X6B_PCT
            for m in METHODS
        ]
    if plan == "x7b":  # equal safety margin: hide below xi_t = (1 - delta) xi
        out = []
        for ds in X1_DS:
            for d in (0.05, 0.10, 0.20):
                v = f"d{int(round(d * 100)):02d}"
                out += [_u(plan, f"x1_{ds}", m, v, delta=d) for m in METHODS]
                out += [
                    _u(plan, f"x1_{ds}", "SDIF", f"{v}_{mech}", {"mechanism": mech}, delta=d)
                    for mech in ("greedy-batch", "greedy-unit")
                ]
        return out
    if plan.startswith("smoke"):  # quick check: chess and retail, every method
        return [_u(plan, f"x1_{ds}", m) for ds in ("chess", "retail") for m in METHODS]
    raise ValueError(plan)


# rough per-unit estimates for the log (seconds): (proposal, MSU-MIU, SMSE)
_EST = {
    "chess": (5, 10, 10),
    "mushroom": (8, 20, 20),
    "retail": (20, 20, 60),
    "bms-pos": (60, 60, 600),
    "accident": (300, 300, 7200),
    "kosarak": (150, 120, 1800),
    "chainstore": (120, 100, 1800),
}


def estimate(spec):
    ds = spec["calib"].split("_")[1]
    p, mm, s = _EST.get(ds, (60, 60, 600))
    return {"SMSE": s, "MSU-MIU": mm}.get(spec["method"], p)


# ------------------------------------------------------------------ child
def workload_id(spec):
    """Workload = (workload file, source of S) -- identical across methods."""
    if spec.get("workload_id"):
        return spec["workload_id"]
    src = ":".join(spec["S_from"]) if spec.get("S_from") else spec.get("sset", "S")
    return f"{spec['calib']}|{src}"


def run_unit(spec, out):
    from . import baselines as B
    from . import hiding as H
    from . import metrics as ME
    from . import workloads as W

    resource.setrlimit(resource.RLIMIT_AS, (RLIMIT_AS_BYTES, RLIMIT_AS_BYTES))
    key = unit_key(spec)
    rec = W.read_workload(spec["calib"])
    ds, xi = rec["dataset"], rec["xi"]
    delta = spec.get("delta")
    xi_t = (1.0 - delta) * xi if delta else xi  # hide below xi_t; NS / side effects at xi
    wfile = DA.weights_path(rec["weights_file"]) if "/" in rec["weights_file"] else None
    t0 = time.perf_counter()
    tx, w = DA.load_db(ds, wfile)
    if rec.get("sample"):
        sm = rec["sample"]
        idx = W.sample_indices(len(tx), sm["frac"], sm["seed"])
        if hashlib.sha256(idx.astype("int64").tobytes()).hexdigest() != sm["indices_sha256"]:
            raise RuntimeError("sample indices mismatch")
        tx = [tx[k] for k in idx.tolist()]
    load_s = time.perf_counter() - t0
    fw = W.fwup_of(rec)
    if spec.get("S_from"):  # [workload file, set key]
        srec = W.read_workload(spec["S_from"][0])
        S = [tuple(x) for x in srec["sets"][spec["S_from"][1]]]
    else:  # sset = "S" (default), "S_fixed", "S_x1fixed"
        S = [tuple(x) for x in rec[spec.get("sset", "S")]]
    m = spec["method"]
    cfgd = dict(spec.get("cfg") or {})
    res = {
        "key": key,
        "spec": spec,
        "spec_hash": spec_hash(spec),
        "dataset": ds,
        "mult": rec["mult"],
        "xi": xi,
        "xi_target": xi_t,
        "delta": delta,
        "workload_id": workload_id(spec),
        "seed": spec.get("seed", ""),
        "code_tag": os.environ.get("HIDING_SFWUP_CODE_TAG", "unknown"),
        "method": m,
        "variant": spec.get("variant", "std"),
        "n_S": len(S),
        "S": [list(x) for x in S],
        "calib": spec["calib"],
        "calib_sha256": rec["_sha256"],
        "git_hash": git_hash(),
        "python": platform.python_version(),
        "numpy": __import__("numpy").__version__,
        "host": platform.node(),
        "load_s": load_s,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    res["peak_rss_load_mb"] = peak_rss_mb()
    res["rss_peak_reset"] = reset_peak_rss()
    res["rss_before_hide_mb"] = rss_mb()  # = VmHWM right after the reset
    if m in PROPOSAL:
        hc = H.HideConfig(**cfgd)
        h = H.Hider(tx, w, xi_t, S, fw, m, hc)
        q = h.run()
        res.update(h.times)
        res.update({k: v for k, v in h.stats.items()})
        res["censored"] = bool(h.stats["censored"])
        res["order"] = [[list(X), h.order_keys[X][0], h.order_keys[X][1]] for X in h.order]
        res["sfwup_log"] = h.sfwup_log
        res["hide_config"] = hc.__dict__
        res["mechanism"] = hc.mechanism
        res["regular_step"] = hc.regular_step
        res["t_riswu_share"] = (
            (h.stats["t_order_s"] + h.stats["t_victim_s"]) / h.times["rt_hide"]
            if h.times["rt_hide"] > 0
            else None
        )
        res["pct_pre_hidden"] = h.stats["n_pre_hidden"] / len(S) if S else None
    else:
        budget = cfgd.pop("budget_s", BASELINE_BUDGET_S)
        if m == "SMSE":
            q, st = B.smse(tx, w, fw, S, xi=xi_t, deadline_s=budget, xi_ns=xi, **cfgd)
        else:
            q, st = B.msumiu(tx, w, S, xi=xi_t, deadline_s=budget, **cfgd)
        res.update(st)
        res["rt_phase1"] = st["rt_build"]
        res["cpu_phase1"] = st["cpu_build"]
        res["n_deletions"] = st["n_delete"]
        res["budget_s"] = budget
        res["mechanism"] = "baseline"
        res["regular_step"] = ""
        # from the per-SFWUP log: skipped because below xi_t at its turn, or skipped because
        # removed from the table earlier (counted separately, also when back above xi_t)
        xl = st.get("xlog", [])
        below = sum(1 for x in xl if x.get("action") == "skip_below")
        rem = [x for x in xl if x.get("action") == "skip_removed"]
        rem_above = sum(1 for x in rem if x.get("wus_turn") is not None and x["wus_turn"] >= xi_t)
        res["n_pre_hidden"] = below + len(rem) - rem_above
        res["pct_pre_hidden"] = res["n_pre_hidden"] / len(S) if S else None
        res["n_skip_removed"] = len(rem)
        res["n_skip_removed_above"] = rem_above
    res["peak_rss_hide_mb"] = peak_rss_mb()  # VmHWM since the reset = peak of the hiding phase
    res["delta_rss_hide_mb"] = res["peak_rss_hide_mb"] - res["rss_before_hide_mb"]
    res["peak_rss_load_hide_mb"] = max(res["peak_rss_load_mb"], res["peak_rss_hide_mb"])
    res["threads_after_hide"] = proc_threads()
    res["children_after_hide"] = len(__import__("psutil").Process().children(recursive=True))
    rd, rc = res.get("red_delete"), res.get("red_decrease")
    res["share_delete"] = (
        (rd / (rd + rc)) if rd is not None and rc is not None and (rd + rc) > 0 else None
    )
    res["removal_share_alpha"] = res["share_delete"] if res["mechanism"] == "swm" else None
    a, b = res.get("rs_delete"), res.get("rs_decrease")
    res["removal_share"] = (
        (a / (a + b)) if a is not None and b is not None and (a + b) > 0 else None
    )
    res["n_regular_rounds"] = res.get("n_regular_iter")
    res["n_items_removed"] = res.get("n_deletions")
    res["rt_wall_hide"], res["rt_cpu_hide"], res["rt_wall_total"] = (
        res["rt_hide"],
        res["cpu_hide"],
        res["rt_total"],
    )
    ce = time.process_time()
    ev = ME.evaluate(tx, q, w, xi, S, fw, rec["sum_u_def3"])  # side effects at the original xi
    res["cpu_eval"] = time.process_time() - ce
    res.update(ev)
    res["hf_xi"] = ev["hf"]
    if xi_t != xi:
        res["hf_target"], res["hf_target_rows"] = ME.hf_exact(q, w, xi_t, S)
    else:
        res["hf_target"] = ev["hf"]
    res["n_fwup"], res["n_nsfwup"] = ev["n_fwup_q"], ev["n_ns"]
    res["n_empty_tx_qprime"] = sum(1 for t in q if not t)
    res["n_empty_tx_q"] = sum(1 for t in tx if not t)
    res["qprime_iu_ok"] = all(isinstance(v, int) and v > 0 for t in q for v in t.values())
    os.makedirs(os.path.join(out, "qprime"), exist_ok=True)
    qp = os.path.join(out, "qprime", f"{key}.pkl.gz")
    ME.save_qprime(qp, tx, q)
    res["qprime"] = os.path.relpath(qp, DA.REPO)
    res["peak_rss_mb"] = max(peak_rss_mb(), res["peak_rss_load_hide_mb"])
    res["status"] = "OK"
    res["feasible"] = bool(res["hf_target"] == 0.0 and not res.get("censored"))
    res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.join(out, "units"), exist_ok=True)
    p = os.path.join(out, "units", f"{key}.json")
    with open(p + ".tmp", "w") as f:
        json.dump(res, f, indent=1, default=_jd)
    os.replace(p + ".tmp", p)
    return res


def _jd(o):
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


# ------------------------------------------------------------------ driver
def valid_result(path, spec):
    try:
        r = json.load(open(path))
    except (OSError, ValueError):
        return False
    return r.get("status") == "OK" and r.get("spec_hash") == spec_hash(spec)


def drive(units, out, log_path=None):
    """Run the units sequentially, one child process each; stop at the first failed guarantee
    (see checks.py). Returns 0 on success, 3 after a stop."""
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "units"), exist_ok=True)
    log_path = log_path or os.path.join(out, "runner.log")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    tag = code_tag()
    log(f"[plan] {len(units)} units -> {out}; git {git_hash()[:12]} code_tag {tag}")
    stop_path = os.path.join(out, "STOP.json")
    if os.path.exists(stop_path):
        log(f"[STOP] {stop_path} exists -- inspect and remove it first")
        return 3
    for n, spec in enumerate(units, 1):
        key = unit_key(spec)
        p = os.path.join(out, "units", f"{key}.json")
        if valid_result(p, spec):
            log(f"[skip] {n}/{len(units)} {key} (valid result exists)")
            continue
        est = estimate(spec)
        # outer guard = in-process budget (7200 s for every method) + load/evaluation allowance
        timeout = BASELINE_BUDGET_S + EVAL_ALLOWANCE_S
        log(f"[start] {n}/{len(units)} {key} estimate ~{est:.0f} s (outer guard {timeout:.0f} s)")
        sp = os.path.join(out, "units", f".{key}.spec.json")
        with open(sp, "w") as f:
            json.dump(spec, f)
        env = dict(os.environ, PYTHONHASHSEED="0", HIDING_SFWUP_CODE_TAG=tag)
        t0 = time.perf_counter()
        try:
            r = subprocess.run(
                [sys.executable, "-m", "hiding_sfwup.runner", "unit", sp, out],
                cwd=DA.REPO,
                env=env,
                timeout=timeout,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            status = "OK" if r.returncode == 0 else f"ERROR rc={r.returncode}"
            tail = r.stdout.decode(errors="replace")[-3000:]
        except subprocess.TimeoutExpired:
            status, tail = "OUTER_TIMEOUT", ""
        el = time.perf_counter() - t0
        os.remove(sp)
        if status != "OK":
            with open(p, "w") as f:
                json.dump(
                    {
                        "key": key,
                        "spec": spec,
                        "spec_hash": spec_hash(spec),
                        "status": status,
                        "tail": tail,
                        "elapsed_s": el,
                    },
                    f,
                    indent=1,
                )
            log(f"[FAIL] {key} {status} after {el:.1f} s\n{tail}")
        else:
            from . import checks

            rr = json.load(open(p))
            q = checks.check(rr)
            rr["qc1"] = q
            with open(p + ".tmp", "w") as f:
                json.dump(rr, f, indent=1)
            os.replace(p + ".tmp", p)
            log(
                f"[done] {key} {el:.1f} s: HF={rr['hf_target']:.4f} MC={rr['mc']:.4f} "
                f"AC={rr['ac']:.4f} IUS={rr['ius']:.4f} RT={rr['rt_total']:.2f} "
                f"censored={rr.get('censored')} RSS={rr['peak_rss_mb']:.0f} MB "
                f"checks={'OK' if q['ok'] else q}"
            )
            if q["stop"]:
                with open(stop_path, "w") as f:
                    json.dump(
                        {"key": key, "stop": q["stop"], "time": time.strftime("%Y-%m-%d %H:%M:%S")},
                        f,
                        indent=1,
                    )
                log(f"[STOP] {key}: {q['stop']} -> {stop_path}; run halted")
                write_summary(out)
                return 3
    write_summary(out)
    return 0


def write_summary(out):
    rows = []
    ud = os.path.join(out, "units")
    for fn in sorted(os.listdir(ud)):
        if not fn.endswith(".json") or fn.startswith("."):
            continue
        r = json.load(open(os.path.join(ud, fn)))
        sp = r.get("spec", {})
        row = {c: r.get(c, "") for c in CSV_COLS}
        if isinstance(r.get("qc1"), dict):
            row["qc1"] = "OK" if r["qc1"]["ok"] else ";".join(r["qc1"]["issues"] + r["qc1"]["stop"])
        row["plan"] = sp.get("plan", "")
        row["method"] = r.get("method", sp.get("method", ""))
        row["variant"] = r.get("variant", sp.get("variant", ""))
        row["calib"] = r.get("calib", sp.get("calib", ""))
        if r.get("status") == "OK" and "feasible" not in r:
            row["feasible"] = bool(r["hf"] == 0.0 and not r.get("censored"))
        rows.append(row)
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=CSV_COLS)
        wr.writeheader()
        wr.writerows(rows)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "unit":
        run_unit(json.load(open(sys.argv[2])), sys.argv[3])
    else:
        raise SystemExit(
            "usage: python -m hiding_sfwup.runner unit <spec.json> <out_dir> "
            "(use experiments/run_experiments.py to run an experiment)"
        )
