"""Automatic consistency checks on every unit result, run by the driver right after the unit
finishes. Returns {"ok", "issues", "warn", "stop"}; any entry in "stop" halts the driver
(a violated guarantee of the proposed method).
"""

import math

PROPOSAL = ("SDIF", "Max-RISWU", "Min-RISWU", "SC-only")
UNIT_RANGE = (
    "hf",
    "hf_xi",
    "hf_target",
    "hf_mined",
    "mc",
    "ac",
    "dus",
    "tmr",
    "ddi",
    "removal_share",
    "removal_share_alpha",
    "t_riswu_share",
    "pct_pre_hidden",
    "share_delete",
)
REQUIRED = (
    "dataset",
    "xi",
    "xi_target",
    "workload_id",
    "method",
    "regular_step",
    "seed",
    "code_tag",
    "hf_target",
    "hf_xi",
    "mc",
    "ac",
    "ius",
    "dus",
    "tmr",
    "ddi",
    "rt_wall_hide",
    "rt_cpu_hide",
    "rt_wall_total",
    "peak_rss_hide_mb",
    "rss_before_hide_mb",
    "delta_rss_hide_mb",
    "feasible",
    "censored",
    "n_items_removed",
    "n_S",
    "n_fwup",
    "n_nsfwup",
    "removal_share",
    "n_pre_hidden",
    "qprime",
)
REQUIRED_PROPOSAL = (
    "n_regular_rounds",
    "n_tier1",
    "n_tier2",
    "n_cap_hit",
    "n_rehide",
    "n_final_rounds",
    "t_riswu_share",
    "removal_share_alpha",
)


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def check(r, budget_s=7200.0):
    issues, warn, stop = [], [], []
    if r.get("status") != "OK":
        return {"ok": False, "issues": [f"status {r.get('status')}"], "warn": [], "stop": []}
    prop = r.get("method") in PROPOSAL
    swm = prop and r.get("mechanism", "swm") == "swm"
    cens = bool(r.get("censored"))
    # --- stop conditions
    if swm and not cens and r.get("hf_target", r.get("hf")) != 0.0:
        stop.append(f"proposal HF(xi_target) = {r.get('hf_target')} > 0")
    if prop and r.get("final_check_stuck"):
        stop.append("final verification loop did not terminate (final_check_stuck)")
    if prop and r.get("n_escalate_fail", 0) > 0:
        stop.append(f"Sub-Alg. 4 returned failure with T_X != ∅ " f"({r['n_escalate_fail']} times)")
    if swm and cens:
        warn.append("proposal (SWM) hit the 7200 s budget")
    # --- ranges / NaN
    for k, v in r.items():
        if _num(v) and (math.isnan(v) or math.isinf(v)):
            issues.append(f"{k} is NaN/inf")
    for k in UNIT_RANGE:
        v = r.get(k)
        if _num(v) and not (0.0 <= v <= 1.0):
            issues.append(f"{k} = {v} outside [0, 1]")
    if _num(r.get("ius")) and not (0.0 <= r["ius"] <= 1.0):
        warn.append(f"ius = {r['ius']} outside [0, 1] (Def. 3 IUS is not capped)")
    # --- MC / AC against |FWUP'|, |NSFWUP'|
    if r.get("n_ns"):
        if r["mc"] != r["n_ns_lost"] / r["n_ns"]:
            issues.append("MC != n_ns_lost / n_ns")
    if r.get("n_fwup_q"):
        if r["ac"] != r["n_artificial"] / r["n_fwup_q"]:
            issues.append("AC != n_artificial / |FWUP|")
        s_in = round(r["hf_mined"] * r["n_S"])
        if r["n_fwup_qprime"] != (r["n_ns"] - r["n_ns_lost"]) + r["n_artificial"] + s_in:
            issues.append("|FWUP'| != |NS ∩ NS'| + |artificial| + |S ∩ FWUP'|")
    if r.get("hf_xi", r.get("hf")) != r.get("hf_mined") and not r.get("miner_borderline_qprime"):
        issues.append(f"HF exact {r.get('hf')} != HF via miner {r.get('hf_mined')}")
    # --- feasible / censored consistent with HF and RT
    feas = r.get("hf_target", r.get("hf")) == 0.0 and not cens
    if bool(r.get("feasible")) != feas:
        issues.append("feasible flag inconsistent with HF(xi_target) / censored")
    bud = r.get("budget_s", (r.get("hide_config") or {}).get("budget_s", budget_s))
    if cens and r.get("rt_total", 0) < 0.99 * bud:
        issues.append(f"censored but rt_total {r.get('rt_total')} < budget {bud}")
    if not cens and r.get("rt_total", 0) > bud + 60:
        issues.append(f"not censored but rt_total {r.get('rt_total')} > budget {bud}")
    # --- Q'
    if r.get("qprime_iu_ok") is False:
        issues.append("Q' has a non-integer or non-positive iu")
    if r.get("n_empty_tx_qprime", 0) > r.get("n_empty_tx_q", 0):
        issues.append(
            f"Q' has {r['n_empty_tx_qprime'] - r.get('n_empty_tx_q', 0)} unexpected empty "
            "transactions"
        )
    # --- columns
    req = REQUIRED + (REQUIRED_PROPOSAL if prop else ())
    miss = [k for k in req if k not in r]
    if miss:
        issues.append(f"missing columns {miss}")
    return {"ok": not issues and not stop, "issues": issues, "warn": warn, "stop": stop}
