"""The paper's running example under the proposed method: DB of Tables 2-3, xi = 0.4,
S = {AD, CD, EAD}. Computes every number shown for the example: twu, sumtwu, wus and tidsets of
the 11 FWUP, SC/NSC of items, SC(T)/NSC(T)/Wt(T), RISWU of every item for every SFWUP, the
processing order of the 4 strategies, and for every SWM round alpha, C_j, |S(T_j)|, du_j, diu_j
and the wus of all FWUP afterwards; final Q'.

  env PYTHONHASHSEED=0 python scripts/running_example.py [--json out.json] [--md out.md]
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hiding_sfwup import data as DA  # noqa: E402
from hiding_sfwup import hiding as H  # noqa: E402
from hiding_sfwup import metrics as ME  # noqa: E402
from hiding_sfwup import miner as M  # noqa: E402
from hiding_sfwup import phase1 as P1  # noqa: E402

XI = 0.4
S = [(1, 4), (3, 4), (1, 4, 5)]  # AD, CD, EAD
L = DA.letters


def tname(k):
    return f"T{k + 1}"


def show_t(t):
    return "{" + ", ".join(f"{DA.PAPER_INV[i]}:{q}" for i, q in sorted(t.items())) + "}"


def wus_all(tx, w, pats):
    tw = [DA.twu_of(t, w) for t in tx]
    D = sum(tw)
    return {p: sum(v for t, v in zip(tx, tw) if all(i in t for i in p)) / D for p in pats}


def run_strategy(tx, w, fw, st):
    rounds = []
    h = H.Hider(tx, w, XI, S, fw, st, H.HideConfig(trace=True))
    pats = sorted(fw)
    q = h.run()
    # per-round wus by replaying the trace on a copy of the DB
    cur = [dict(t) for t in tx]
    prev = wus_all(cur, w, pats)
    for tr in h.trace:
        for k, i, d in tr["changes"]:
            cur[k][i] -= d
            if cur[k][i] <= 0:
                del cur[k][i]
        now = wus_all(cur, w, pats)
        rounds.append(
            {
                "X": L(tr["X"]),
                "iter": tr["iter"],
                "victim": DA.PAPER_INV[tr["victim"]],
                "N": tr["N"],
                "D": tr["D"],
                "wus_X_before": tr["N"] / tr["D"],
                "alpha": tr["alpha"],
                "changes": [(tname(k), DA.PAPER_INV[i], d) for k, i, d in tr["changes"]],
                "wus_after": {L(p): now[p] for p in pats},
                "increased": [L(p) for p in pats if now[p] > prev[p] + 1e-15],
            }
        )
        prev = now
    # Prop. 4 details per round, recomputed on the replayed DB state
    hh = H.Hider(tx, w, XI, S, fw, st, H.HideConfig())
    hh._setup()
    hh.m = h.m
    hh.wt = h.wt
    for r, tr in zip(rounds, h.trace):
        TX = hh.tids_of(tuple(tr["X"]))
        C = hh.contributions(tr["victim"], TX)
        S_len = [len(hh.tx[k]) for k in TX.tolist()]
        w_total = math.fsum(c / s for c, s in zip(C.tolist(), S_len))
        e = w[tr["victim"]]
        if tr["alpha"] is not None and tr["alpha"] > 0:
            r["prop4"] = [
                {
                    "T": tname(k),
                    "iu_before": hh.tx[k][tr["victim"]],
                    "twu": float(hh.twu[k]),
                    "Wt": float(hh.wt[k]),
                    "C": c,
                    "S_len": s,
                    "du": tr["alpha"] * c / w_total,
                    "du_over_eu": tr["alpha"] * c / w_total / e,
                    "diu": min(hh.tx[k][tr["victim"]], math.ceil(tr["alpha"] * c / w_total / e)),
                }
                for k, c, s in zip(TX.tolist(), C.tolist(), S_len)
            ]
            r["w_total"] = w_total
        for k, i, d in tr["changes"]:
            hh.apply(k, i, d)
    return {
        "strategy": st,
        "order": [L(X) for X in h.order],
        "order_keys": {
            L(X): [h.order_keys[X][0], DA.PAPER_INV[h.order_keys[X][1]]] for X in h.order
        },
        "victims_run": [
            (
                L(r["X"]),
                DA.PAPER_INV.get(r["victim"]) if r["victim"] else None,
                r.get("pre_hidden", False),
                r["rehide"],
            )
            for r in h.sfwup_log
        ],
        "rounds": rounds,
        "n_rehide": h.stats["n_rehide"],
        "n_final_rounds": h.stats["n_final_rounds"],
        "n_pre_hidden": h.stats["n_pre_hidden"],
        "qprime": [show_t(t) for t in q],
        "hf": ME.hf_exact(q, w, XI, S)[0],
        "removal_share": h.stats["rs_delete"] / (h.stats["rs_delete"] + h.stats["rs_decrease"]),
    }


def compute():
    tx, w = DA.paper_example()
    fw, _ = M.mine(tx, w, XI)
    pats = sorted(fw, key=lambda p: (-fw[p], len(p), p))
    tw = [DA.twu_of(t, w) for t in tx]
    D = sum(tw)
    iidx = DA.item_index(tx)
    m = P1.compute(tx, S, list(fw))
    h = H.Hider(tx, w, XI, S, fw, "SDIF", H.HideConfig())
    h._setup()
    h.m, h.wt = m, m["wt"]
    riswu = {}
    for X in S:
        TX = h.tids_of(X)
        W = h.twu[TX] * h.wt[TX]
        riswu[L(X)] = {
            DA.PAPER_INV[i]: (
                h.riswu(i, X, TX)
                if i in X  # i not in X: u(i, T) = 0 if i not in T
                else float(
                    sum(w[i] * tx[k].get(i, 0) * Wk for k, Wk in zip(TX.tolist(), W.tolist()))
                    / W.sum()
                )
            )
            for i in sorted(DA.PAPER_INV)
        }
    res = {
        "xi": XI,
        "S": [L(X) for X in S],
        "db": [show_t(t) for t in tx],
        "eu": {DA.PAPER_INV[i]: v for i, v in sorted(w.items())},
        "tu": [DA.tu_of(t, w) for t in tx],
        "S_len": [len(t) for t in tx],
        "twu": tw,
        "sumtwu": D,
        "fwup": [(L(p), fw[p], [tname(k) for k in DA.tidset(p, iidx).tolist()]) for p in pats],
        "sc_item": {DA.PAPER_INV[i]: m["sc_item"].get(i, 0) for i in sorted(DA.PAPER_INV)},
        "nsc_item": {DA.PAPER_INV[i]: m["nsc_item"].get(i, 0) for i in sorted(DA.PAPER_INV)},
        "sc_T": m["sc_trans"].tolist(),
        "nsc_T": m["nsc_trans"].tolist(),
        "wt_T": m["wt"].tolist(),
        "riswu": riswu,
        "strategies": {},
    }
    for st in H.STRATEGIES:
        res["strategies"][st] = run_strategy(tx, w, fw, st)
    return res


def to_md(r):
    out = []
    p = out.append
    p("# Running example (Tables 2-3 of the paper)\n")
    p(
        f"xi = {r['xi']}; S = {{{', '.join(r['S'])}}} (ADE = EAD; patterns are written in "
        "alphabetical order).\n"
    )
    p("## DB, TU, twu, sumtwu\n")
    p("| T | Transaction | TU(T) | \\|S(T)\\| | twu(T) |\n|---|---|---:|---:|---:|")
    for k, (t, tu, s, tw) in enumerate(zip(r["db"], r["tu"], r["S_len"], r["twu"])):
        p(f"| T{k + 1} | {t} | {tu:.1f} | {s} | {tw:.3f} |")
    p(
        f"\nsumtwu = D = **{r['sumtwu']:.3f}**. eu: "
        + ", ".join(f"{k} = {v}" for k, v in r["eu"].items())
        + "\n"
    )
    p("## The 11 FWUP: wus and tidset\n")
    p("| FWUP | wus | tidset | in S |\n|---|---:|---|---|")
    for name, v, ts in r["fwup"]:
        p(f"| {name} | {v:.4f} | {', '.join(ts)} | {'yes' if name in r['S'] else ''} |")
    p("\n## SC and NSC of items\n")
    p("| Item | " + " | ".join(r["sc_item"]) + " |\n|---|" + "---:|" * len(r["sc_item"]))
    p("| SC(i) | " + " | ".join(str(v) for v in r["sc_item"].values()) + " |")
    p("| NSC(i) | " + " | ".join(str(v) for v in r["nsc_item"].values()) + " |")
    p("\n## SC(T), NSC(T), Wt(T) = SC(T)/(NSC(T)+1)\n")
    p("| T | SC(T) | NSC(T) | Wt(T) |\n|---|---:|---:|---:|")
    for k, (a, b, c) in enumerate(zip(r["sc_T"], r["nsc_T"], r["wt_T"])):
        p(f"| T{k + 1} | {a} | {b} | {c:.4f} |")
    p("\n## RISWU(i, X) on the original DB (W = twu*Wt); items not in X in parentheses\n")
    items = list(r["eu"])
    p("| X | " + " | ".join(items) + " |\n|---|" + "---:|" * len(items))
    for X, d in r["riswu"].items():
        p(
            f"| {X} | "
            + " | ".join((f"{d[i]:.4f}" if i in X else f"({d[i]:.4f})") for i in items)
            + " |"
        )
    p("\n## Victim and processing order (original DB) for the 4 strategies\n")
    p(
        "| Strategy | Processing order (RISWU(victim, X), victim) | Victims during the run "
        "(* = already hidden at its turn) | n_pre_hidden | n_rehide | HF | removal_share |\n"
        "|---|---|---|---:|---:|---:|---:|"
    )
    for st, s in r["strategies"].items():
        order = ", ".join(
            f"{X} ({s['order_keys'][X][0]:.4f}, {s['order_keys'][X][1]})" for X in s["order"]
        )
        vr = ", ".join(
            f"{X}: {'*' if pre else v}" + (" (re-hidden)" if reh else "")
            for X, v, pre, reh in s["victims_run"]
        )
        p(
            f"| {st} | {order} | {vr} | {s['n_pre_hidden']} | {s['n_rehide']} | {s['hf']} | "
            f"{s['removal_share']:.4f} |"
        )
    for n, (st, s) in enumerate(r["strategies"].items(), 1):
        p(f"\n## {st}: rounds\n")
        for rd in s["rounds"]:
            p(
                f"**X = {rd['X']}, round {rd['iter']}, victim {rd['victim']}**: N = {rd['N']:.6f}, "
                f"D = {rd['D']:.6f}, wus(X) = {rd['wus_X_before']:.6f}, "
                f"alpha = (N - xi D)/(1 - xi) = {rd['alpha']:.6f}"
                + (f", sum C_k/\\|S_k\\| = {rd['w_total']:.6f}" if rd.get("w_total") else "")
                + "\n"
            )
            if rd.get("prop4"):
                p(
                    "| T | iu(victim) | twu(T) | Wt(T) | C_j | \\|S(T_j)\\| | du_j | du_j/eu | "
                    "diu_j = min(iu, ceil(du_j/eu)) |\n"
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
                )
                for x in rd["prop4"]:
                    p(
                        f"| {x['T']} | {x['iu_before']} | {x['twu']:.3f} | {x['Wt']:.4f} | "
                        f"{x['C']:.4f} | {x['S_len']} | {x['du']:.4f} | {x['du_over_eu']:.4f} | "
                        f"{x['diu']} |"
                    )
            p("\nChanges: " + ", ".join(f"{t}: {i} -{d}" for t, i, d in rd["changes"]) + "\n")
            p(
                "wus after the round (^ = increased): "
                + ", ".join(
                    f"{k} = {v:.4f}{' ^' if k in rd['increased'] else ''}"
                    for k, v in rd["wus_after"].items()
                )
                + "\n"
            )
        p(
            f"n_rehide = {s['n_rehide']}, n_final_rounds = {s['n_final_rounds']}. Q' = "
            + "; ".join(f"T{k + 1} = {t}" for k, t in enumerate(s["qprime"]))
            + "\n"
        )
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    r = compute()
    if "--json" in sys.argv:
        json.dump(r, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
    if "--md" in sys.argv:
        open(sys.argv[sys.argv.index("--md") + 1], "w").write(to_md(r))
    elif "--json" not in sys.argv:
        print(to_md(r))
