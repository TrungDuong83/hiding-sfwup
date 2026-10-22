"""Smoke test and determinism: chess and retail at the main thresholds, 4 proposed strategies +
2 baselines, run twice through the runner (one child process per unit) into two directories;
every non-RT field of every unit and Q' must be identical."""

import gzip
import json
import os
import pickle

import pytest

from hiding_sfwup import runner as R

VOLATILE = {
    "rt_total",
    "rt_hide",
    "rt_phase1",
    "rt_build",
    "cpu_total",
    "cpu_hide",
    "cpu_phase1",
    "cpu_eval",
    "eval_s",
    "load_s",
    "peak_rss_mb",
    "peak_rss_hide_mb",
    "cpu_build",
    "threads_after_hide",
    "started",
    "finished",
    "qprime",
    "sfwup_log",
    "rt_wall_hide",
    "rt_cpu_hide",
    "rt_wall_total",
    "rss_before_hide_mb",
    "delta_rss_hide_mb",
    "peak_rss_load_mb",
    "peak_rss_load_hide_mb",
    "t_order_s",
    "t_victim_s",
    "t_riswu_share",
}


def _strip(r):
    out = {k: v for k, v in r.items() if k not in VOLATILE}
    out["sfwup_log"] = [dict(x) for x in r.get("sfwup_log", [])]
    return out


@pytest.mark.slow
def test_smoke_deterministic(tmp_path):
    units = R.plan_units("smoke")
    for run in ("run1", "run2"):
        assert R.drive(units, os.path.join(tmp_path, run)) == 0
    for u in units:
        key = R.unit_key(u)
        r1 = json.load(open(os.path.join(tmp_path, "run1", "units", f"{key}.json")))
        r2 = json.load(open(os.path.join(tmp_path, "run2", "units", f"{key}.json")))
        assert r1["status"] == r2["status"] == "OK", key
        assert _strip(r1) == _strip(r2), key
        q1 = pickle.load(gzip.open(os.path.join(tmp_path, "run1", "qprime", f"{key}.pkl.gz")))
        q2 = pickle.load(gzip.open(os.path.join(tmp_path, "run2", "qprime", f"{key}.pkl.gz")))
        assert q1 == q2, key
        assert not r1.get("censored"), key
        if u["method"] in R.PROPOSAL:
            assert r1["hf"] == 0.0, key  # baselines may leave resurfaced SFWUP
