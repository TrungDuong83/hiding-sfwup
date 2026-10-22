"""Runner columns and the automatic checks; one run of each kind on chess compared with the
published results (non-RT columns and SHA-256 of Q')."""

import csv
import gzip
import hashlib
import json
import os
import pickle

import pytest

from hiding_sfwup import checks
from hiding_sfwup import data as DA
from hiding_sfwup import runner as R


def test_checks_stop():
    base = {
        "status": "OK",
        "method": "SDIF",
        "mechanism": "swm",
        "censored": False,
        "hf": 0.1,
        "hf_xi": 0.1,
        "hf_target": 0.1,
        "hf_mined": 0.1,
        "feasible": False,
        "n_S": 10,
        "rt_total": 1.0,
    }
    q = checks.check(base)
    assert q["stop"] and not q["ok"]
    q = checks.check(dict(base, method="SMSE", mechanism="baseline"))
    assert not q["stop"]
    q = checks.check(
        dict(base, hf=0.0, hf_xi=0.0, hf_target=0.0, hf_mined=0.0, feasible=True, n_escalate_fail=1)
    )
    assert any("Sub-Alg. 4" in x for x in q["stop"])


def test_experiments_cover_configs():
    cfg_dir = os.path.join(DA.REPO, "experiments", "configs")
    runs = {
        f[:-5]
        for f in os.listdir(cfg_dir)
        if json.load(open(os.path.join(cfg_dir, f)))["type"] == "runs"
    }
    assert runs == set(R.EXPERIMENTS)


def qprime_sha256(path):
    """SHA-256 of Q' in canonical form: the saved diff {tid: {item: iu}} sorted by tid, item."""
    with gzip.open(path, "rb") as f:
        diff = pickle.load(f)
    canon = [
        [int(k), sorted((int(i), int(q)) for i, q in t.items())] for k, t in sorted(diff.items())
    ]
    return hashlib.sha256(json.dumps(canon, separators=(",", ":")).encode()).hexdigest()


def _published(exp):
    rows = {
        r["key"]: r
        for r in csv.DictReader(open(os.path.join(DA.REPO, "results", exp, "summary.csv")))
    }
    sha = {
        r["key"]: r["qprime_sha256"]
        for r in csv.DictReader(open(os.path.join(DA.REPO, "results", exp, "qprime_sha256.csv")))
    }
    return rows, sha


@pytest.mark.slow
def test_runner_columns(tmp_path):
    os.environ["HIDING_SFWUP_CODE_TAG"] = "test"
    specs = (
        R._u("x1", "x1_chess", "SDIF"),
        R._u("x1", "x1_chess", "SMSE"),
        R._u("x7b", "x1_chess", "Min-RISWU", "d10", delta=0.10),
        R._u("x7b", "x1_chess", "SMSE", "d10", delta=0.10),
        R._u(
            "x7b", "x1_chess", "SDIF", "d10_greedy-batch", {"mechanism": "greedy-batch"}, delta=0.10
        ),
    )
    for spec in specs:
        r = R.run_unit(spec, str(tmp_path))
        q = checks.check(json.loads(json.dumps(r, default=R._jd)))
        assert q["ok"], (spec, q)
        assert r["xi_target"] == pytest.approx((1 - (spec.get("delta") or 0)) * 0.89)
        assert (
            r["peak_rss_hide_mb"] >= r["rss_before_hide_mb"] - 1.0 and r["rss_before_hide_mb"] > 0
        )
        if spec["method"] != "SMSE":
            assert r["hf_target"] == 0.0 and 0 <= r["t_riswu_share"] <= 1
        assert 0 <= r["removal_share"] <= 1
        exp = "main_comparison" if spec["plan"] == "x1" else "equal_margin"
        rows, sha = _published(exp)
        key = R.unit_key(spec)
        for k in ("hf", "hf_target", "mc", "ac", "ius", "dus", "tmr", "ddi", "n_deletions"):
            assert repr(r[k]) == rows[key][k] or str(r[k]) == rows[key][k], (key, k)
        assert qprime_sha256(os.path.join(tmp_path, "qprime", f"{key}.pkl.gz")) == sha[key]
