"""Miner: exact FWUP set against brute-force enumeration (minimal DBs that an Apriori-style
miner with an unsound pruning bound gets wrong, and random small DBs), and the exact decision
inside the band."""

import itertools
import random

import pytest

from hiding_sfwup import data as DA
from hiding_sfwup import miner as M

# (xi, eu, transactions, a FWUP that must be found)
MINIMAL = {
    "superset_of_infrequent_subset": (
        0.2,
        {1: 6.0, 2: 1.0, 3: 2.0},
        [{2: 2, 1: 2, 3: 1}, {1: 1}, {3: 3}, {2: 2}],
        (1, 2, 3),
    ),
    "pair_above_its_items_bound": (
        0.5,
        {1: 3.0, 2: 2.0, 3: 1.0},
        [{1: 1, 2: 2}, {1: 2}, {3: 1}, {2: 1, 1: 3}],
        (1, 2),
    ),
    "triple_with_light_item": (
        0.2,
        {1: 1.0, 2: 2.0, 3: 3.0, 4: 3.0},
        [{1: 3}, {4: 3, 1: 3, 3: 2}, {4: 2}, {3: 3}],
        (1, 3, 4),
    ),
}


def brute(tx, w, xi):
    twu = [DA.twu_of(t, w) for t in tx]
    D = sum(twu)
    items = sorted({i for t in tx for i in t})
    out = {}
    for r in range(1, len(items) + 1):
        for X in itertools.combinations(items, r):
            n = sum(tw for t, tw in zip(tx, twu) if set(X) <= set(t))
            if n / D >= xi:
                out[X] = n / D
    return out


@pytest.mark.parametrize("case", sorted(MINIMAL))
def test_minimal_dbs(case):
    xi, w, tx, must = MINIMAL[case]
    got, _ = M.mine(tx, w, xi)
    assert set(got) == set(brute(tx, w, xi))
    assert must in got


@pytest.mark.parametrize("seed", range(30))
def test_random_dbs_equal_brute_force(seed):
    rng = random.Random(seed)
    n_items = rng.randint(3, 7)
    tx = [
        {i: rng.randint(1, 5) for i in rng.sample(range(1, n_items + 1), rng.randint(1, n_items))}
        for _ in range(rng.randint(3, 15))
    ]
    w = {i: float(rng.randint(1, 10)) for i in range(1, n_items + 1)}
    xi = rng.choice([0.1, 0.2, 0.3, 0.45])
    got, m = M.mine(tx, w, xi, collect_u=True)
    want = brute(tx, w, xi)
    assert set(got) == set(want)
    assert all(abs(got[p] - want[p]) < 1e-12 for p in got)
    for p in got:  # Definition 3 utility
        assert m.u[p] == pytest.approx(
            sum(sum(w[i] * t[i] for i in p) for t in tx if set(p) <= set(t)), rel=1e-12
        )
