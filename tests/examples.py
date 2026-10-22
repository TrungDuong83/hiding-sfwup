"""Small DBs shared by the tests."""

from hiding_sfwup import data as DA
from hiding_sfwup import miner as M

A, B, C, D, E = 1, 2, 3, 4, 5
XI = 0.4
S_PAPER = [(A, D), (C, D), (A, D, E)]  # AD, CD, EAD (paper, running example)


def paper():
    """(tx, w, FWUP, S) of the running example."""
    tx, w = DA.paper_example()
    fw, _ = M.mine(tx, w, XI)
    return tx, w, fw, S_PAPER


# 4 transactions, 4 items (A..D), eu A=1 B=8 C=9 D=9, xi = 0.25, S = {AC, AD, AB}: with
# Max-RISWU the SFWUP AD is below xi when its turn ends but back above xi after AC and AB are
# hidden; the final verification loop hides it again.
RESURFACE_TX = [{1: 2, 2: 1}, {1: 4, 3: 3}, {1: 4, 2: 1, 3: 3, 4: 5}, {2: 4, 3: 5, 4: 5}]
RESURFACE_W = {1: 1.0, 2: 8.0, 3: 9.0, 4: 9.0}
RESURFACE_XI = 0.25
RESURFACE_S = [(1, 3), (1, 4), (1, 2)]
