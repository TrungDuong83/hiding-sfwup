"""Hiding sensitive frequent weighted utility patterns (SFWUP) by weighted modification.

The manuscript specifies the proposed method; SMSE (Liu et al., Appl. Intell. 2020) and
MSU-MIU (Lin et al., EAAI 2016) specify the two baselines.

Modules
  data       dataset loading, checksum verification, the paper's running example
  exact      exact (Fraction / integer) decisions inside the 1e-9 band around xi
  miner      FWUP miner (Eclat on tidsets, no length cap)
  phase1     SC / NSC / Wt (Phase 1 of Algorithm 1)
  hiding     proposed algorithm (Algorithm 1, Sub-Algorithms 1-4, Prop. 3-4, final check)
  baselines  SMSE and MSU-MIU on the wus measure (and on utility, for their worked examples)
  metrics    HF, MC, AC, IUS, DUS, TMR, DDI
  workloads  SFWUP selection and the workload files of every experiment
  checks     automatic consistency checks on every unit result
  runner     one unit = one child process; units run sequentially and are resumable
"""

import os

if os.environ.get("PYTHONHASHSEED") != "0":
    raise RuntimeError(
        "hiding_sfwup requires PYTHONHASHSEED=0 for deterministic set iteration; "
        "run with: env PYTHONHASHSEED=0 python ..."
    )

__version__ = "1.0.0"
