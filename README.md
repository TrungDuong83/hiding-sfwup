# Hiding_SFWUP: code and data

This repository holds the code, input data and results for the paper "An Efficient Method for
Hiding Sensitive Frequent Weighted Utility Patterns" by Duong M. Trung, Trinh D. D. Nguyen,
Philippe Fournier-Viger and Bay Vo. The proposed method hides a set S of sensitive frequent
weighted utility patterns (SFWUP). It modifies the internal utilities of a transaction database
so that each pattern's weighted utility support (wus) drops below the threshold ξ, while it limits
the side effects on non-sensitive patterns. The repository contains the proposed algorithm
(Algorithm 1 with Sub-Algorithms 1–4, Propositions 3–4 and the final verification pass), the FWUP
miner, the evaluation metrics, two baselines re-implemented from their papers (SMSE and
MSU-MIU), and every input file. It also contains the per-run results behind every table and
figure, and scripts that regenerate those tables and figures.

## Requirements

- Python 3.10.12. Any recent Linux works; the paper's runs used Ubuntu 22.04 on a 12-vCPU
  machine with 64 GB RAM.
- The runner reads `/proc/self/status` for memory figures, so it needs Linux.
- RAM: most runs need under 2 GB. The largest SMSE runs on dense data peaked at about 5 GB.
  The runner caps each run at 50 GB of address space.

## Installation

```bash
git clone https://github.com/TrungDuong83/hiding-sfwup.git hiding-sfwup && cd hiding-sfwup
python3.10 -m venv .venv                          # Python 3.10.12
.venv/bin/pip install -r requirements-lock.txt    # every package pinned (pip freeze --all)
source .venv/bin/activate
```

`requirements-lock.txt` fixes the version of every installed package, including indirect
dependencies, pip and setuptools (NumPy 2.2.6, SciPy 1.14.1, Matplotlib 3.10.9, psutil 7.2.2,
pytest 9.1.1). `requirements.txt` lists only the direct dependencies, at the same versions.

Every command must run with `PYTHONHASHSEED=0`, because set iteration order has to be
deterministic. The package refuses to start without it.

## Data

All input data is included in this repository and in the source archive of the release; nothing
has to be downloaded from elsewhere.

| Dataset | File in `data/datasets/` | Size (gzip) | Transactions | Items | Origin of the item lists |
|---|---|---:|---:|---:|---|
| Chess | `chess_fimi_quantities.txt.gz` | 0.1 MB | 3,196 | 75 | SPMF [1] (UCI, via FIMI) |
| Mushroom | `mushroom_quantities.txt.gz` | 0.2 MB | 8,416 | 119 | SPMF [1] (UCI, via FIMI) |
| Accident | `accident_quantities.txt.gz` | 16.0 MB | 340,183 | 468 | SPMF [1] (FIMI) |
| Retail | `retail_quantities.txt.gz` | 2.3 MB | 88,162 | 16,470 | SPMF [1] (FIMI) |
| Kosarak | `kosarak_quantities.txt.gz` | 19.1 MB | 990,002 | 41,270 | SPMF [1] (FIMI) |
| BMS-POS | `bms-pos_quantities.txt.gz` | 6.2 MB | 515,596 | 1,657 | KDD Cup 2000, Blue Martini Software [2] |
| Chainstore | `chainstore_quantities.txt.gz` | 24.3 MB | 1,112,949 | 46,086 | SPMF [1] (NU-MineBench) |

[1] P. Fournier-Viger et al. SPMF: a Java open-source pattern mining library. Journal of Machine
Learning Research 15:3569–3573, 2014.
[2] R. Kohavi et al. KDD-Cup 2000 organizers' report: peeling the onion. SIGKDD Explorations
2(2):86–93, 2000.

- `data/datasets/`: transactions with internal utilities (iu), one `item:iu` list per line;
- `data/utilities/`: external utilities (eu) of each dataset, and in `variants/` the tables of
  the utility-distribution experiment;
- `data/workloads/`: FWUP(Q, ξ), the candidate pool, S and ξ of every configuration.

The iu and eu values are integers in {1, …, 10}, generated once by the authors without a recorded
seed, so these files are the only way to reproduce the results. Kosarak has 27 transactions that
list an item twice; the loader keeps the last occurrence. Details and the references to cite:
[`data/SOURCES.md`](data/SOURCES.md).

`data/checksums.txt` lists two SHA-256 values for every data file: of its uncompressed content
and of the file as stored. The code checks the content value before reading a file and stops on
a mismatch. To check the stored files:

```bash
awk '!/^#/ {print $2 "  " $3}' data/checksums.txt | sha256sum -c --quiet && echo "all data files OK"
```

## Quick start

```bash
env PYTHONHASHSEED=0 python -m pytest -q -m "not slow"     # ~5 s: unit tests, paper example
env PYTHONHASHSEED=0 python scripts/running_example.py     # the paper's running example
env PYTHONHASHSEED=0 python experiments/run_experiments.py main_comparison --only x1_chess
python scripts/compare_results.py main_comparison           # 6/6 units identical
```

The chess runs of the main comparison take about 15 s in total. The full test suite
(`python -m pytest -q`) takes about 2 minutes.

## Reproducing the experiments

Each experiment has a configuration file in `experiments/configs/` and runs with one command:

```bash
env PYTHONHASHSEED=0 python experiments/run_experiments.py <experiment>
python scripts/compare_results.py <experiment>
```

How a run works:

- Runs (units) execute one after another, each in its own child process; nothing runs in
  parallel.
- Results go to `runs/<experiment>/`: one JSON file per run, Q' as a compressed diff, and
  `summary.csv`.
- An interrupted experiment resumes where it stopped.
- `--only <substring>` or `--keys <key,…>` run a subset of the units; `--list` lists them.

`compare_results.py` checks every non-runtime column and the SHA-256 of Q' against the
published files.

Each run has a wall-clock budget of 7,200 s. A run that exceeds it is marked `censored`, and its
quality metrics then depend on how far it got, which depends on the machine. Censored runs
therefore cannot be expected to reproduce. The published results contain 13 such runs (see the
table below), all of them greedy-mechanism or baseline runs.

| Experiment | Paper | Runs | Censored | Reference time* | Command |
|---|---|---:|---:|---:|---|
| main_comparison | §5.3.1, Tables 11–12, Fig. 2 | 42 | 0 | 0.3 h | `run_experiments.py main_comparison` |
| riswu_variants | §5.3.2, Table 13(a) | 28 | 0 | 0.1 h | `run_experiments.py riswu_variants` |
| ordering_control | §5.3.2, Table 13(a) | 7 | 0 | < 0.1 h | `run_experiments.py ordering_control` |
| reduction_mechanism | §5.3.2, Table 13(b) | 14 | 3 | 6.7 h | `run_experiments.py reduction_mechanism` |
| cap_ablation | §5.3.2, Table 13(c) | 16 | 0 | 0.1 h | `run_experiments.py cap_ablation` |
| threshold | §5.3.3, Table 14 | 192 | 1 | 5.2 h | `run_experiments.py threshold` |
| workload | §5.3.4, Table 15 | 504 | 0 | 1.6 h | `run_experiments.py workload` |
| utility_distribution | §5.3.5, Table 16 | 360 | 0 | 1.7 h | `run_experiments.py utility_distribution` |
| scalability | §5.3.6, Table 17, Fig. 3 | 384 | 0 | 2.0 h | `run_experiments.py scalability` |
| equal_margin | §5.3.7, Table 18 | 168 | 9 | 21.1 h | `run_experiments.py equal_margin` |
| threshold_lowering | §5.3.7, Table 18 | (analysis) | – | ~1 h | `scripts/reexposure.py <experiments>`, see below |
| ddi_correlation | §5.3.8 | (analysis) | – | seconds | `scripts/paper_assets.py --tables-only` |

\* Sum of hiding and evaluation time per run on the machine described above, plus a few seconds
per run for loading data. Most of the time of reduction_mechanism and equal_margin goes to
censored runs that run until the 7,200 s budget.

threshold_lowering mines the feasible Q' of earlier runs, so it needs a re-run of its input
experiments first (the published results contain only the SHA-256 of Q'):

```bash
env PYTHONHASHSEED=0 python scripts/reexposure.py main_comparison cap_ablation \
    reduction_mechanism riswu_variants ordering_control equal_margin
```

Experiments that run independently can be started in parallel, as separate processes with
different output directories.

The other parts of the paper:

| Paper | In this repository |
|---|---|
| §2.3 (literature search behind the comparison of hiding problems) | `literature_search/`, `scripts/literature_search.py` |
| §4.4 Algorithm 1, Fig. 1 | `hiding_sfwup/hiding.py`; `results/paper/figures/fig1_algorithm.*` |
| Tables 2–7 (running example, ξ = 0.4) | `scripts/running_example.py`, `tests/test_running_example.py` |
| §4.6 Example 4 (a hidden pattern rises above ξ again; database of Table 2, ξ = 0.6) | the final verification pass in `hiding_sfwup/hiding.py`; `tests/test_example4.py` |
| Table 8 (victim selection rules) | `hiding_sfwup/hiding.py` |
| Theorem 3 (final verification pass), further case | `tests/test_resurfacing.py` |
| §5.1 Setup: Tables 9 (datasets), 10 (main configuration); overview of the experiment design | `results/paper/tables/T_datasets.csv`, `T_main_config.csv`, `T_design.csv` |
| §5.2 Measures | `hiding_sfwup/metrics.py` |

## Regenerating tables and figures

The tables and figures of the paper are regenerated from `results/` (no experiment is run):

```bash
env PYTHONHASHSEED=0 python scripts/paper_assets.py                 # tables and figures
env PYTHONHASHSEED=0 python scripts/paper_assets.py --tables-only   # tables only
```

They are written to `results/paper/` (overwriting the published copies there); `--runs` reads
your re-run in `runs/` instead. Each table is one long-format CSV; its content is described in
[`results/paper/README_tables.md`](results/paper/README_tables.md).

| Paper | File in `results/paper/` |
|---|---|
| Table 9 | `tables/T_datasets.csv` |
| Table 10 | `tables/T_main_config.csv` |
| overview of the experiment design (no table in the paper) | `tables/T_design.csv` |
| Table 11 | `tables/T_main_quality.csv` |
| Table 12 | `tables/T_resources.csv` |
| Table 13(a)(b)(c) | `tables/T_ablation.csv` |
| Table 14 | `tables/T_threshold.csv` |
| Table 15 | `tables/T_workload.csv` |
| Table 16 | `tables/T_utility.csv` |
| Table 17 | `tables/T_scalability_slopes.csv` |
| values per database size (plotted in Fig. 3) | `tables/T_scalability.csv` |
| Table 18 | `tables/T_attack.csv` |
| §5.3.8 (no table) | `tables/T_ddi.csv` |
| numbers quoted in the text only | `tables/T_main_aux.csv`, `tables/T_scalability_early.csv` |
| Fig. 1 | `figures/fig1_algorithm.{pdf,eps,png}`; Graphviz source `figures/fig1_algorithm.dot` |
| Fig. 2 | `figures/F_time.{pdf,eps,png}` |
| Fig. 3 | `figures/F_scal.{pdf,eps,png}` |

The paper writes the threshold-lowering ratio as γ (attacker threshold (1 − γ)ξ; equal margin:
ξ_t = (1 − γ)ξ). The data and result files keep the original name `delta`: columns `delta`,
run variants `d05`, `d10`, `d20` = γ of 5, 10 and 20 %.

`results/paper/` holds the only tables and figures of this repository. Some inputs of
`paper_assets.py` are derived files that are already included in `results/`: dataset and workload
statistics in `results/tables/` (`*.csv`, with `*.json` copies), the RISWU victim agreement and the
re-exposure analysis. These scripts regenerate them:

| Script | Regenerates | Needs |
|---|---|---|
| `scripts/workload_stats.py` | dataset and workload statistics | data only |
| `scripts/riswu_agreement.py` | RISWU victim agreement | data only |
| `scripts/reexposure.py` | re-exposure after threshold lowering | Q' from a re-run |

## Outputs

`results/<experiment>/summary.csv` has one row per run (1,715 runs in total). The measures
follow §5.2:

- HF: hiding failure at ξ.
- `hf_target`: hiding failure at ξ_t.
- MC: missing cost.
- AC: artificial cost.
- IUS: itemset utility similarity (not capped at 1).
- DUS: database utility similarity.
- TMR: transaction modification ratio.
- DDI: share of transactions that lost an item.

`results/<experiment>/qprime_sha256.csv` gives the SHA-256 of each sanitized database Q', taken
over a canonical form: the diff {tid: {item: iu}} against Q, sorted.
`results/threshold_lowering/reexposure.jsonl` holds the threshold-lowering analysis per run, and
`results/riswu_variants/victim_agreement.json` the RISWU victim agreement.

The other columns of `summary.csv`:

| Column(s) | Meaning |
|---|---|
| `key`, `plan`, `calib`, `workload_id`, `method`, `variant`, `seed` | run identity: workload file, method, variant (`plan` = unit group of the experiment, see below) |
| `dataset`, `mult`, `xi`, `xi_target` | dataset, threshold multiplier, ξ, hiding target ξ_t (= ξ except equal_margin) |
| `mechanism`, `regular_step` | `swm` / `greedy-batch` / `greedy-unit` / `baseline`; regular SWM step `spec` or `cap` |
| `n_S`, `n_fwup`, `n_nsfwup` | \|S\|, \|FWUP(Q, ξ)\|, \|FWUP \\ S\| |
| `status`, `feasible`, `censored` | feasible ⇔ HF(ξ_t) = 0 and not censored |
| `hf_xi`, `hf_mined` | HF at ξ (= `hf`); HF recomputed by mining Q' (cross-check) |
| `rt_wall_total`, `rt_wall_hide`, `rt_cpu_hide`, `rt_phase1`, `cpu_*`, `eval_s` | wall time of Phase 1 + hiding, hiding only, CPU time; evaluation time (not in RT) |
| `rt_total`, `rt_hide` | the same as `rt_wall_total`, `rt_wall_hide` (older names) |
| `rss_before_hide_mb`, `peak_rss_hide_mb`, `delta_rss_hide_mb`, `peak_rss_*` | resident memory before / during hiding |
| `n_iter`, `n_regular_rounds`, `n_regular_iter`, `n_escalation_iter`, `n_tier1`, `n_tier2`, `n_cap_hit`, `n_rehide`, `n_final_rounds`, `n_wus_eval` | proposed method: iterations, Sub-Algorithm 4 tiers, MaxIter hits, final verification pass |
| `n_items_removed`, `n_deletions`, `n_steps`, `n_delete`, `n_decrease` | item deletions / modification steps |
| `removal_share`, `removal_share_alpha`, `red_delete`, `red_decrease`, `share_delete` | share of the reduction of N(X) (resp. of the gap N_X − ξD) due to deletions vs decrements |
| `n_pre_hidden`, `pct_pre_hidden`, `n_skip_removed`, `n_skip_removed_above` | SFWUP already below ξ_t at their turn |
| `t_riswu_share` | share of hiding time spent on RISWU (order + victim) |
| `threads_after_hide`, `children_after_hide` | threads and child processes of the run's process after hiding. No child process was started in any run. The thread count (12 on the 12-vCPU machine) includes NumPy's idle BLAS thread pool; CPU time ≈ wall time shows that the computation used one thread |
| `qc1` | result of the automatic consistency checks (`hiding_sfwup/checks.py`) |
| `calib_sha256` | SHA-256 of the workload file content |
| `code_tag`, `git_hash` | provenance of a run; empty in the published files (they referred to the authors' development repository) |

48 baseline runs use an older column layout: SMSE and MSU-MIU in the threshold experiment
with S re-selected (plan `x3_base`, 32 runs) and in the first scalability run (plan `x6`,
16 runs). Their times are in these columns only: `rt_total`
(sanitization time, same definition as `rt_wall_total`), `rt_hide` and `cpu_hide` (hiding loop,
same definitions as `rt_wall_hide`, `rt_cpu_hide`) and `cpu_total`. They have no
`rss_before_hide_mb`, `delta_rss_hide_mb` (additional memory) or `workload_id`, and their
`peak_rss_hide_mb` was read without resetting the peak after loading the data, so it is the peak
of loading and hiding together (as `peak_rss_load_hide_mb` of the other runs).

The runner also writes five `n_guard_*` columns, which the published files do not have (see
"Verification before release"). They count how often a decision fell inside the 10⁻⁹ band around ξ and was taken
exactly, and how often the exact result differed from floating point.

## Determinism

Everything except time and memory is deterministic and reproduces exactly: every quality metric,
every count and Q' itself. This holds with `PYTHONHASHSEED=0` and the pinned package versions.
All random choices use fixed seeds that are recorded in the configuration files and the workload
files: seeds 0–4 for the random selection of S and for the utility distributions, seeds 0–2 for
the database samples of the scalability experiment. Running time and memory depend on the
hardware; only their trends reproduce.

## Verification before release

This release takes every decision that falls within 10⁻⁹ of ξ in exact (rational) arithmetic:
the step of Proposition 3, the reduction amounts and the removal tests of the baselines, and
every comparison of wus with ξ. Before release, 458 of the published runs, including every run
of the main comparison and every MSU-MIU run, were re-executed with this code, and 75 runs (all
42 runs of the main comparison and 33 runs covering the other experiments) were re-executed
from this package itself. In every case, every sanitized database Q' and every measure except
time matched the published results, and no exact decision differed from the floating-point
one. Re-executed baseline runs on kosarak took up to about 1.35× the published times.

## Repository layout

```
hiding_sfwup/       algorithm, baselines, miner, metrics, workloads, runner
experiments/        configs/<experiment>.json, run_experiments.py
scripts/            paper_assets.py, paper_figures.py, compare_results.py, workload_stats.py,
                    riswu_agreement.py, reexposure.py, running_example.py, literature_search.py
data/               datasets/, utilities/, workloads/, checksums.txt, SOURCES.md
results/            <experiment>/summary.csv, qprime_sha256.csv; paper/ (tables and figures of
                    the paper); tables/ (derived dataset and workload statistics)
literature_search/  queries, records, eligibility review, PRISMA counts (see its README)
tests/              pytest suite (`-m "not slow"` skips the tests on real datasets)
```

The unit-group names (`plan`: `x1`, `x1_cap`, `x2`, …) and workload identifiers are the
identifiers of the original runs. They are kept because they appear in the published result
files and inside the checksummed workload files.

## Citation

Please cite the paper ("An Efficient Method for Hiding Sensitive Frequent Weighted Utility
Patterns", Duong M. Trung, Trinh D. D. Nguyen, Philippe Fournier-Viger, Bay Vo) and this software
(https://github.com/TrungDuong83/hiding-sfwup, release v1.0.0; see `CITATION.cff`). If you use the
datasets, please also cite their original sources (`data/SOURCES.md`).

## License

- Code: MIT (`LICENSE`).
- Data created by the authors: CC BY 4.0 (`LICENSE-DATA`): the internal and external utilities,
  the workload files, the results, and the screening decisions and counts of the literature
  search.
- The item lists of the datasets belong to their original sources; please cite them as listed in
  `data/SOURCES.md`.
- Bibliographic metadata in `literature_search/`: the terms of each source service (see
  `literature_search/README.md`).
