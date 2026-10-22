# Paper tables and figures

Generated from the published results (no experiment is run) by

```bash
env PYTHONHASHSEED=0 python scripts/paper_assets.py     # tables + figures
env PYTHONHASHSEED=0 python scripts/paper_assets.py --tables-only
```

The script reads `results/<experiment>/summary.csv`, the analysis files in `results/` and, for
T_datasets only, `data/datasets/`. `--runs` reads a re-run in `runs/` instead of `results/`.

Rules for every table:

- One long-format CSV per table. Values are not rounded, except the display columns of T_datasets.
- `feasible` = 1 if HF = 0 (at ξ; equal margin: at ξ_t) and the run is not censored (7 200 s
  budget); `censored` = 1 if the run hit the budget. Infeasible runs keep their values and carry
  `mark` = †.
- `best_<metric>` = 1 marks the best value among the feasible runs of the same cell. Lower is better
  for HF, AC, MC, TMR, DDI and times; higher is better for IUS and DUS. `tie_<metric>` = 1 when more
  than one feasible run of the cell reaches that value.
- Method names and order: SC-only, SDIF, Max-RISWU, Min-RISWU, SMSE, MSU-MIU. Ablation names:
  greedy-batch, greedy-unit, SWM without removal, Max/Min-RISWU without twu, Max/Min-RISWU without
  Wt, SC-only ordered by wus.
- Measures: sanitization time = wall time of Phase 1 + hiding loop (`rt_wall_total`); hiding-loop
  time = `rt_wall_hide`; CPU time = `cpu_total`; peak resident memory during sanitization =
  `peak_rss_hide_mb` (VmHWM since the reset before Phase 1); additional memory used by
  sanitization = `delta_rss_hide_mb`.
- Columns with a `_mean`, `_min` and `_max` suffix aggregate over seeds, draws or orders, with
  `n_feasible` of `n_runs`.

## Tables

| File | Content |
|---|---|
| `T_datasets.csv` | 7 datasets: #transactions, #items, average length (2 decimals) and maximum length (items per transaction after parsing), density (% = avg. length / #items × 100, 3 decimals; unrounded values in `*_unrounded`), type, #transactions with a repeated item. |
| `T_main_config.csv` | Main comparison per dataset: ξ, \|FWUP\|, #candidates, \|S\|, mean length of FWUPs and of S, % transactions supporting at least one pattern of S. |
| `T_design.csv` | Experimental design: datasets, varied factor, selection of S, seeds/orders, runs per method and per role (our method, baselines, greedy, SWM without removal) per experiment; total rows, including the runs used for the DDI correlation. |
| `T_main_quality.csv` | Main comparison, dataset × method: HF, AC, MC, IUS, DUS, TMR, DDI, feasible, censored, best/tie per metric. |
| `T_main_aux.csv` | (not printed) Main comparison: regular rounds, % of S already hidden at its turn, removal share, share of the reduction (and, for the baselines, of the modifications) that only lowers a quantity, \|TMR − DDI\|, MC rank among the four strategies with ties. |
| `T_resources.csv` | Main comparison, dataset × method: sanitization time, hiding-loop time, CPU time, CPU/wall of the hiding loop, peak resident memory, additional memory, share of the hiding-loop time spent computing RISWU (our strategies). |
| `T_ablation.csv` | Panel a (victim selection): the four strategies; RISWU variants without twu / without Wt (same-victim %, metrics, \|Δ\| to the full RISWU); SC-only ordered by wus (MC, ΔMC); Spearman ρ(twu, Wt), median over the supporting transactions of each pattern of S. Panel b (modification): SWM (SDIF), greedy-batch, greedy-unit; SWM without removal (4 datasets × 4 strategies) with its time ratio to the main run, Tier-1 calls and re-hide calls. |
| `T_threshold.csv` | Panel task: dataset × ξ multiplier × mode (fixed S / re-selected S): ξ, \|FWUP\|, \|NSFWUP\|, #candidates, \|S\|, mean lengths, % transactions supporting S. Panel results: the same cells × method with quality metrics, sanitization time, best/tie. |
| `T_workload.csv` | Dataset × selection (high overlap, low overlap, random) × proportion of S (10, 25, 50 %) × method; random = 5 seeds (mean, min, max, n_feasible). `best_mc_cells` counts the cells (one per S) in which the method has the lowest MC among the feasible runs (ties included). |
| `T_utility.csv` | Chess and Kosarak × external-utility distribution (uniform, normal, log-normal) × mode (S re-selected / S fixed) × method, over 5 draws; rows "original utilities" repeat the main comparison. |
| `T_scalability.csv` | Kosarak and Chainstore × sample size (10–100 %) × method, over 3 orders: \|Q\|, % transactions supporting S, sanitization time, CPU time, hiding-loop time, peak and additional memory. |
| `T_scalability_slopes.csv` | Log–log OLS slope of each measure on \|Q\| (95 % CI from t, R², n) per method × dataset × measure × scope (all runs / feasible runs); runs with a non-positive value are left out of the fit. |
| `T_scalability_early.csv` | (not printed) Earlier scalability run: 4 sizes, one order, SDIF and the baselines. |
| `T_attack.csv` | Panel a (lower threshold, feasible Q′ of the main comparison and of SWM without removal): per method group and δ = 5, 10, 20 %, re-exposed share r (mean, min, max), #new patterns m, share of S among the new patterns q, n, and the range over runs of the median wus(X)/ξ over S. Panel b (equal margin): dataset × δ × 8 methods with HF at ξ_t and at ξ, quality at ξ, time, r at δ = 20 %. Panel b_counts: cells where the best of our strategies has MC ≤ each comparator (both feasible), and our runs identical to the main comparison. |
| `T_ddi.csv` | (not printed) Spearman correlation of DDI with MC, IUS and TMR: all runs, our method, baselines. |

## Figures

Each figure is written as vector PDF, EPS and PNG (1000 dpi) at print size, with TrueType fonts
embedded and text of at least 7 pt.

| File | Content |
|---|---|
| `F_time.*` | Sanitization time of the main comparison (log scale), by dataset and method; hollow marker = infeasible run (HF > 0). 190 mm wide. |
| `F_scal.*` | Sanitization time against \|Q\| (log–log), Kosarak and Chainstore; line = mean over 3 orders, band = min–max; hollow marker = at least one infeasible order. 190 mm wide. |
| `fig1_algorithm.*` | Block diagram of Algorithm 1 and its procedure HideOne, with line numbers. 140 mm wide. `fig1_algorithm.dot` is the Graphviz source of the same diagram (`dot -Tpdf`). |
