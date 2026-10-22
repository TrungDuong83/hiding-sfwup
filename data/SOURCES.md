# Data sources

Every file in `data/` is the exact input used for the experiments of the paper. The SHA-256 of
each file's content (after `gunzip` for `.gz` files) is in `data/checksums.txt`. The code
checks it before reading the file.

## Datasets (`data/datasets/*_quantities.txt.gz`)

Each line is one transaction, written as `item:iu` tokens separated by blanks.

| Dataset | File | Transactions | Distinct items | Origin of the item lists |
|---|---|---:|---:|---|
| chess | `chess_fimi_quantities.txt.gz` | 3,196 | 75 | SPMF; prepared from the UCI chess dataset (source: FIMI repository) |
| mushroom | `mushroom_quantities.txt.gz` | 8,416 | 119 | SPMF; prepared from the UCI mushroom dataset (source: FIMI repository) |
| accident | `accident_quantities.txt.gz` | 340,183 | 468 | SPMF; anonymized traffic accident data (source: FIMI repository) |
| retail | `retail_quantities.txt.gz` | 88,162 | 16,470 | SPMF; customer transactions from an anonymous Belgian retail store (source: FIMI repository) |
| kosarak | `kosarak_quantities.txt.gz` | 990,002 | 41,270 | SPMF; click-stream data from a Hungarian news portal (source: FIMI repository) |
| bms-pos | `bms-pos_quantities.txt.gz` | 515,596 | 1,657 | BMS-POS point-of-sale data provided by Blue Martini Software for KDD Cup 2000 (Kohavi et al., 2000) |
| chainstore | `chainstore_quantities.txt.gz` | 1,112,949 | 46,086 | SPMF; customer transactions from a major grocery store chain in California, USA (from NU-MineBench) |

SPMF datasets page: https://www.philippe-fournier-viger.com/spmf/index.php?link=datasets.php
(chess, mushroom, accident, retail, kosarak, chainstore). Please cite SPMF as Fournier-Viger et
al. (2014).

BMS-POS: point-of-sale data provided by Blue Martini Software for KDD Cup 2000 (Kohavi et al.,
2000; also used as a benchmark by Zheng, Kohavi and Mason, 2001). No download URL was recorded.
The item lists come from a horizontal CSV file `BMS-POS.csv` (one transaction per line, item IDs
0..1656 separated by commas; 515,596 transactions, 1,657 items; 11,900,318 bytes, SHA-256
42bfb6f906c20996d1b80181d5c9d1f1f31e2b532d1446ca75f20004c33d1392). It was converted to one
blank-separated line per transaction (line ends and trailing commas removed), and iu values were
added as for the other datasets. That CSV file is not distributed; the converted file used in
this study is `data/datasets/bms-pos_quantities.txt.gz` (content SHA-256
61b9dbf573b1da7e2cd2b63c225333728842bcad44a712f35774a3ed29498f87).

**The item lists come from the original datasets.** The internal utilities (iu, the quantity of
each item in each transaction) were generated at random by the authors as integers 1..10. No seed
was recorded, so the files in this repository are the only way to reproduce the results. If you
use these files, please also cite the original sources of the datasets listed above.

Kosarak has 27 transactions in which an item occurs twice (with an iu at each occurrence). The
loader (`hiding_sfwup.data.load_transactions`) keeps the last occurrence; the transaction and item
counts above are those after loading.

## External utilities (`data/utilities/`)

- `<dataset>_weights.txt`: one `item:eu` per line. The eu values were generated at random by the
  authors as integers 1..10, with no recorded seed. These are the tables used by every experiment
  except the utility-distribution experiment.
- `variants/<dataset>_<distribution>_seed<k>.txt`: the tables of the utility-distribution
  experiment. They were drawn with `numpy.random.default_rng(k)`, one value per item in ascending
  item order (see `hiding_sfwup.workloads.make_weights`):
  - `uniform`: integers on {1..10};
  - `normal`: `rint(N(5.5, 2.0²))` clipped to [1, 10];
  - `lognormal10`: `rint(lognormal(1.0, 0.8))` clipped to [1, 10].

## Workloads (`data/workloads/<id>.json.gz`)

Each file holds FWUP(Q, ξ) with the wus of every pattern, the candidate pool, the selected SFWUP
set S and ξ for one configuration. These are the exact inputs of the runner. Their identifiers
(`x1_<dataset>`, `x3_<dataset>_<multiplier>`, …) are the original run identifiers, and they also
appear in the result files (columns `calib`, `key` and `workload_id`). The table
"Experiments ↔ paper" in `README.md` maps each prefix to its experiment. Regenerate the files with
`python -m hiding_sfwup.workloads <experiment>`. Only the timing field `mine_s` and the
`xc` cross-check record (made against an independent miner that is not included) can differ.

## References

- P. Fournier-Viger, A. Gomariz, T. Gueniche, A. Soltani, C.-W. Wu, V. S. Tseng. SPMF: a Java
  open-source pattern mining library. Journal of Machine Learning Research 15(104):3569–3573,
  2014. https://jmlr.org/papers/v15/fournierviger14a.html
- R. Kohavi, C. E. Brodley, B. Frasca, L. Mason, Z. Zheng. KDD-Cup 2000 organizers' report: peeling
  the onion. ACM SIGKDD Explorations Newsletter 2(2):86–93, 2000. https://doi.org/10.1145/380995.381033
- Z. Zheng, R. Kohavi, L. Mason. Real world performance of association rule algorithms. In
  Proceedings of KDD 2001, 401–406. https://doi.org/10.1145/502512.502572
