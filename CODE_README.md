# scGATES Review Package

This package contains the code and run instructions needed to review the current scGATES paper experiments. It intentionally excludes large data files, model weights, full result directories, and temporary diagnostic scripts.

## Purpose

scGATES aligns cell-side gene distribution sets \(D_g\) with text evidence sets \(P_g\). The current paper setting evaluates set-level denoising aggregation under noisy text evidence and paired-all-genes retrieval.

## Main Entry Point

The main training and evaluation entry point is:

```bash
python scripts/training/run_scgates_alignment.py
```

Core model code is in:

- `scgates/models.py`
- `scgates/stage1.py`
- `scgates/io.py`
- `scgates/losses.py`

The upstream Stage1 neighbor-prior module is packaged separately as:

- `stage1_neighbor_prior/`

This module is used to generate Stage1 neighbor or instance-score files. The
main scGATES trainer reads those precomputed files through
`--instance-score-npz` and `--eval-instance-score-npz`.

## Main Experiment Setting

- Dataset layer: strict text_min8
- Evaluation: paired-all-genes
- Text confidence mode: `text_only`
- Text confidence gate: `prior_sigmoid`
- Text confidence gate scale: `2.0`
- Stage1 instance score: real
- Instance score alpha: `0.3`
- Instance score mode: `signed`
- Text easy noise: train and eval, ratio `1.0`, max `64`, uniform prior

Main result:

| Method | R@1 | R@5 | R@10 | MRR |
| --- | ---: | ---: | ---: | ---: |
| Mean pooling | 0.464 | 0.692 | 0.765 | 0.570 |
| Full scGATES | 0.598 | 0.871 | 0.924 | 0.720 |

## Recommended Review Order

1. Read `METHOD_SUMMARY.md`.
2. Check required data paths in `DATA_MANIFEST.md`.
3. If reviewing Stage1 score generation, inspect `stage1_neighbor_prior/`.
4. Review training commands in `run_scripts/`.
5. Review current result tables in `expected_tables/`.
6. Check result locations in `RESULTS_MANIFEST.md`.

## Mini Smoke Test

The package includes a synthetic mini example for checking that the code can
generate small stores, run the training entry point, and summarize outputs:

```bash
bash run_scripts/99_run_mini_example.sh
```

The generated data are synthetic and only validate the pipeline format. They
are not used for paper claims.

## Missing Or Optional Files

This package does not include:

- model weights
- `.store` directories
- `.npz` data files
- full `results/` directories
- raw single-cell files
- raw text corpora

These files must be prepared externally according to `DATA_MANIFEST.md`.

