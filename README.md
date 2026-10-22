# scGATES Review Package

This package contains the code and run instructions needed to review the current scGATES paper experiments. It intentionally excludes large data files, model weights, full result directories, and temporary diagnostic scripts to comply with double-blind review requirements.

---

## 1. Overview

scGATES aligns cell-side gene distribution sets \(D_g\) with text evidence sets \(P_g\) by learning a shared retrieval space. Each gene is represented by two sets:

- \(D_g\): cell-side distribution set built from contextual cell or gene embeddings.
- \(P_g\): text evidence set built from AD-Alterome evidence sentences.

The model evaluates whether the matched \(D_g\) and \(P_g\) rank each other highly among candidate genes. The current paper setting evaluates set-level denoising aggregation under noisy text evidence and paired-all-genes retrieval.

### Main Claims Supported By Current Results

- Full scGATES outperforms mean pooling under noisy text evidence.
- Text-side denoising is the most important ablation factor.
- Cell-side denoising and Stage1 instance scores provide smaller auxiliary effects.
- Unmasked gene names produce much higher retrieval scores, so masked text is used as the conservative main setting.

---

## 2. Directory Structure

```
scGATES_Codes/
├── scgates/                    # Core Python library
│   ├── __init__.py
│   ├── models.py               # Core model definitions (set encoder, alignment)
│   ├── stage1.py               # Stage1 instance-score integration
│   ├── io.py                   # Data I/O, store reading/writing
│   ├── losses.py               # Training loss functions
│   ├── metrics.py              # Evaluation metrics (Recall@K, MRR)
│   ├── retrieval.py            # Retrieval and ranking operations
│   ├── sampling.py             # Sampling / batching utilities
│   ├── structure.py            # Data structure definitions
│   ├── text.py                 # Text-side processing and confidence weighting
│   ├── training.py             # Training loop utilities
│   ├── neighbor_prior.py       # Neighbor prior integration
│   ├── stage3_confusion.py     # Confusion analysis utilities
│   └── ot.py                   # Optimal transport utilities
│
├── scripts/                    # Executable scripts
│   ├── training/
│   │   └── run_scgates_alignment.py   # [MAIN ENTRY] scGATES trainer
│   ├── data_preprocessing/
│   │   ├── split_text_evidence_store.py        # Split text stores into train/eval
│   │   ├── build_exact_unmasked_textsplit_store.py  # Build unmasked text variant stores
│   │   ├── build_gene_identifier_replacement_tsv.py  # Generate gene alias mappings
│   │   ├── encode_ad_alterome_text_set.py       # Encode AD-Alterome text
│   │   ├── encode_easy_noise_bank.py            # Encode synthetic noise text
│   │   ├── make_synthetic_easy_noise_bank.py    # Generate synthetic noise bank
│   │   └── export_text_evidence_store.py        # Export text evidence stores
│   ├── summarization/
│   │   ├── summarize_run_metrics_table.py       # Aggregate run metrics
│   │   ├── summarize_text_noise_weights.py      # Analyze text noise weights
│   │   ├── summarize_text_noise_by_gene.py      # Per-gene text noise analysis
│   │   └── summarize_cell_noise_weights.py      # Analyze cell noise weights
│   └── application/
│       ├── build_gene_evidence_application_table.py  # Build downstream application tables
│       ├── export_gene_evidence_explanations.py       # Export gene-evidence explanations
│       ├── run_evidence_gap_analysis.py               # Evidence gap analysis
│       └── run_confidence_gap_analysis.py             # Confidence gap analysis
│
├── stage1_neighbor_prior/      # Upstream Stage1 neighbor-prior module (standalone)
│   ├── __init__.py
│   ├── train.py                # Train Geneformer-native neighbor alignment
│   ├── model.py                # Contrastive neighbor-prior model
│   ├── data.py                 # Data loading and processing
│   ├── export_prior.py         # Export trained prior to .npz
│   ├── config.example.yaml     # Example configuration
│   └── README.md
│
├── run_scripts/                # Shell scripts for experiment pipeline
│   ├── 00_prepare_text_stores.sh              # Prepare text evidence stores
│   ├── 01_run_main_full_scgates.sh            # Run main full scGATES
│   ├── 02_run_main_ablation.sh                # Run ablation study
│   ├── 03_run_text_noise_ratio_sweep.sh       # Text noise ratio sweep
│   ├── 04_run_cell_text_noise_ratio_sweep.sh  # Cell+text noise ratio sweep
│   ├── 05_run_mask_identifier_ablation.sh     # Mask/unmasked/identifier ablation
│   └── 06_summarize_all_results.sh            # Summarize all results
│
├── configs/                    # YAML configuration files
│   ├── main_strict_text_min8_full_scgates.yaml  # Main experiment config
│   ├── ablation_strict_text_min8.yaml            # Ablation study configs
│   ├── noise_ratio_sweep.yaml                    # Noise ratio sweep configs
│   └── gene_identifier_ablation.yaml             # Gene identifier ablation configs
│
├── example_data/               # Mini synthetic data for smoke testing
│   ├── README.md
│   ├── text/
│   │   ├── real_ad_alterome_sample.tsv     # Real AD-Alterome text sample
│   │   ├── mini_ad_alterome.train.tsv      # Synthetic train text
│   │   └── mini_ad_alterome.eval.tsv       # Synthetic eval text
│   └── stores/
│       ├── gene_distributions.train.store/ # Cell-side distribution store (train)
│       ├── gene_distributions.eval.store/  # Cell-side distribution store (eval)
│       ├── text_train.store/               # Text evidence store (train)
│       ├── text_eval.store/                # Text evidence store (eval)
│       ├── stage1_instance_scores.train.npz # Stage1 scores (train)
│       └── stage1_instance_scores.eval.npz  # Stage1 scores (eval)
│
├── example_outputs/            # Expected output from mini smoke test
│   ├── mini_scgates_smoke_metrics.csv
│   └── mini_scgates_smoke_metrics.md
│
├── expected_tables/            # Expected result tables from full experiments
│   ├── main_ablation_table.md               # Ablation study results
│   ├── multiseed_table.md                   # Multi-seed robustness results
│   ├── mask_vs_unmasked_table.md            # Mask vs unmasked comparison
│   ├── gene_identifier_ablation_table.md    # Gene identifier variants
│   ├── text_noise_ratio_table.md            # Text noise ratio sweep
│   └── cell_text_noise_ratio_table.md       # Cell+text noise ratio sweep
│
├── METHOD_SUMMARY.md           # Concise method overview
├── DATA_MANIFEST.md            # Required external data files
├── RESULTS_MANIFEST.md         # Expected result file locations
└── README.md                   # This file
```

---

## 3. Method Summary

### Pipeline Components

1. **Upstream Stage1 neighbor-prior estimation** (`stage1_neighbor_prior/`)
   - Geneformer-native: reads cell embedding JSONL and trains a contrastive model between target gene contextual embeddings and neighbor gene embeddings.
   - Produces gene-neighbor prior or instance-score `.npz` files.
   - Current main runs consume precomputed Stage1 files rather than recomputing them.

2. **Set-based \(D_g\)-\(P_g\) alignment** (`scripts/training/run_scgates_alignment.py`)
   - Main scGATES trainer. The file name is historical; it is the current primary training entry point.
   - Learns a shared retrieval space where matched cell-side and text-side gene representations rank each other highly.

3. **Text-side confidence weighting** with `text_only` confidence mode.

4. **Prior-gated text denoising** through `prior_sigmoid` gating.

5. **Cell-side denoising** through the cell set encoder.

6. **Stage1 instance-score guidance** as an auxiliary cell-side signal.

### Main Experiment Setting

| Parameter | Value |
|---|---|
| Dataset layer | strict text_min8 |
| Evaluation | paired-all-genes |
| Text confidence mode | `text_only` |
| Text confidence gate | `prior_sigmoid` |
| Text confidence gate scale | `2.0` |
| Stage1 instance score | real |
| Instance score alpha | `0.3` |
| Instance score mode | `signed` |
| Text easy noise | train and eval, ratio `1.0`, max `64`, uniform prior |

---

## 4. Core Library (`scgates/`)

The `scgates` package provides all foundational modules:

| Module | Description |
|---|---|
| `models.py` | Core model definitions: set encoder, alignment module, denoising layers |
| `stage1.py` | Stage1 instance-score integration into the alignment pipeline |
| `io.py` | Reading/writing gene distribution stores, text evidence stores, and NPZ files |
| `losses.py` | Training objectives including contrastive loss and noise-confidence regularization |
| `metrics.py` | Evaluation: Recall@K, Mean Reciprocal Rank (MRR), confidence-filtered variants |
| `retrieval.py` | Candidate retrieval and ranking within the shared embedding space |
| `sampling.py` | Gene-wise and cell-wise sampling strategies |
| `structure.py` | Dataclass definitions for stores and configuration objects |
| `text.py` | Text evidence encoding, confidence weighting, and text-side denoising |
| `training.py` | Training loop, checkpointing, and logging utilities |
| `neighbor_prior.py` | Loading and applying frozen neighbor priors |
| `stage3_confusion.py` | Confusion matrix analysis for result interpretation |
| `ot.py` | Optimal transport utilities for distribution matching |

---

## 5. Scripts (`scripts/`)

### 5.1 Training — Main Entry Point

**`scripts/training/run_scgates_alignment.py`**

The primary training script. Despite its name (historical), this is the current main scGATES trainer. It:

- Loads cell-side gene distribution stores and text evidence stores.
- Optionally loads Stage1 instance-score NPZ files.
- Trains the set-level alignment model.
- Evaluates recall and MRR metrics.
- Outputs metrics files and model checkpoints.

### 5.2 Data Preprocessing Scripts

| Script | Purpose |
|---|---|
| `split_text_evidence_store.py` | Splits full text evidence store into masked strict train/eval splits |
| `build_exact_unmasked_textsplit_store.py` | Builds unmasked (exact gene name) and EntrezID/random-alias variant stores |
| `build_gene_identifier_replacement_tsv.py` | Generates gene alias mapping (e.g., EntrezID, random aliases) |
| `encode_ad_alterome_text_set.py` | Encodes AD-Alterome evidence sentences into embedding stores |
| `encode_easy_noise_bank.py` | Encodes synthetic easy-noise text for noise experiments |
| `make_synthetic_easy_noise_bank.py` | Generates synthetic noise text from a gene vocabulary |
| `export_text_evidence_store.py` | Exports encoded text evidence into the store format |

### 5.3 Summarization Scripts

| Script | Purpose |
|---|---|
| `summarize_run_metrics_table.py` | Aggregates metrics from multiple runs into comparison tables |
| `summarize_text_noise_weights.py` | Analyzes learned text-side noise confidence weights |
| `summarize_text_noise_by_gene.py` | Per-gene breakdown of text noise confidence |
| `summarize_cell_noise_weights.py` | Analyzes learned cell-side noise weights |

### 5.4 Application Scripts

| Script | Purpose |
|---|---|
| `build_gene_evidence_application_table.py` | Builds downstream application-ready gene-evidence tables |
| `export_gene_evidence_explanations.py` | Exports human-readable gene-evidence explanations |
| `run_evidence_gap_analysis.py` | Runs evidence gap analysis for genes with low text support |
| `run_confidence_gap_analysis.py` | Analyzes confidence gap between matched and mismatched evidence |

---

## 6. Run Scripts (`run_scripts/`)

Shell scripts that implement the full experiment pipeline. Each script calls `run_scgates_alignment.py` with appropriate configuration YAMLs.

| Script | Description |
|---|---|
| `00_prepare_text_stores.sh` | Prepares masked/unmasked text evidence stores before training |
| `01_run_main_full_scgates.sh` | Runs the main Full scGATES experiment |
| `02_run_main_ablation.sh` | Runs ablation: w/o cell denoising, w/o text denoising, w/o both, w/o Stage1 |
| `03_run_text_noise_ratio_sweep.sh` | Sweeps text noise ratio from 10% to 60% |
| `04_run_cell_text_noise_ratio_sweep.sh` | Sweeps cell noise ratio from 10% to 60% (text noise fixed near 30%) |
| `05_run_mask_identifier_ablation.sh` | Runs gene identifier variants: masked, unmasked, EntrezID, random alias |
| `06_summarize_all_results.sh` | Calls summarization scripts to generate result tables |

---

## 7. Configuration Files (`configs/`)

YAML files defining experiment parameters. Key configs:

| Config | Purpose |
|---|---|
| `main_strict_text_min8_full_scgates.yaml` | Main Full scGATES experiment settings |
| `ablation_strict_text_min8.yaml` | Ablation study: controlling which components are active |
| `noise_ratio_sweep.yaml` | Defines noise ratio sweep ranges for text and cell noise |
| `gene_identifier_ablation.yaml` | Defines gene identifier variations for ablation |

---

## 8. Stage1 Neighbor Prior Module (`stage1_neighbor_prior/`)

A standalone Geneformer-native module for upstream Stage1 neighbor-prior estimation. It:

- Reads Geneformer cell embedding JSONL files.
- Trains a contrastive model between target gene contextual embeddings and their neighbor gene embeddings in the same cell.
- Exports a frozen prior `.npz` file (one normalized vector per gene).
- The current main scGATES trainer reads these precomputed files via `--instance-score-npz` and `--eval-instance-score-npz`.

**Key files:**

| File | Description |
|---|---|
| `train.py` | Training entry point |
| `model.py` | Contrastive model definition |
| `data.py` | Data loading from JSONL and neighbor CSV |
| `export_prior.py` | Export trained checkpoint to `.npz` prior file |
| `config.example.yaml` | Example training configuration |

---

## 9. Example Data (`example_data/`)

Contains a synthetic mini dataset for smoke testing the pipeline. The bundled text preview (`real_ad_alterome_sample.tsv`) is real AD-Alterome evidence. The runnable mini stores (`stores/`) are generated by sampling from real project stores.

**Purpose:** Validate file formats, code execution, and pipeline integrity. These data are not used for paper claims.

**To regenerate:** `bash run_scripts/99_run_mini_example.sh`

---

## 10. Example Outputs (`example_outputs/`)

Expected outputs from the mini smoke test:

- `mini_scgates_smoke_metrics.csv` — CSV metrics table (e.g., R@1, R@5, MRR)
- `mini_scgates_smoke_metrics.md` — Same metrics in markdown format

These files demonstrate the expected output format after running the mini smoke test.

---

## 11. Expected Result Tables (`expected_tables/`)

Pre-computed result tables from full experiments for reference. These tables document the paper's key quantitative claims.

### 11.1 Main Results — Ablation Study (`main_ablation_table.md`)

| Model setting | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| Mean pooling | 0.464 | 0.692 | 0.765 | 0.570 |
| Full scGATES | **0.598** | **0.871** | **0.924** | **0.720** |
| w/o cell-side denoising | 0.594 | 0.870 | 0.927 | 0.717 |
| w/o text-side denoising | 0.414 | 0.742 | 0.832 | 0.564 |
| w/o cell and text denoising | 0.427 | 0.741 | 0.824 | 0.570 |
| w/o Stage1 | 0.589 | 0.873 | 0.921 | 0.718 |

Full scGATES achieves the highest R@1 (0.598) and MRR (0.720). Text-side denoising is the most important component (R@1 drops from 0.598 to 0.414 when removed).

### 11.2 Multi-seed Robustness (`multiseed_table.md`)

| Seed | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| 42 | 0.598 | 0.871 | 0.924 | 0.720 |
| 43 | 0.570 | 0.862 | 0.929 | 0.698 |
| 44 | 0.553 | 0.852 | 0.906 | 0.682 |
| Average +/- std | 0.574 +/- 0.023 | 0.862 +/- 0.010 | 0.920 +/- 0.012 | 0.700 +/- 0.019 |

### 11.3 Mask vs Unmasked (`mask_vs_unmasked_table.md`)

| Text setting | Seed | R@1 | MRR |
|---|---|---|---|
| Masked | 42 | 0.598 | 0.720 |
| Unmasked | 42 | **0.815** | **0.888** |

Unmasked gene names produce substantially higher scores. Masked text is the conservative main setting to avoid gene name leakage.

### 11.4 Gene Identifier Ablation (`gene_identifier_ablation_table.md`)

| Text setting | R@1 | MRR |
|---|---|---|
| Gene masked | 0.598 | 0.720 |
| Gene unmasked | 0.815 | 0.888 |
| Gene to EntrezID | 0.676 | 0.787 |
| Gene to random alias | 0.600 | 0.725 |

EntrezID conversion preserves partial signal; random aliases perform near the masked baseline.

### 11.5 Text Noise Ratio Sweep (`text_noise_ratio_table.md`)

R@1 degrades from 0.614 at 10% noise to 0.474 at 60% noise, demonstrating the model's robustness under increasing text-side noise.

### 11.6 Cell + Text Noise Ratio Sweep (`cell_text_noise_ratio_table.md`)

R@1 degrades from 0.598 (no cell noise) to 0.268 (60% cell noise), showing cell-side noise impact with text noise fixed near 30%.

---

## 12. Required External Data (`DATA_MANIFEST.md`)

Large data files that **must be prepared externally** before running experiments. These are **not included** in this review package.

### 12.1 Core Required Files

| Path | Purpose | Generated by |
|---|---|---|
| `data/gene_distributions.train_donor.stage1.store` | Train cell-side gene distribution store | Cell embedding preprocessing |
| `data/gene_distributions.eval_donor.stage1.store` | Eval cell-side gene distribution store | Cell embedding preprocessing |
| `data/benchmarks/v2_textsplit_strict/text/train_text50_strict.store` | Masked strict train text evidence store | `split_text_evidence_store.py` |
| `data/benchmarks/v2_textsplit_strict/text/eval_text50_strict.store` | Masked strict eval text evidence store | `split_text_evidence_store.py` |
| `data/stage1_instance_scores.train_donor.allgenes.real.npz` | Train Stage1 instance scores | Stage1 preprocessing |
| `data/stage1_instance_scores.eval_donor.allgenes.real.npz` | Eval Stage1 instance scores | Stage1 preprocessing |
| `results/textsplit_consistency/gene_sets/strict_text_min8_genes.txt` | Gene allowlist | `make_text_evidence_gene_sets.py` |

### 12.2 Noise Experiment Files

| Path | Purpose | Generated by |
|---|---|---|
| `data/noise_text/easy_random_text_train_v2_1000.pubmedbert.npz` | Encoded easy-noise bank (train) | `encode_easy_noise_bank.py` |
| `data/noise_text/easy_random_text_eval_v2_1000.pubmedbert.npz` | Encoded easy-noise bank (eval) | `encode_easy_noise_bank.py` |

### 12.3 Gene Identifier Variant Files

| Path | Purpose | Generated by |
|---|---|---|
| `data/gene_id_maps/all_train_eval_genes_EntrezID.csv` | Gene to EntrezID mapping | External mapping |
| `data/gene_id_maps/random_gene_alias_seed42.csv` | Gene to random alias mapping | `build_gene_identifier_replacement_tsv.py` |

### 12.4 Unmasked Text Variant Stores

| Path | Purpose | Generated by |
|---|---|---|
| `data/benchmarks/v2_textsplit_strict_unmasked_exact/text/*.store` | Exact unmasked strict split stores | `build_exact_unmasked_textsplit_store.py` |
| `data/benchmarks/v2_textsplit_strict_entrezid_exact/text/*.store` | Exact EntrezID strict split stores | `build_exact_unmasked_textsplit_store.py` |
| `data/benchmarks/v2_textsplit_strict_randomgene_exact/text/*.store` | Exact random-alias strict split stores | `build_exact_unmasked_textsplit_store.py` |

### 12.5 Files NOT Included in This Package

- Model weights
- `.store` directories
- `.npz` data files
- Full `results/` directories
- Raw single-cell files
- Raw text corpora

---

## 13. Expected Result Locations (`RESULTS_MANIFEST.md`)

After running experiments, results are expected at these locations (not included in this package):

| Result group | Relative location |
|---|---|
| Main full scGATES | `results/paired_eval_all_genes_ablation/strict_text_min8/` |
| Main ablation | `results/paired_eval_all_genes_ablation/strict_text_min8/` |
| Multi-seed full scGATES | `results/paired_eval_all_genes_multiseed/strict_text_min8/` |
| Mask vs unmasked | `results/mask_ablation/strict_text_min8/` |
| Gene identifier ablation | `results/gene_identifier_ablation/strict_text_min8/` |
| Text noise ratio sweep | `results/text_noise_target_ratio_sweep/strict_text_min8/` |
| Cell and text noise ratio sweep | `results/cell_noise_target_ratio_sweep/strict_text_min8/` |

---

## 14. Mini Smoke Test

The package includes a synthetic mini example to validate that the code can generate small stores, run the training entry point, and summarize outputs:

```bash
bash run_scripts/99_run_mini_example.sh
```

This creates data under `example_data/stores/` and produces output in `example_outputs/`. The generated data are synthetic and only validate the pipeline format — they are **not** used for paper claims.

---

## 15. Recommended Review Order

1. Read this README for the overall structure.
2. Review `METHOD_SUMMARY.md` for concise method description.
3. Check required data paths in `DATA_MANIFEST.md`.
4. If reviewing Stage1 score generation, inspect `stage1_neighbor_prior/`.
5. Review experiment commands in `run_scripts/` and their configs in `configs/`.
6. Check current expected result tables in `expected_tables/`.
7. Check result locations in `RESULTS_MANIFEST.md`.
8. Run the mini smoke test to validate code execution.

---

## 16. Stage Naming Note

The current main scGATES cell-text alignment trainer is `scripts/training/run_scgates_alignment.py`. The upstream Stage1 score generator is packaged under `stage1_neighbor_prior/`. The "Stage1" naming in the code refers to the upstream neighbor-prior module; "Stage2" and "Stage3" terminology is historical and not used in the current pipeline.

## Supplementary Materials

- [Supplementary appendix](supplementary/Appendix-scGATES.pdf)
- [S1: application confidence gap scores](supplementary/S1.Application_confidence_gap_scores.csv)
- [S2: application gene evidence sentences](supplementary/S2.Application_gene_evidence_sentences.xlsx)
- [Results for more diseases](supplementary/scGATES_Results_for_more_diseases.zip)
- [Original reproduction code archive](supplementary/scGATES_Codes_for_reproduce.rar)
- [Original code package README](CODE_README.md)