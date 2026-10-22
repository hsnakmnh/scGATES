# scGATES

This repository contains the scGATES implementation, experiment commands, and a small input-data example. Prepare full datasets and model weights separately using the data documentation.

---

## 1. Overview

scGATES aligns cell-side gene distribution sets \(D_g\) with text evidence sets \(P_g\) by learning a shared retrieval space. Each gene is represented by two sets:

- \(D_g\): cell-side distribution set built from contextual cell or gene embeddings.
- \(P_g\): text evidence set built from AD-Alterome evidence sentences.

The model evaluates whether the matched \(D_g\) and \(P_g\) rank each other highly among candidate genes. The current paper setting evaluates set-level denoising aggregation under noisy text evidence and paired-all-genes retrieval.

---

## 2. Directory Structure

```
scGATES/
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
├── example_data/               # Small real-data input subset
│   ├── README.md
│   ├── text/
│   │   ├── real_ad_alterome_sample.tsv     # Real AD-Alterome text sample
│   │   ├── mini_ad_alterome.train.tsv      # Sampled train text
│   │   └── mini_ad_alterome.eval.tsv       # Sampled eval text
│   └── stores/
│       ├── gene_distributions.train.store/ # Cell-side distribution store (train)
│       ├── gene_distributions.eval.store/  # Cell-side distribution store (eval)
│       ├── text_train.store/               # Text evidence store (train)
│       ├── text_eval.store/                # Text evidence store (eval)
│       ├── stage1_instance_scores.train.npz # Stage1 scores (train)
│       └── stage1_instance_scores.eval.npz  # Stage1 scores (eval)
│
├── METHOD_SUMMARY.md           # Concise method overview
├── DATA_MANIFEST.md            # Required external data files
├── docs/DATA_PREPARATION.md    # Data sources and preprocessing notes
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

The bundled input stores and text previews are a small real-data subset, as recorded in `example_data/README.json` and the store metadata. They contain sampled cell distributions, text evidence, and Stage1 instance scores for checking input formats and running a small example.

See [the example data guide](example_data/README.md) for the included files and [the data preparation notes](docs/DATA_PREPARATION.md) for dataset sources and split procedures.

---

## 10. Required External Data (`DATA_MANIFEST.md`)

Large data files that **must be prepared externally** before running experiments. These are **not included** in this review package.

### 10.1 Core Required Files

| Path | Purpose | Generated by |
|---|---|---|
| `data/gene_distributions.train_donor.stage1.store` | Train cell-side gene distribution store | Cell embedding preprocessing |
| `data/gene_distributions.eval_donor.stage1.store` | Eval cell-side gene distribution store | Cell embedding preprocessing |
| `data/benchmarks/v2_textsplit_strict/text/train_text50_strict.store` | Masked strict train text evidence store | `split_text_evidence_store.py` |
| `data/benchmarks/v2_textsplit_strict/text/eval_text50_strict.store` | Masked strict eval text evidence store | `split_text_evidence_store.py` |
| `data/stage1_instance_scores.train_donor.allgenes.real.npz` | Train Stage1 instance scores | Stage1 preprocessing |
| `data/stage1_instance_scores.eval_donor.allgenes.real.npz` | Eval Stage1 instance scores | Stage1 preprocessing |
| `results/textsplit_consistency/gene_sets/strict_text_min8_genes.txt` | Gene allowlist | `make_text_evidence_gene_sets.py` |

### 10.2 Noise Experiment Files

| Path | Purpose | Generated by |
|---|---|---|
| `data/noise_text/easy_random_text_train_v2_1000.pubmedbert.npz` | Encoded easy-noise bank (train) | `encode_easy_noise_bank.py` |
| `data/noise_text/easy_random_text_eval_v2_1000.pubmedbert.npz` | Encoded easy-noise bank (eval) | `encode_easy_noise_bank.py` |

### 10.3 Gene Identifier Variant Files

| Path | Purpose | Generated by |
|---|---|---|
| `data/gene_id_maps/all_train_eval_genes_EntrezID.csv` | Gene to EntrezID mapping | External mapping |
| `data/gene_id_maps/random_gene_alias_seed42.csv` | Gene to random alias mapping | `build_gene_identifier_replacement_tsv.py` |

### 10.4 Unmasked Text Variant Stores

| Path | Purpose | Generated by |
|---|---|---|
| `data/benchmarks/v2_textsplit_strict_unmasked_exact/text/*.store` | Exact unmasked strict split stores | `build_exact_unmasked_textsplit_store.py` |
| `data/benchmarks/v2_textsplit_strict_entrezid_exact/text/*.store` | Exact EntrezID strict split stores | `build_exact_unmasked_textsplit_store.py` |
| `data/benchmarks/v2_textsplit_strict_randomgene_exact/text/*.store` | Exact random-alias strict split stores | `build_exact_unmasked_textsplit_store.py` |

### 10.5 Files NOT Included in This Package

- Model weights
- Full-size `.store` directories and `.npz` data files beyond the bundled input example
- Full `results/` directories
- Raw single-cell files
- Raw text corpora

---

## 11. Small Input Example

The input stores are already included under `example_data/stores/`. With NumPy and PyTorch installed, the following command runs one training epoch using those inputs. Run it from the repository root in Bash:

```bash
python scripts/training/run_scgates_alignment.py \
  --distribution-store example_data/stores/gene_distributions.train.store \
  --eval-distribution-store example_data/stores/gene_distributions.eval.store \
  --text-evidence-store example_data/stores/text_train.store \
  --eval-text-evidence-store example_data/stores/text_eval.store \
  --instance-score-npz example_data/stores/stage1_instance_scores.train.npz \
  --eval-instance-score-npz example_data/stores/stage1_instance_scores.eval.npz \
  --paired-eval-all-genes \
  --max-neighbors 0 \
  --epochs 1 \
  --batch-size 4 \
  --device cpu \
  --no-wandb \
  --output-dir outputs/mini_example
```

Training writes metrics and checkpoints into the directory selected by `--output-dir`. The full experiment scripts use `results/` for run artifacts, `logs/` for logs, and `tables/` for summaries. These generated directories are ignored by Git.

---

## 12. Recommended Reading Order

1. Read this README for the overall structure.
2. Review [METHOD_SUMMARY.md](METHOD_SUMMARY.md) for the method description.
3. Check [DATA_MANIFEST.md](DATA_MANIFEST.md) and [data preparation notes](docs/DATA_PREPARATION.md) for required inputs and preprocessing.
4. Inspect `stage1_neighbor_prior/` for upstream score generation.
5. Review experiment commands in `run_scripts/` and configurations in `configs/`.
6. Use the small input example above to check the training interface.

---

## 13. Stage Naming Note

The current main scGATES cell-text alignment trainer is `scripts/training/run_scgates_alignment.py`. The upstream Stage1 score generator is packaged under `stage1_neighbor_prior/`. The "Stage1" naming in the code refers to the upstream neighbor-prior module; "Stage2" and "Stage3" terminology is historical and not used in the current pipeline.
