# scGATES

scGATES learns a shared retrieval space for cell-side gene distributions and gene-associated literature evidence. Each gene is represented by a cell embedding set $D_g$ and a text evidence set $P_g$. Set encoders learn representations and reliability weights for matching the two modalities.

## Installation

Use Python 3.11. Clone the repository and install the core dependencies:

```bash
git clone https://github.com/hsnakmnh/scGATES.git
cd scGATES
python -m pip install -r requirements.txt
```

The core example requires NumPy and PyTorch. The NumPy upper bound supports older PyTorch builds. For GPU training, install a PyTorch build suitable for your hardware using the [official installation guide](https://pytorch.org/get-started/locally/).

The CPU quick start was checked with Python 3.11, NumPy 1.26.4, and PyTorch 2.0.0.

Additional dependencies are needed only for these tasks:

| Task | Additional packages |
|---|---|
| Encode text with a pretrained transformer | `transformers` |
| Train the upstream gene-neighbor prior | `pytorch-lightning`, `PyYAML` |
| Produce application-analysis plots | `matplotlib` |

For example: `python -m pip install transformers`. Transformer model weights are supplied separately through `--model-name-or-path`.

## Quick Start

The repository includes small, prepared input stores. Run this one-epoch CPU check from the repository root; the multiline command uses Bash syntax:

```bash
python scripts/training/run_scgates_alignment.py \
  --paper-main \
  --run-id mini_example \
  --distribution-store example_data/stores/gene_distributions.train.store \
  --eval-distribution-store example_data/stores/gene_distributions.eval.store \
  --text-evidence-store example_data/stores/text_train.store \
  --eval-text-evidence-store example_data/stores/text_eval.store \
  --instance-score-npz example_data/stores/stage1_instance_scores.train.npz \
  --eval-instance-score-npz example_data/stores/stage1_instance_scores.eval.npz \
  --instance-score-alpha 0.3 \
  --paired-eval-all-genes \
  --max-neighbors 0 \
  --epochs 1 \
  --batch-size 4 \
  --device cpu \
  --output-dir outputs
```

This writes `metrics.json` and `checkpoint.pt` under `outputs/mini_example/`. The small input subset checks the training interface and file formats; it does not reproduce full-scale experiments.

`--paper-main` selects the set-alignment loss settings needed by this example. `--run-id` is required, and the trainer writes each run into `<output-dir>/<run-id>/`. See [the example data guide](example_data/README.md) for the input layout.

To inspect all training options:

```bash
python scripts/training/run_scgates_alignment.py --help
```

## Full Experiments

Prepare the external inputs listed in [DATA_MANIFEST.md](DATA_MANIFEST.md). [Data preparation notes](docs/DATA_PREPARATION.md) describe sources and split procedures. Full cell embedding extraction, model weights, and full-size instance-score inputs must be supplied separately.

The Bash recipes below pass command-line arguments directly to the Python scripts. Review their input and model paths before running them from the repository root.

| Recipe | Purpose |
|---|---|
| [00_prepare_text_stores.sh](run_scripts/00_prepare_text_stores.sh) | Build EntrezID and randomized-gene text variants from prepared base splits |
| [01_run_main_full_scgates.sh](run_scripts/01_run_main_full_scgates.sh) | Main alignment experiment |
| [02_run_main_ablation.sh](run_scripts/02_run_main_ablation.sh) | Component ablations |
| [03_run_text_noise_ratio_sweep.sh](run_scripts/03_run_text_noise_ratio_sweep.sh) | Text-noise sweep |
| [04_run_cell_text_noise_ratio_sweep.sh](run_scripts/04_run_cell_text_noise_ratio_sweep.sh) | Cell-noise sweep |
| [05_run_mask_identifier_ablation.sh](run_scripts/05_run_mask_identifier_ablation.sh) | Masked, unmasked, and alternative gene identifiers |
| [06_summarize_all_results.sh](run_scripts/06_summarize_all_results.sh) | Summarize completed runs |

For example:

```bash
bash run_scripts/01_run_main_full_scgates.sh
```

The optional [gene-neighbor prior module](stage1_neighbor_prior/README.md) exports a gene-level vector prior consumed through `--gene-neighbor-prior-npz`. Per-instance scores supplied through `--instance-score-npz` and `--eval-instance-score-npz` use a separate file format; see [the input contracts](DATA_MANIFEST.md#auxiliary-input-formats).

## Repository Layout

```text
scGATES/
  scgates/                    Model, losses, data loading, and retrieval
  scripts/
    training/                 Main alignment entry point
    data_preprocessing/       Text encoding and store preparation
    summarization/            Run and confidence summaries
    application/              Gene-evidence and confidence-gap analysis
  run_scripts/                Bash experiment recipes
  stage1_neighbor_prior/      Optional upstream prior training and export
  example_data/               Prepared inputs, provenance, and raw text preview
  docs/                      Data preparation documentation
  DATA_MANIFEST.md            Full-experiment input paths and formats
  requirements.txt           Core Python dependencies
```

The small stores and their metadata are input examples. Generated run artifacts in `outputs/`, `results/`, `logs/`, and `tables/` are ignored by Git.
