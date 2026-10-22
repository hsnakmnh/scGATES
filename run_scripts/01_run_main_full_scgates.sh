#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs results/paired_eval_all_genes_ablation/strict_text_min8

WANDB_DISABLED=true python -u scripts/training/run_scgates_alignment.py \
  --paper-main \
  --distribution-store data/gene_distributions.train_donor.stage1.store \
  --eval-distribution-store data/gene_distributions.eval_donor.stage1.store \
  --text-evidence-store data/benchmarks/v2_textsplit_strict/text/train_text50_strict.store \
  --eval-text-evidence-store data/benchmarks/v2_textsplit_strict/text/eval_text50_strict.store \
  --output-dir results/paired_eval_all_genes_ablation/strict_text_min8 \
  --run-id main_full_scgates_seed42 \
  --context-filter "Dementia|Non-neuronal and Non-neural" \
  --gene-allowlist results/textsplit_consistency/gene_sets/strict_text_min8_genes.txt \
  --paired-eval-all-genes \
  --max-neighbors 0 \
  --instance-score-npz data/stage1_instance_scores.train_donor.allgenes.real.npz \
  --eval-instance-score-npz data/stage1_instance_scores.eval_donor.allgenes.real.npz \
  --instance-score-alpha 0.3 \
  --instance-score-mode signed \
  --text-conf-mode text_only \
  --text-conf-gate-mode prior_sigmoid \
  --text-conf-gate-scale 2.0 \
  --text-noise-kind easy_random_text \
  --text-noise-train-bank data/noise_text/easy_random_text_train_v2_1000.pubmedbert.npz \
  --text-noise-eval-bank data/noise_text/easy_random_text_eval_v2_1000.pubmedbert.npz \
  --text-noise-ratio 1.0 \
  --text-noise-max 64 \
  --text-noise-splits both \
  --text-noise-prior-mode uniform \
  --text-noise-seed 42 \
  --epochs 40 \
  --batch-size 32 \
  --lr 1e-4 \
  --hidden-dim 256 \
  --align-dim 256 \
  --dropout 0.1 \
  --cell-num-layers 2 \
  --cell-num-heads 4 \
  --max-distributions 0 \
  --max-instances 128 \
  --max-text-evidence 96 \
  --text-evidence-sampling prior_diverse \
  --text-diversity-weight 0.7 \
  --lambda-struct 0 \
  --lambda-denoise 0.01 \
  --lambda-cons-cell 0.05 \
  --lambda-cons-text 0.05 \
  --warmup-epochs 3 \
  --eval-train-split \
  --no-wandb \
  --eval-every-epoch \
  --eval-every-epoch-train \
  --seed 42 \
  > logs/main_full_scgates_seed42.log 2>&1


