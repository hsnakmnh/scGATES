#!/usr/bin/env bash
set -euo pipefail

mkdir -p tables

python scripts/summarization/summarize_run_metrics_table.py \
  results/paired_eval_all_genes_ablation/strict_text_min8/main_full_scgates_seed42 \
  results/paired_eval_all_genes_ablation/strict_text_min8/ablation_without_cell_denoising_seed42 \
  results/paired_eval_all_genes_ablation/strict_text_min8/ablation_without_text_denoising_seed42 \
  results/paired_eval_all_genes_ablation/strict_text_min8/ablation_without_cell_and_text_denoising_seed42 \
  results/paired_eval_all_genes_ablation/strict_text_min8/ablation_without_stage1_seed42 \
  --compact \
  --output-csv tables/main_ablation_metrics.csv \
  --output-md tables/main_ablation_metrics.md

python scripts/summarization/summarize_run_metrics_table.py \
  results/gene_identifier_ablation/strict_text_min8/masked_seed42 \
  results/gene_identifier_ablation/strict_text_min8/unmasked_seed42 \
  results/gene_identifier_ablation/strict_text_min8/entrezid_seed42 \
  results/gene_identifier_ablation/strict_text_min8/random_alias_seed42 \
  --compact \
  --output-csv tables/gene_identifier_ablation_metrics.csv \
  --output-md tables/gene_identifier_ablation_metrics.md


