#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs data/gene_id_maps

python scripts/data_preprocessing/build_gene_identifier_replacement_tsv.py \
  --input-tsv data/text_data/ad-alterome-all-data.tsv \
  --entrez-map data/gene_id_maps/all_train_eval_genes_EntrezID.csv \
  --output-entrez-tsv data/text_data/ad-alterome-all-data.entrezid.tsv \
  --output-random-tsv data/text_data/ad-alterome-all-data.randomgene_seed42.tsv \
  --random-map-csv data/gene_id_maps/random_gene_alias_seed42.csv \
  --seed 42

python scripts/data_preprocessing/encode_ad_alterome_text_set.py \
  --input-tsv data/text_data/ad-alterome-all-data.entrezid.tsv \
  --output-jsonl data/ad_alterome_raw_text_set.stage1.entrezid.jsonl \
  --output-npz data/ad_alterome_raw_text_set.stage1.entrezid.pubmedbert.pg.npz \
  --evidence-output-csv data/ad_alterome_raw_text_set.stage1.entrezid.evidence.csv \
  --encoder transformer \
  --model-name-or-path model/text_encoder/S-PubMedBert-MS-MARCO \
  --batch-size 64 \
  --max-length 256 \
  --pooling mean \
  --dedup-key gene_pmid_sentence \
  --device auto

python scripts/data_preprocessing/encode_ad_alterome_text_set.py \
  --input-tsv data/text_data/ad-alterome-all-data.randomgene_seed42.tsv \
  --output-jsonl data/ad_alterome_raw_text_set.stage1.randomgene_seed42.jsonl \
  --output-npz data/ad_alterome_raw_text_set.stage1.randomgene_seed42.pubmedbert.pg.npz \
  --evidence-output-csv data/ad_alterome_raw_text_set.stage1.randomgene_seed42.evidence.csv \
  --encoder transformer \
  --model-name-or-path model/text_encoder/S-PubMedBert-MS-MARCO \
  --batch-size 64 \
  --max-length 256 \
  --pooling mean \
  --dedup-key gene_pmid_sentence \
  --device auto

python scripts/data_preprocessing/export_text_evidence_store.py \
  --input-npz data/ad_alterome_raw_text_set.stage1.entrezid.pubmedbert.pg.npz \
  --output-store data/ad_alterome_raw_text_set.stage1.entrezid.pubmedbert.pg.store \
  --overwrite

python scripts/data_preprocessing/export_text_evidence_store.py \
  --input-npz data/ad_alterome_raw_text_set.stage1.randomgene_seed42.pubmedbert.pg.npz \
  --output-store data/ad_alterome_raw_text_set.stage1.randomgene_seed42.pubmedbert.pg.store \
  --overwrite

python scripts/data_preprocessing/build_exact_unmasked_textsplit_store.py \
  --unmasked-source-store data/ad_alterome_raw_text_set.stage1.entrezid.pubmedbert.pg.store \
  --masked-train-store data/benchmarks/v2_textsplit_strict/text/train_text50_strict.store \
  --masked-eval-store data/benchmarks/v2_textsplit_strict/text/eval_text50_strict.store \
  --train-output-store data/benchmarks/v2_textsplit_strict_entrezid_exact/text/train_text50_strict_entrezid_exact.store \
  --eval-output-store data/benchmarks/v2_textsplit_strict_entrezid_exact/text/eval_text50_strict_entrezid_exact.store \
  --summary-json data/benchmarks/v2_textsplit_strict_entrezid_exact/text/build_exact_entrezid_summary.json \
  --overwrite

python scripts/data_preprocessing/build_exact_unmasked_textsplit_store.py \
  --unmasked-source-store data/ad_alterome_raw_text_set.stage1.randomgene_seed42.pubmedbert.pg.store \
  --masked-train-store data/benchmarks/v2_textsplit_strict/text/train_text50_strict.store \
  --masked-eval-store data/benchmarks/v2_textsplit_strict/text/eval_text50_strict.store \
  --train-output-store data/benchmarks/v2_textsplit_strict_randomgene_exact/text/train_text50_strict_randomgene_exact.store \
  --eval-output-store data/benchmarks/v2_textsplit_strict_randomgene_exact/text/eval_text50_strict_randomgene_exact.store \
  --summary-json data/benchmarks/v2_textsplit_strict_randomgene_exact/text/build_exact_randomgene_summary.json \
  --overwrite


