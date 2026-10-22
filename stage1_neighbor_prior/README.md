# Geneformer Neighbor Alignment

This package is the Step-1 pretraining module for the current scGATES pipeline.
It is Geneformer-native: it reads the JSONL produced by
`scripts/export_geneformer_embeddings.py` and trains a contrastive model between:

- target gene contextual Geneformer embedding in one cell;
- the set of contextual Geneformer embeddings of that gene's neighbor genes in
  the same cell.

The reusable output for Stage1 is:

```text
data/geneformer_neighbor_prior.stage1.npz
```

The prior contains one normalized vector per gene and can later be loaded by
Stage1 as a frozen auxiliary prior.

## Expected Neighbor File

The neighbor file can be JSON:

```json
{
  "GENE1": ["A", "B", "C"],
  "GENE2": ["D", "E"]
}
```

or a Stage1 neighbor CSV with at least:

```text
query_gene,neighbor_gene,rank
```

Gene matching is case-insensitive against this file, while exported gene names
preserve the spelling from `cell_embeddings.jsonl`.

## Train And Export Prior

Recommended remote pattern:

```bash
cd <PROJECT_ROOT>
WANDB_DISABLED=true python -m stage1_neighbor_prior.train \
  --cell_embeddings_jsonl <PROJECT_ROOT>/data/cell_embeddings.train_donor.jsonl \
  --gene_neighbor_path <PROJECT_ROOT>/data/neighbors.train_donor.stage1.csv \
  --gene_allowlist_jsonl <PROJECT_ROOT>/data/ad_alterome_text_sapbert.jsonl \
  --context_fields "condition,context" \
  --context_filter "Dementia|Non-neuronal and Non-neural" \
  --output_dir <PROJECT_ROOT>/outputs \
  --filename GF_neighbor_prior_main \
  --output_prior_npz <PROJECT_ROOT>/data/geneformer_neighbor_prior.stage1.npz \
  --max_cells 3000 \
  --max_genes_per_cell 512 \
  --min_pairs_per_cell 4 \
  --min_neighbors_per_gene 2 \
  --max_neighbors_per_gene 10 \
  --max_epochs 20
```

The trainer writes:

```text
outputs/GF_neighbor_prior_main/
  resolved_config.json
  training_manifest.json
  final.ckpt
  checkpoints/
  logs/
```

## Export From A Checkpoint

If the prior was not exported during training:

```bash
python -m stage1_neighbor_prior.export_prior \
  --checkpoint <PROJECT_ROOT>/outputs/GF_neighbor_prior_main/final.ckpt \
  --cell-embeddings-jsonl <PROJECT_ROOT>/data/cell_embeddings.train_donor.jsonl \
  --gene-neighbor-path <PROJECT_ROOT>/data/neighbors.train_donor.stage1.csv \
  --gene-allowlist-jsonl <PROJECT_ROOT>/data/ad_alterome_text_sapbert.jsonl \
  --context-fields "condition,context" \
  --context-filter "Dementia|Non-neuronal and Non-neural" \
  --output-npz <PROJECT_ROOT>/data/geneformer_neighbor_prior.stage1.npz
```

The exporter also writes:

```text
data/geneformer_neighbor_prior.stage1.npz.metadata.json
```

## Notes

- Batch size is fixed to `1` because each item is one cell with a variable
  number of target gene-neighbor pairs.
- This module no longer reads `.h5ad` directly and does not use a separate
  character text encoder for neighbor names.
- Use only train-donor `cell_embeddings.jsonl` to train/export priors for
  donor-split experiments, otherwise the prior can leak eval donor information.


