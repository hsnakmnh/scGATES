# Geneformer Neighbor Prior

This optional module trains a contrastive model between a target gene embedding and the embeddings of its neighboring genes within a cell. It exports one normalized prior vector per gene.

## Dependencies and Inputs

Install the core dependencies from the repository root, then the module-specific packages:

```bash
python -m pip install -r requirements.txt
python -m pip install pytorch-lightning PyYAML
```

Provide externally prepared cell embedding JSONL and a neighbor map. Cell embedding extraction is upstream of this module. The JSONL record format is read by `GeneformerNeighborDataset` in [data.py](data.py).

Neighbor maps can be JSON objects such as `{"GENE1": ["A", "B"]}`, or CSV files with `query_gene`, `neighbor_gene`, and `rank` columns. Use training-donor cell embeddings when preparing a prior for donor-split evaluation.

## Train and Export

Copy [config.example.yaml](config.example.yaml) to `stage1_neighbor_prior/config.local.yaml`, replace the `<PROJECT_ROOT>` placeholders, and set the paths to your inputs. The `gene_allowlist_jsonl` input is optional and can be set to an empty string.

Run from the repository root:

```bash
python -m stage1_neighbor_prior.train --config stage1_neighbor_prior/config.local.yaml
```

The YAML values override matching command-line values. Keep `batch_size: 1`, as required by this trainer. The configured `output_prior_npz` path controls the exported vector file; checkpoints, logs, and the resolved configuration are written under `<output_dir>/<filename>/`.

To export again from a checkpoint (Bash):

```bash
python -m stage1_neighbor_prior.export_prior \
  --checkpoint outputs/geneformer-neighbor-align/final.ckpt \
  --cell-embeddings-jsonl data/cell_embeddings.train_donor.jsonl \
  --gene-neighbor-path data/gene_neighbors.json \
  --context-fields condition,context \
  --context-filter 'Dementia|Non-neuronal and Non-neural' \
  --output-npz data/geneformer_neighbor_prior.stage1.npz
```

Adjust those paths and filtering options to match the configuration used for training. The export creates an NPZ with `genes`, `vectors`, `counts`, and `metadata_json`, plus a `.npz.metadata.json` sidecar.

## Use in the Alignment Trainer

Load the exported vectors through `--gene-neighbor-prior-npz` and enable the desired prior loss or fusion option. Inspect `python scripts/training/run_scgates_alignment.py --help` for those options.

The `--instance-score-npz` and `--eval-instance-score-npz` inputs instead require per-instance scores with `keys`, `offsets`, and `scores`. See [the input contracts](../DATA_MANIFEST.md#auxiliary-input-formats) for their format. This module's vector export cannot be used as an instance-score file.
