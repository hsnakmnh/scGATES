# Small Input Data Example

This directory contains a prepared real-data subset for checking input formats and the scGATES training interface. Provenance, source-store paths, genes, and subset limits are recorded in `README.json` and each store's `metadata.json` (`synthetic: false`).

| Included input | Role |
|---|---|
| `stores/gene_distributions.train.store/`, `stores/gene_distributions.eval.store/` | Cell distributions and their metadata |
| `stores/text_train.store/`, `stores/text_eval.store/` | Encoded evidence, sentence metadata, and priors |
| `stores/stage1_instance_scores.train.npz`, `stores/stage1_instance_scores.eval.npz` | Per-instance scores aligned to the cell stores |
| `text/real_ad_alterome_sample.tsv` | Human-readable preview of the source evidence format |

Read sentences and text IDs from each text store's `evidence_metadata.jsonl`.

Run [the quick-start command](../README.md#quick-start) directly with the prepared stores. Full-data requirements are in [DATA_MANIFEST.md](../DATA_MANIFEST.md).
