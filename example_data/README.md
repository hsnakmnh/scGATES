# Small Input Data Example

This directory contains a small real-data subset for checking scGATES input formats and the training interface. The bundled text previews contain AD-Alterome evidence; the stores contain sampled cell distributions, text embeddings, and Stage1 instance-score inputs.

The provenance and subset limits are recorded in `README.json` and each store's `metadata.json`. These records mark the subset with `synthetic: false`.

## Included Inputs

```text
example_data/
  README.json
  text/
    real_ad_alterome_sample.tsv
    mini_ad_alterome.train.tsv
    mini_ad_alterome.eval.tsv
  stores/
    gene_distributions.train.store/
    gene_distributions.eval.store/
    text_train.store/
    text_eval.store/
    stage1_instance_scores.train.npz
    stage1_instance_scores.eval.npz
```

The stores are already prepared. See [the small input command](../README.md#11-small-input-example) to pass them to the training script. Prepare full datasets using [DATA_MANIFEST.md](../DATA_MANIFEST.md) and [the data preparation notes](../docs/DATA_PREPARATION.md).
