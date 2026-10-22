# Mini Review Example Data

This directory is intentionally small. The bundled text preview is real
AD-Alterome evidence. The runnable mini stores are generated from the real
project stores by:

```bash
bash run_scripts/99_run_mini_example.sh
```

The generated mini stores sample real cell distributions, real text evidence,
and real Stage1 instance-score files from the project data. They are only for
validating file formats and execution, not for paper claims.

Expected generated structure:

```text
example_data/
  text/
    real_ad_alterome_sample.tsv
  stores/
    gene_distributions.train.store/
    gene_distributions.eval.store/
    text_train.store/
    text_eval.store/
    stage1_instance_scores.train.npz
    stage1_instance_scores.eval.npz
```
