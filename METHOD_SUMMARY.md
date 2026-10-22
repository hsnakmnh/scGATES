# Method Summary

scGATES represents each gene with two sets:

- \(D_g\): cell-side distribution set built from contextual cell or gene embeddings.
- \(P_g\): text evidence set built from AD-Alterome evidence sentences.

The model learns a shared retrieval space and evaluates whether the matched \(D_g\) and \(P_g\) rank each other highly among candidate genes.

## Components

1. Upstream Stage1 neighbor-prior estimation.
   - Packaged in `stage1_neighbor_prior/`.
   - Produces gene-neighbor or instance-score files.
   - Current main runs consume these files rather than recomputing them.
2. Set-based \(D_g\)-\(P_g\) alignment.
   - Implemented by `scripts/training/run_scgates_alignment.py`.
   - The file name is historical; it is the current main scGATES trainer.
3. Text-side confidence weighting with `text_only` confidence.
4. Prior-gated text denoising through `prior_sigmoid`.
5. Cell-side denoising through the cell set encoder.
6. Stage1 instance-score guidance as an auxiliary cell-side signal.
