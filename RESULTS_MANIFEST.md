# Results Manifest

All paths are project-relative. Full result directories are not included in this package.

| Result group | Relative location | Notes |
| --- | --- | --- |
| Main full scGATES | `results/paired_eval_all_genes_ablation/strict_text_min8/` | Main paired-all-genes strict text_min8 run |
| Main ablation | `results/paired_eval_all_genes_ablation/strict_text_min8/` | Full, w/o cell denoising, w/o text denoising, w/o both denoising, w/o Stage1 |
| Multi-seed full scGATES | `results/paired_eval_all_genes_multiseed/strict_text_min8/` | Seeds 42, 43, 44 |
| Mask vs unmasked | `results/mask_ablation/strict_text_min8/` | Conservative masked setting and unmasked upper-bound setting |
| Gene identifier ablation | `results/gene_identifier_ablation/strict_text_min8/` | EntrezID and random-alias variants |
| Text noise ratio sweep | `results/text_noise_target_ratio_sweep/strict_text_min8/` | Text noise from 10% to 60% |
| Cell and text noise ratio sweep | `results/cell_noise_target_ratio_sweep/strict_text_min8/` | Text noise fixed near 30%, cell noise from 10% to 60% |

## Stage Naming Note

The current main scGATES cell-text alignment trainer is `scripts/training/run_scgates_alignment.py`. The upstream Stage1 score generator is packaged under `stage1_neighbor_prior/`.

