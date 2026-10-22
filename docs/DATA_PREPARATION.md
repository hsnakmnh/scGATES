# Data Preparation Notes

## Purpose

Dataset sources and preprocessing stages for preparing full-experiment inputs.

## Data Sources

### Alzheimer Disease

- Single-cell data source: SEA-AD, https://brain-map.org/consortia/sea-ad
- Text evidence source: AD-Alterome by gene, http://lit-evi.hzau.edu.cn/ADAlterome/by-gene
- Main filtering context:
  - `Dementia|Non-neuronal and Non-neural`
  - context fields: `Cognitive Status`, `Class`
- Text gene set:
  - genes with sufficient text evidence were retained for evaluation

### Colorectal Cancer

- Single-cell data source: Progressive plasticity during colorectal cancer metastasis, CZ CELLxGENE Discover, https://cellxgene.cziscience.com/collections/1cbfb478-2c7f-4d15-b522-9f74e9fe52a8
- Cancer text evidence source: PanCancer Alterome by gene, http://lit-evi.hzau.edu.cn/PanCancer/by-gene
- Analysis subset:
  - colorectal cancer cells were selected and split by patient to reduce cell-level leakage.
  - the analysis used a cancer-only subset with held-out patients for evaluation.
- Text evidence processing:
  - disease-specific PanCancer Alterome sentences were deduplicated by gene and sentence.
  - genes with sufficient evidence were retained, and evidence per gene was capped to reduce imbalance.

### Lung Cancer

- Single-cell data source provided for this cross-disease experiment:
  - Dissecting novel myeloid-derived cell states through single-cell RNA-Seq and its impact on clinical outcome across tumor types, CZ CELLxGENE Discover, https://cellxgene.cziscience.com/collections/3f7c572c-cd73-4b51-a313-207c7f20f188
- Cancer text evidence source: PanCancer Alterome by gene, http://lit-evi.hzau.edu.cn/PanCancer/by-gene
- Analysis subset:
  - lung cancer cells were selected and split by donor to reduce cell-level leakage.
  - the analysis used a cancer-only subset with held-out donors for evaluation.
- Text evidence processing:
  - disease-specific PanCancer Alterome sentences were deduplicated by gene and sentence.
  - genes with sufficient evidence were retained, and evidence per gene was capped to reduce imbalance.

## Processing Pipeline

### Text Evidence Processing

1. Normalize raw disease-specific text evidence into a tabular format with gene and sentence fields.
2. Remove duplicated evidence by `Gene + Sentence`.
3. Select genes with sufficient text evidence.
4. Cap evidence per gene to reduce imbalance across genes.
5. Mask explicit mentions of the target gene with `[GENE]`.
6. Encode evidence sentences with the configured transformer and pooling method. The packaged variant recipe uses a local S-PubMedBert-MS-MARCO model and mean pooling; set `--model-name-or-path` to the matching model weights.
7. Export encoded evidence into text evidence stores.
8. Split text evidence into train and eval stores with a balanced split.

### Cell-Side Processing

1. Filter disease-relevant single cells.
2. Split cells by patient or donor to avoid cell-level leakage.
3. Extract Geneformer-derived contextual gene distributions.
4. Match text genes to the single-cell gene identifiers.
5. Keep gene/context distributions with sufficient cell-side support.
6. Cap stored instances per distribution to control computational cost.

## Split Protocol

For the strict Alzheimer disease setting, retain genes with at least eight evidence sentences. Split evidence into disjoint train and evaluation subsets, and mask the target gene symbol before text encoding.

Construct cell distribution stores separately for train and evaluation donors. The donor split and held-out text split serve different purposes and must both be preserved when preparing inputs.

Use [DATA_MANIFEST.md](../DATA_MANIFEST.md) for the input paths consumed by the packaged experiment scripts. The small prepared input subset is described in [example_data/README.md](../example_data/README.md).
