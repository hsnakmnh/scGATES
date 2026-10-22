from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class GeneNeighborPriorStore:
    genes: list[str]
    vectors: np.ndarray
    counts: np.ndarray
    gene_to_index: dict[str, int]
    metadata: dict

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])


def load_gene_neighbor_prior_npz(path: str | Path) -> GeneNeighborPriorStore:
    data = np.load(path, allow_pickle=True)
    genes = [str(gene) for gene in data["genes"].astype(str).tolist()]
    vectors = data["vectors"].astype(np.float32)
    if vectors.ndim != 2 or len(genes) != vectors.shape[0]:
        raise ValueError(f"Bad gene neighbor prior shape: genes={len(genes)}, vectors={vectors.shape}")
    if "counts" in data:
        counts = data["counts"].astype(np.float32)
    else:
        counts = np.ones(len(genes), dtype=np.float32)
    if counts.ndim != 1 or counts.shape[0] != len(genes):
        raise ValueError(f"Bad gene neighbor prior counts shape: genes={len(genes)}, counts={counts.shape}")
    metadata: dict = {}
    if "metadata_json" in data:
        try:
            metadata = json.loads(str(data["metadata_json"].item()))
        except Exception:
            metadata = {}
    gene_to_index = {gene: idx for idx, gene in enumerate(genes)}
    return GeneNeighborPriorStore(
        genes=genes,
        vectors=vectors,
        counts=counts,
        gene_to_index=gene_to_index,
        metadata=metadata,
    )


def batch_gene_neighbor_prior(
    store: GeneNeighborPriorStore,
    genes: Sequence[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vectors = torch.zeros(len(genes), store.dim, dtype=torch.float32, device=device)
    mask = torch.zeros(len(genes), dtype=torch.bool, device=device)
    counts = torch.zeros(len(genes), dtype=torch.float32, device=device)
    for row_idx, gene in enumerate(genes):
        idx = store.gene_to_index.get(str(gene))
        if idx is None:
            continue
        vectors[row_idx] = torch.tensor(store.vectors[idx], dtype=torch.float32, device=device)
        counts[row_idx] = float(store.counts[idx])
        mask[row_idx] = True
    return vectors, mask, counts
