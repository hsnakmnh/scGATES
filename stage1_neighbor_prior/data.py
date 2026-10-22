from __future__ import annotations

import json
import csv
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_gene_neighbor_map(json_path: str | Path) -> Dict[str, List[str]]:
    path = Path(json_path)
    if path.suffix.lower() == ".csv":
        return load_gene_neighbor_csv(path)
    with path.open("r", encoding="utf-8-sig") as f:
        raw = json.load(f)

    mapping: dict[str, list[str]] = {}
    for gene, neighbors in raw.items():
        if isinstance(neighbors, dict):
            neighbors = neighbors.get("neighbors", [])
        if isinstance(neighbors, str):
            neighbors = [x.strip() for x in neighbors.split(",") if x.strip()]

        cleaned: list[str] = []
        seen: set[str] = set()
        for neighbor in neighbors:
            neighbor = str(neighbor).strip()
            if not neighbor:
                continue
            key = neighbor.upper()
            if key == str(gene).upper() or key in seen:
                continue
            seen.add(key)
            cleaned.append(neighbor)
        mapping[str(gene).upper()] = cleaned
    return mapping


def load_gene_neighbor_csv(path: str | Path) -> Dict[str, List[str]]:
    rows_by_gene: dict[str, list[tuple[int, str]]] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"query_gene", "neighbor_gene"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"Neighbor CSV must contain columns {sorted(required)}; got {reader.fieldnames}"
            )
        for row in reader:
            query = str(row.get("query_gene", "")).strip()
            neighbor = str(row.get("neighbor_gene", "")).strip()
            if not query or not neighbor or query.upper() == neighbor.upper():
                continue
            try:
                rank = int(row.get("rank", 10**9))
            except ValueError:
                rank = 10**9
            rows_by_gene.setdefault(query.upper(), []).append((rank, neighbor))

    mapping: dict[str, list[str]] = {}
    for query_key, items in rows_by_gene.items():
        cleaned: list[str] = []
        seen: set[str] = set()
        for _rank, neighbor in sorted(items, key=lambda item: item[0]):
            key = neighbor.upper()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(neighbor)
        mapping[query_key] = cleaned
    return mapping


def load_gene_allowlist_jsonl(path: str | Path) -> set[str]:
    genes: set[str] = set()
    for record in iter_jsonl(path):
        gene = str(record.get("gene", "")).strip()
        if gene:
            genes.add(gene)
    if not genes:
        raise ValueError(f"No gene entries found in allowlist JSONL: {path}")
    return genes


def resolve_context(record: dict[str, Any], context_fields: list[str], context_template: str = "") -> str:
    if context_template:
        values = {str(k): str(v) for k, v in record.items()}
        try:
            return context_template.format_map(values)
        except KeyError as exc:
            raise ValueError(f"Missing context template field {exc.args[0]!r}") from exc
    if context_fields:
        return "|".join(str(record.get(field, "unknown")) for field in context_fields)
    return str(record.get("context", "global"))


class GeneformerNeighborDataset(Dataset):
    """Cell-level pairs for Geneformer-native neighbor prior training.

    Each JSONL record is expected to follow scripts/export_geneformer_embeddings.py:
    {cell_id, context, condition, genes, embeddings, ranks, expr, ...}.
    For every target gene g in one cell, the dataset builds a positive neighbor
    set from Geneformer embeddings of N(g) genes that are also present in that
    cell. The model then aligns h_g with this neighbor-set representation.
    """

    def __init__(
        self,
        cell_embeddings_jsonl: str | Path,
        gene_neighbor_path: str | Path,
        context_fields: Optional[list[str]] = None,
        context_template: str = "",
        context_filter: str = "",
        gene_allowlist_jsonl: str = "",
        max_cells: Optional[int] = None,
        sample_ratio: float = 1.0,
        seed: int = 42,
        max_genes_per_cell: int = 512,
        min_pairs_per_cell: int = 4,
        min_neighbors_per_gene: int = 2,
        max_neighbors_per_gene: int = 10,
    ) -> None:
        super().__init__()
        self.cell_embeddings_jsonl = str(cell_embeddings_jsonl)
        self.gene_neighbor_path = str(gene_neighbor_path)
        self.context_fields = context_fields or ["condition", "context"]
        self.context_template = context_template
        self.context_filter = context_filter
        self.max_genes_per_cell = int(max_genes_per_cell)
        self.min_pairs_per_cell = int(min_pairs_per_cell)
        self.min_neighbors_per_gene = int(min_neighbors_per_gene)
        self.max_neighbors_per_gene = int(max_neighbors_per_gene)
        self.gene_to_neighbors = load_gene_neighbor_map(gene_neighbor_path)
        self.gene_allowlist = load_gene_allowlist_jsonl(gene_allowlist_jsonl) if gene_allowlist_jsonl else None

        rows_seen = 0
        rows_after_context = 0
        self.samples: list[dict[str, Any]] = []
        self.embedding_dim = 0
        pair_counter: Counter[str] = Counter()
        neighbor_counter: Counter[str] = Counter()
        skipped_no_pairs = 0
        rng = np.random.default_rng(seed)
        max_valid_samples = int(max_cells) if max_cells is not None and int(max_cells) > 0 else 0
        sample_ratio = float(sample_ratio if sample_ratio is not None else 1.0)
        for record in iter_jsonl(cell_embeddings_jsonl):
            rows_seen += 1
            context = resolve_context(record, self.context_fields, self.context_template)
            if context_filter and context != context_filter:
                continue
            rows_after_context += 1
            if sample_ratio < 1.0 and float(rng.random()) > sample_ratio:
                continue
            record["_resolved_context"] = context
            sample = self._build_sample(record)
            if sample is None:
                skipped_no_pairs += 1
                continue
            self.samples.append(sample)
            pair_counter.update(sample["target_genes"])
            for gene, count in zip(sample["target_genes"], sample["neighbor_counts"]):
                neighbor_counter[gene] += int(count)
            if max_valid_samples > 0 and len(self.samples) >= max_valid_samples:
                break

        if not self.samples:
            raise ValueError("No valid Geneformer neighbor samples after filtering")

        self.summary = {
            "cell_embeddings_jsonl": self.cell_embeddings_jsonl,
            "gene_neighbor_path": self.gene_neighbor_path,
            "rows_seen": int(rows_seen),
            "rows_after_context_filter": int(rows_after_context),
            "rows_selected": int(len(self.samples)),
            "valid_cell_samples": int(len(self.samples)),
            "skipped_cells_no_pairs": int(skipped_no_pairs),
            "embedding_dim": int(self.embedding_dim),
            "num_neighbor_map_genes": int(len(self.gene_to_neighbors)),
            "num_target_genes": int(len(pair_counter)),
            "num_target_pairs": int(sum(pair_counter.values())),
            "context_fields": self.context_fields,
            "context_template": self.context_template,
            "context_filter": self.context_filter,
            "gene_allowlist_size": int(len(self.gene_allowlist)) if self.gene_allowlist is not None else 0,
            "max_genes_per_cell": int(self.max_genes_per_cell),
            "min_pairs_per_cell": int(self.min_pairs_per_cell),
            "min_neighbors_per_gene": int(self.min_neighbors_per_gene),
            "max_neighbors_per_gene": int(self.max_neighbors_per_gene),
        }

    def _build_sample(self, record: dict[str, Any]) -> dict[str, Any] | None:
        genes_raw = [str(g) for g in record.get("genes", [])]
        embeddings_raw = record.get("embeddings", [])
        if not genes_raw or len(genes_raw) != len(embeddings_raw):
            return None
        limit = len(genes_raw)
        if self.max_genes_per_cell > 0:
            limit = min(limit, self.max_genes_per_cell)
        genes = genes_raw[:limit]
        embeddings = np.asarray(embeddings_raw[:limit], dtype=np.float32)
        if embeddings.ndim != 2:
            return None
        self.embedding_dim = self.embedding_dim or int(embeddings.shape[1])
        if int(embeddings.shape[1]) != self.embedding_dim:
            raise ValueError(f"Inconsistent Geneformer embedding dim: {embeddings.shape[1]} != {self.embedding_dim}")

        key_to_index = {gene.upper(): i for i, gene in enumerate(genes)}
        target_embeddings: list[np.ndarray] = []
        neighbor_embeddings: list[np.ndarray] = []
        neighbor_masks: list[np.ndarray] = []
        target_genes: list[str] = []
        neighbor_counts: list[int] = []
        for idx, gene in enumerate(genes):
            if self.gene_allowlist is not None and gene not in self.gene_allowlist:
                continue
            neighbors = self.gene_to_neighbors.get(gene.upper(), [])
            if not neighbors:
                continue
            neighbor_indices: list[int] = []
            neighbor_names: list[str] = []
            for neighbor in neighbors:
                neighbor_idx = key_to_index.get(neighbor.upper())
                if neighbor_idx is None or neighbor_idx == idx:
                    continue
                neighbor_indices.append(neighbor_idx)
                neighbor_names.append(neighbor)
                if len(neighbor_indices) >= self.max_neighbors_per_gene:
                    break
            if len(neighbor_indices) < self.min_neighbors_per_gene:
                continue
            padded = np.zeros((self.max_neighbors_per_gene, self.embedding_dim), dtype=np.float32)
            mask = np.zeros(self.max_neighbors_per_gene, dtype=bool)
            n = len(neighbor_indices)
            padded[:n] = embeddings[neighbor_indices]
            mask[:n] = True
            target_embeddings.append(embeddings[idx])
            neighbor_embeddings.append(padded)
            neighbor_masks.append(mask)
            target_genes.append(gene)
            neighbor_counts.append(n)

        if len(target_embeddings) < self.min_pairs_per_cell:
            return None
        return {
            "cell_id": str(record.get("cell_id", len(self.samples))),
            "context": str(record.get("_resolved_context", "global")),
            "target_genes": target_genes,
            "target_embeddings": torch.tensor(
                np.stack(target_embeddings, axis=0).astype(np.float32).tolist(),
                dtype=torch.float32,
            ),
            "neighbor_embeddings": torch.tensor(
                np.stack(neighbor_embeddings, axis=0).astype(np.float32).tolist(),
                dtype=torch.float32,
            ),
            "neighbor_mask": torch.tensor(np.stack(neighbor_masks, axis=0).tolist(), dtype=torch.bool),
            "neighbor_counts": neighbor_counts,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


class GeneformerNeighborCollator:
    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        if len(batch) != 1:
            raise ValueError("GeneformerNeighborCollator expects batch_size=1 because each item is one cell.")
        return batch[0]


def aggregate_gene_priors(
    rows: Iterable[tuple[str, np.ndarray]],
    min_observations: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sums: dict[str, np.ndarray] = {}
    counts: Counter[str] = Counter()
    for gene, vector in rows:
        gene = str(gene)
        vector = np.asarray(vector, dtype=np.float32)
        if gene not in sums:
            sums[gene] = np.zeros_like(vector, dtype=np.float32)
        sums[gene] += vector
        counts[gene] += 1
    genes = sorted(gene for gene, count in counts.items() if count >= min_observations)
    if not genes:
        raise ValueError("No gene priors passed min_observations")
    vectors = np.stack([sums[gene] / max(float(counts[gene]), 1.0) for gene in genes], axis=0).astype(np.float32)
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
    count_arr = np.asarray([counts[gene] for gene in genes], dtype=np.int64)
    return np.asarray(genes, dtype=object), vectors.astype(np.float32), count_arr

