from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scgates.io import (
    GeneDistribution,
    TextEvidenceSet,
    attach_instance_scores,
    load_distributions_npz,
    load_distributions_store,
    load_instance_score_npz,
    load_text_evidence_genes_npz,
    load_text_evidence_genes_store,
    load_text_evidence_sets_npz,
    load_text_evidence_sets_store,
    summarize_distributions,
    write_csv,
    write_json,
)
from scgates.losses import denoise_loss, multi_positive_align_loss, multi_view_consistency_loss, set_align_loss, single_align_loss, soft_positive_ce_loss
from scgates.metrics import retrieval_metrics, retrieval_metrics_by_label
from scgates.neighbor_prior import GeneNeighborPriorStore, batch_gene_neighbor_prior, load_gene_neighbor_prior_npz
from scgates.retrieval import Neighbor, load_neighbor_csv, retrieve_neighbors, write_neighbor_csv
from scgates.stage1 import Stage1AlignmentModel
from scgates.stage3_confusion import stage3_confusion_loss
from scgates.structure import structure_consistency_loss


@dataclass
class Stage1NeighborExample:
    dist: GeneDistribution
    evidence: TextEvidenceSet
    info: Neighbor


@dataclass
class Stage1Example:
    dist: GeneDistribution
    evidence: TextEvidenceSet
    neighbors: list[Stage1NeighborExample] = field(default_factory=list)
    confusions: list[Stage1NeighborExample] = field(default_factory=list)


@dataclass(frozen=True)
class EasyNoiseBank:
    noise_ids: list[str]
    categories: list[str]
    texts: list[str]
    embeddings: np.ndarray
    source_path: str


CellNoiseMetadata = dict[tuple[str, int], dict[str, str]]


TEXT_METADATA_KEYS = (
    "text",
    "sentence",
    "evidence_text",
    "content",
    "abstract",
    "title",
)


def _metadata_text(metadata: dict[str, str]) -> str:
    for key in TEXT_METADATA_KEYS:
        value = str(metadata.get(key, "")).strip()
        if value:
            return value
    return ""


def read_gene_allowlist(path: str | Path) -> set[str]:
    genes: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            gene = line.strip()
            if gene and not gene.startswith("#"):
                genes.add(gene)
    return genes


def load_easy_noise_bank(path: str | Path) -> EasyNoiseBank:
    data = np.load(path, allow_pickle=True)
    required = {"noise_ids", "categories", "texts", "embeddings"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"Easy noise bank is missing required arrays {sorted(missing)}: {path}")
    noise_ids = [str(x) for x in data["noise_ids"].astype(str)]
    categories = [str(x) for x in data["categories"].astype(str)]
    texts = [str(x) for x in data["texts"].astype(str)]
    embeddings = data["embeddings"].astype(np.float32)
    if not (len(noise_ids) == len(categories) == len(texts) == embeddings.shape[0]):
        raise ValueError(
            "Easy noise bank arrays have inconsistent lengths: "
            f"ids={len(noise_ids)} categories={len(categories)} texts={len(texts)} embeddings={embeddings.shape[0]}"
        )
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError(f"Easy noise bank embeddings must be a non-empty 2D array: {path}")
    return EasyNoiseBank(
        noise_ids=noise_ids,
        categories=categories,
        texts=texts,
        embeddings=embeddings,
        source_path=str(path),
    )


class ContrastiveMemoryBank:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.cell: torch.Tensor | None = None
        self.text: torch.Tensor | None = None
        self.labels: list[str] = []

    def __len__(self) -> int:
        return len(self.labels)

    def enqueue(self, cell_z: torch.Tensor, text_z: torch.Tensor, labels: list[str]) -> None:
        if self.capacity <= 0 or not labels:
            return
        cell_z = F.normalize(cell_z.detach(), dim=-1)
        text_z = F.normalize(text_z.detach(), dim=-1)
        if self.cell is None or self.text is None:
            self.cell = cell_z
            self.text = text_z
            self.labels = list(labels)
        else:
            self.cell = torch.cat([self.cell, cell_z], dim=0)
            self.text = torch.cat([self.text, text_z], dim=0)
            self.labels.extend(labels)
        if len(self.labels) > self.capacity:
            keep = len(self.labels) - self.capacity
            self.cell = self.cell[keep:]
            self.text = self.text[keep:]
            self.labels = self.labels[keep:]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cli_flag_was_passed(flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv[1:])


def split_examples(examples: list[Stage1Example], seed: int) -> tuple[list[Stage1Example], list[Stage1Example]]:
    if len(examples) < 4:
        return examples, examples
    genes = sorted({e.dist.gene for e in examples})
    rng = random.Random(seed)
    rng.shuffle(genes)
    train_genes = set(genes[: max(2, int(0.7 * len(genes)))])
    train = [e for e in examples if e.dist.gene in train_genes]
    eval_items = [e for e in examples if e.dist.gene not in train_genes]
    return train, eval_items or train


def limit_distributions(
    distributions: list[GeneDistribution],
    max_distributions: int,
    sample: bool,
    seed: int,
) -> list[GeneDistribution]:
    if max_distributions <= 0 or len(distributions) <= max_distributions:
        return distributions
    if not sample:
        return distributions[:max_distributions]
    rng = random.Random(seed)
    selected = sorted(rng.sample(range(len(distributions)), max_distributions))
    return [distributions[i] for i in selected]


def build_examples(
    distributions: list[GeneDistribution],
    evidence_sets: list[TextEvidenceSet],
    neighbor_map: dict[str, list[Neighbor]] | None = None,
    confusion_map: dict[str, list[Neighbor]] | None = None,
) -> list[Stage1Example]:
    evidence_by_gene = {e.gene: e for e in evidence_sets}
    dist_by_key = {d.key: d for d in distributions}
    examples: list[Stage1Example] = []
    for dist in distributions:
        if dist.gene not in evidence_by_gene:
            continue
        neighbors: list[Stage1NeighborExample] = []
        if neighbor_map is not None:
            for info in neighbor_map.get(dist.key, []):
                neighbor_dist = dist_by_key.get(info.neighbor_key)
                neighbor_evidence = evidence_by_gene.get(info.neighbor_gene)
                if neighbor_dist is None or neighbor_evidence is None:
                    continue
                neighbors.append(Stage1NeighborExample(dist=neighbor_dist, evidence=neighbor_evidence, info=info))
        confusions: list[Stage1NeighborExample] = []
        if confusion_map is not None:
            for info in confusion_map.get(dist.key, []):
                confusion_dist = dist_by_key.get(info.neighbor_key)
                confusion_evidence = evidence_by_gene.get(info.neighbor_gene)
                if confusion_dist is None or confusion_evidence is None:
                    continue
                confusions.append(Stage1NeighborExample(dist=confusion_dist, evidence=confusion_evidence, info=info))
        examples.append(Stage1Example(dist=dist, evidence=evidence_by_gene[dist.gene], neighbors=neighbors, confusions=confusions))
    if not examples:
        raise ValueError("No matched gene distributions and text evidence sets")
    return examples


def inject_text_evidence_noise(
    evidence_sets: list[TextEvidenceSet],
    ratio: float,
    max_noise: int,
    seed: int,
    prior_mode: str = "uniform",
) -> tuple[list[TextEvidenceSet], dict[str, object]]:
    """Append marked cross-gene text evidence as synthetic noise.

    Noise labels are stored only in metadata for auditing; they are not exposed
    as model input features. ``prior_mode=uniform`` avoids giving the model an
    easy prior-based noise label.
    """
    clean_total = sum(len(e.text_ids) for e in evidence_sets)
    if ratio <= 0 or len(evidence_sets) < 2:
        return evidence_sets, {
            "enabled": False,
            "sets": len(evidence_sets),
            "clean_evidence": clean_total,
            "noise_evidence": 0,
            "noise_fraction": 0.0,
        }

    rng = np.random.RandomState(seed)
    candidates: list[tuple[int, int]] = [
        (set_idx, local_idx)
        for set_idx, evidence in enumerate(evidence_sets)
        for local_idx in range(len(evidence.text_ids))
    ]
    if not candidates:
        return evidence_sets, {
            "enabled": False,
            "sets": len(evidence_sets),
            "clean_evidence": 0,
            "noise_evidence": 0,
            "noise_fraction": 0.0,
        }

    out: list[TextEvidenceSet] = []
    total_noise = 0
    for target_idx, evidence in enumerate(evidence_sets):
        clean_n = len(evidence.text_ids)
        noise_n = int(np.ceil(clean_n * float(ratio)))
        if max_noise > 0:
            noise_n = min(noise_n, int(max_noise))
        if noise_n <= 0:
            out.append(evidence)
            continue

        chosen: list[tuple[int, int]] = []
        used: set[tuple[int, int]] = set()
        attempts = 0
        max_attempts = max(100, noise_n * 100)
        while len(chosen) < noise_n and attempts < max_attempts:
            attempts += 1
            src_set_idx, src_local_idx = candidates[int(rng.randint(0, len(candidates)))]
            source = evidence_sets[src_set_idx]
            key = (src_set_idx, src_local_idx)
            if src_set_idx == target_idx or source.gene == evidence.gene or key in used:
                continue
            chosen.append(key)
            used.add(key)

        if not chosen:
            out.append(evidence)
            continue

        noise_embeddings: list[np.ndarray] = []
        noise_text_ids: list[str] = []
        noise_prior: list[float] = []
        noise_metadata: list[dict[str, str]] = []
        noise_metadata_features: list[np.ndarray] = []
        target_metadata_dim = int(evidence.metadata_features.shape[1]) if evidence.metadata_features is not None else 0

        for src_set_idx, src_local_idx in chosen:
            source = evidence_sets[src_set_idx]
            source_text_id = str(source.text_ids[src_local_idx])
            noise_embeddings.append(np.asarray(source.embeddings[src_local_idx], dtype=np.float32))
            noise_text_id = f"noise::{source.gene}::{source_text_id}"
            noise_text_ids.append(noise_text_id)
            noise_prior.append(float(source.prior[src_local_idx]) if source.prior.size else 1.0)
            metadata = dict(source.metadata[src_local_idx]) if src_local_idx < len(source.metadata) else {}
            metadata.update(
                {
                    "text_id": noise_text_id,
                    "is_noise": "1",
                    "noise_kind": "cross_gene_text",
                    "source_gene": str(source.gene),
                    "source_context": str(source.context),
                    "source_text_id": source_text_id,
                    "target_gene": str(evidence.gene),
                    "target_context": str(evidence.context),
                }
            )
            noise_metadata.append(metadata)
            if target_metadata_dim:
                if source.metadata_features is not None and source.metadata_features.shape[1] == target_metadata_dim:
                    noise_metadata_features.append(np.asarray(source.metadata_features[src_local_idx], dtype=np.float32))
                else:
                    noise_metadata_features.append(np.zeros(target_metadata_dim, dtype=np.float32))

        clean_metadata: list[dict[str, str]] = []
        for text_id, metadata in zip(evidence.text_ids, evidence.metadata):
            item = dict(metadata)
            item.setdefault("text_id", str(text_id))
            item.update(
                {
                    "is_noise": "0",
                    "noise_kind": "clean",
                    "source_gene": str(evidence.gene),
                    "source_context": str(evidence.context),
                    "source_text_id": str(text_id),
                    "target_gene": str(evidence.gene),
                    "target_context": str(evidence.context),
                }
            )
            clean_metadata.append(item)

        embeddings = np.concatenate(
            [np.asarray(evidence.embeddings, dtype=np.float32), np.stack(noise_embeddings).astype(np.float32)],
            axis=0,
        )
        text_ids = [*evidence.text_ids, *noise_text_ids]
        if prior_mode == "uniform":
            prior = np.ones(len(text_ids), dtype=np.float32)
        elif prior_mode == "source":
            prior = np.concatenate(
                [np.asarray(evidence.prior, dtype=np.float32), np.asarray(noise_prior, dtype=np.float32)],
                axis=0,
            )
        else:
            raise ValueError(f"Unsupported text noise prior mode: {prior_mode}")
        prior = prior / max(float(prior.sum()), 1e-8)

        metadata_features = None
        if target_metadata_dim:
            metadata_features = np.concatenate(
                [
                    np.asarray(evidence.metadata_features, dtype=np.float32),
                    np.stack(noise_metadata_features).astype(np.float32),
                ],
                axis=0,
            )

        total_noise += len(noise_text_ids)
        out.append(
            TextEvidenceSet(
                gene=evidence.gene,
                context=evidence.context,
                text_ids=text_ids,
                embeddings=embeddings,
                prior=prior.astype(np.float32),
                metadata=[*clean_metadata, *noise_metadata],
                metadata_features=metadata_features,
            )
        )

    denom = clean_total + total_noise
    return out, {
        "enabled": True,
        "sets": len(out),
        "clean_evidence": clean_total,
        "noise_evidence": total_noise,
        "noise_fraction": float(total_noise / denom) if denom else 0.0,
        "ratio": float(ratio),
        "max_noise": int(max_noise),
        "prior_mode": prior_mode,
        "seed": int(seed),
    }


def inject_easy_text_evidence_noise(
    evidence_sets: list[TextEvidenceSet],
    bank: EasyNoiseBank,
    ratio: float,
    max_noise: int,
    seed: int,
    prior_mode: str = "uniform",
) -> tuple[list[TextEvidenceSet], dict[str, object]]:
    """Append marked general-domain easy-noise evidence from a pre-encoded bank."""
    clean_total = sum(len(e.text_ids) for e in evidence_sets)
    if ratio <= 0:
        return evidence_sets, {
            "enabled": False,
            "kind": "easy_random_text",
            "sets": len(evidence_sets),
            "clean_evidence": clean_total,
            "noise_evidence": 0,
            "noise_fraction": 0.0,
            "replacement": False,
            "bank_path": bank.source_path,
            "bank_size": len(bank.noise_ids),
        }
    if not evidence_sets:
        return evidence_sets, {
            "enabled": False,
            "kind": "easy_random_text",
            "sets": 0,
            "clean_evidence": 0,
            "noise_evidence": 0,
            "noise_fraction": 0.0,
            "replacement": False,
            "bank_path": bank.source_path,
            "bank_size": len(bank.noise_ids),
        }
    if bank.embeddings.shape[1] != evidence_sets[0].embeddings.shape[1]:
        raise ValueError(
            "Easy noise embedding dimension does not match text evidence embeddings: "
            f"bank={bank.embeddings.shape[1]} evidence={evidence_sets[0].embeddings.shape[1]}"
        )

    rng = np.random.RandomState(seed)
    out: list[TextEvidenceSet] = []
    total_noise = 0
    used_replacement = False
    bank_size = len(bank.noise_ids)

    for evidence in evidence_sets:
        clean_n = len(evidence.text_ids)
        noise_n = int(np.ceil(clean_n * float(ratio)))
        if max_noise > 0:
            noise_n = min(noise_n, int(max_noise))
        if noise_n <= 0:
            out.append(evidence)
            continue

        replace = noise_n > bank_size
        used_replacement = used_replacement or replace
        chosen = rng.choice(bank_size, size=noise_n, replace=replace)
        noise_embeddings: list[np.ndarray] = []
        noise_text_ids: list[str] = []
        noise_prior: list[float] = []
        noise_metadata: list[dict[str, str]] = []
        noise_metadata_features: list[np.ndarray] = []
        target_metadata_dim = int(evidence.metadata_features.shape[1]) if evidence.metadata_features is not None else 0

        for bank_idx in chosen:
            idx = int(bank_idx)
            noise_id = str(bank.noise_ids[idx])
            noise_text_id = f"noise::EASY_RANDOM::{noise_id}"
            noise_embeddings.append(np.asarray(bank.embeddings[idx], dtype=np.float32))
            noise_text_ids.append(noise_text_id)
            noise_prior.append(1.0)
            noise_metadata.append(
                {
                    "text_id": noise_text_id,
                    "text": str(bank.texts[idx]),
                    "is_noise": "1",
                    "noise_kind": "easy_random_text",
                    "source_gene": "EASY_RANDOM",
                    "source_context": "external",
                    "source_text_id": noise_id,
                    "target_gene": str(evidence.gene),
                    "target_context": str(evidence.context),
                    "category": str(bank.categories[idx]),
                }
            )
            if target_metadata_dim:
                noise_metadata_features.append(np.zeros(target_metadata_dim, dtype=np.float32))

        clean_metadata: list[dict[str, str]] = []
        for text_id, metadata in zip(evidence.text_ids, evidence.metadata):
            item = dict(metadata)
            item.setdefault("text_id", str(text_id))
            item.update(
                {
                    "is_noise": "0",
                    "noise_kind": "clean",
                    "source_gene": str(evidence.gene),
                    "source_context": str(evidence.context),
                    "source_text_id": str(text_id),
                    "target_gene": str(evidence.gene),
                    "target_context": str(evidence.context),
                }
            )
            clean_metadata.append(item)

        embeddings = np.concatenate(
            [np.asarray(evidence.embeddings, dtype=np.float32), np.stack(noise_embeddings).astype(np.float32)],
            axis=0,
        )
        text_ids = [*evidence.text_ids, *noise_text_ids]
        if prior_mode == "uniform":
            prior = np.ones(len(text_ids), dtype=np.float32)
        elif prior_mode == "source":
            prior = np.concatenate(
                [np.asarray(evidence.prior, dtype=np.float32), np.asarray(noise_prior, dtype=np.float32)],
                axis=0,
            )
        else:
            raise ValueError(f"Unsupported text noise prior mode: {prior_mode}")
        prior = prior / max(float(prior.sum()), 1e-8)

        metadata_features = None
        if target_metadata_dim:
            metadata_features = np.concatenate(
                [
                    np.asarray(evidence.metadata_features, dtype=np.float32),
                    np.stack(noise_metadata_features).astype(np.float32),
                ],
                axis=0,
            )

        total_noise += len(noise_text_ids)
        out.append(
            TextEvidenceSet(
                gene=evidence.gene,
                context=evidence.context,
                text_ids=text_ids,
                embeddings=embeddings,
                prior=prior.astype(np.float32),
                metadata=[*clean_metadata, *noise_metadata],
                metadata_features=metadata_features,
            )
        )

    denom = clean_total + total_noise
    return out, {
        "enabled": True,
        "kind": "easy_random_text",
        "sets": len(out),
        "clean_evidence": clean_total,
        "noise_evidence": total_noise,
        "noise_fraction": float(total_noise / denom) if denom else 0.0,
        "ratio": float(ratio),
        "max_noise": int(max_noise),
        "prior_mode": prior_mode,
        "seed": int(seed),
        "replacement": bool(used_replacement),
        "bank_path": bank.source_path,
        "bank_size": int(bank_size),
    }


def inject_cell_distribution_noise(
    examples: list[Stage1Example],
    ratio: float,
    max_noise: int,
    seed: int,
    source_mode: str = "same_context",
    instance_score_mode: str = "zero",
) -> tuple[list[Stage1Example], dict[str, object], CellNoiseMetadata]:
    """Append marked cross-gene cell embeddings as synthetic cell-side noise."""
    clean_total = sum(int(example.dist.embeddings.shape[0]) for example in examples)
    if ratio <= 0 or len(examples) < 2:
        return examples, {
            "enabled": False,
            "sets": len(examples),
            "clean_instances": clean_total,
            "noise_instances": 0,
            "noise_fraction": 0.0,
            "ratio": float(ratio),
            "max_noise": int(max_noise),
            "source": source_mode,
            "instance_score_mode": instance_score_mode,
            "seed": int(seed),
        }, {}

    if source_mode not in {"same_context", "global"}:
        raise ValueError(f"Unsupported cell noise source mode: {source_mode}")
    if instance_score_mode not in {"zero", "source", "uniform"}:
        raise ValueError(f"Unsupported cell noise instance score mode: {instance_score_mode}")

    rng = np.random.RandomState(seed)
    candidates_by_context: dict[str, list[tuple[int, int]]] = {}
    global_candidates: list[tuple[int, int]] = []
    for example_idx, example in enumerate(examples):
        n = int(example.dist.embeddings.shape[0])
        local_candidates = [(example_idx, local_idx) for local_idx in range(n)]
        candidates_by_context.setdefault(example.dist.context, []).extend(local_candidates)
        global_candidates.extend(local_candidates)

    out: list[Stage1Example] = []
    metadata: CellNoiseMetadata = {}
    total_noise = 0
    groups_with_noise = 0

    for target_idx, example in enumerate(examples):
        dist = example.dist
        clean_n = int(dist.embeddings.shape[0])
        noise_n = int(np.ceil(clean_n * float(ratio)))
        if max_noise > 0:
            noise_n = min(noise_n, int(max_noise))
        if noise_n <= 0:
            out.append(example)
            for local_idx in range(clean_n):
                metadata[(dist.key, local_idx)] = {
                    "is_cell_noise": "0",
                    "cell_noise_kind": "clean",
                    "source_gene": dist.gene,
                    "source_context": dist.context,
                    "target_gene": dist.gene,
                    "target_context": dist.context,
                }
            continue

        pool = candidates_by_context.get(dist.context, []) if source_mode == "same_context" else global_candidates
        if not pool:
            out.append(example)
            continue

        chosen: list[tuple[int, int]] = []
        used: set[tuple[int, int]] = set()
        attempts = 0
        max_attempts = max(100, noise_n * 100)
        while len(chosen) < noise_n and attempts < max_attempts:
            attempts += 1
            source_example_idx, source_local_idx = pool[int(rng.randint(0, len(pool)))]
            source_example = examples[source_example_idx]
            key = (source_example_idx, source_local_idx)
            if source_example_idx == target_idx or source_example.dist.gene == dist.gene or key in used:
                continue
            chosen.append(key)
            used.add(key)

        if not chosen:
            out.append(example)
            for local_idx in range(clean_n):
                metadata[(dist.key, local_idx)] = {
                    "is_cell_noise": "0",
                    "cell_noise_kind": "clean",
                    "source_gene": dist.gene,
                    "source_context": dist.context,
                    "target_gene": dist.gene,
                    "target_context": dist.context,
                }
            continue

        noise_embeddings: list[np.ndarray] = []
        noise_ranks: list[float] = []
        noise_expr: list[float] = []
        noise_scores: list[float] = []
        for source_example_idx, source_local_idx in chosen:
            source_dist = examples[source_example_idx].dist
            noise_embeddings.append(np.asarray(source_dist.embeddings[source_local_idx], dtype=np.float32))
            noise_ranks.append(float(source_dist.ranks[source_local_idx]))
            noise_expr.append(float(source_dist.expr[source_local_idx]))
            if instance_score_mode == "source" and source_dist.instance_scores is not None:
                noise_scores.append(float(source_dist.instance_scores[source_local_idx]))
            elif instance_score_mode == "uniform":
                noise_scores.append(1.0)
            else:
                noise_scores.append(0.0)

        clean_scores = (
            np.asarray(dist.instance_scores, dtype=np.float32)
            if dist.instance_scores is not None
            else np.zeros(clean_n, dtype=np.float32)
        )
        new_dist = GeneDistribution(
            key=dist.key,
            gene=dist.gene,
            context=dist.context,
            embeddings=np.concatenate(
                [np.asarray(dist.embeddings, dtype=np.float32), np.stack(noise_embeddings).astype(np.float32)],
                axis=0,
            ),
            ranks=np.concatenate([np.asarray(dist.ranks, dtype=np.float32), np.asarray(noise_ranks, dtype=np.float32)]),
            expr=np.concatenate([np.asarray(dist.expr, dtype=np.float32), np.asarray(noise_expr, dtype=np.float32)]),
            occurrence_rate=dist.occurrence_rate,
            instance_scores=np.concatenate([clean_scores, np.asarray(noise_scores, dtype=np.float32)]),
        )
        for local_idx in range(clean_n):
            metadata[(new_dist.key, local_idx)] = {
                "is_cell_noise": "0",
                "cell_noise_kind": "clean",
                "source_gene": dist.gene,
                "source_context": dist.context,
                "target_gene": dist.gene,
                "target_context": dist.context,
            }
        for offset, (source_example_idx, source_local_idx) in enumerate(chosen):
            source_dist = examples[source_example_idx].dist
            noisy_idx = clean_n + offset
            metadata[(new_dist.key, noisy_idx)] = {
                "is_cell_noise": "1",
                "cell_noise_kind": "cross_gene_cell",
                "source_gene": source_dist.gene,
                "source_context": source_dist.context,
                "source_instance_index": str(source_local_idx),
                "target_gene": dist.gene,
                "target_context": dist.context,
            }
        total_noise += len(chosen)
        groups_with_noise += 1
        out.append(
            Stage1Example(
                dist=new_dist,
                evidence=example.evidence,
                neighbors=example.neighbors,
                confusions=example.confusions,
            )
        )

    denom = clean_total + total_noise
    return out, {
        "enabled": True,
        "sets": len(out),
        "sets_with_noise": int(groups_with_noise),
        "clean_instances": int(clean_total),
        "noise_instances": int(total_noise),
        "noise_fraction": float(total_noise / denom) if denom else 0.0,
        "ratio": float(ratio),
        "max_noise": int(max_noise),
        "source": source_mode,
        "instance_score_mode": instance_score_mode,
        "seed": int(seed),
    }, metadata


def log_step(message: str, start_time: float | None = None) -> float:
    now = time.time()
    if start_time is None:
        print(f"[stage1] {message}", flush=True)
    else:
        print(f"[stage1] {message} ({now - start_time:.2f}s)", flush=True)
    return now


def embedding_diagnostics(prefix: str, arr: np.ndarray) -> dict[str, float]:
    if arr.size == 0:
        return {
            f"{prefix}_dim_std_mean": 0.0,
            f"{prefix}_effective_rank": 0.0,
            f"{prefix}_mean_pairwise_cos": 0.0,
            f"{prefix}_top1_concentration": 0.0,
        }
    arr = arr.astype(np.float32)
    normalized = arr / np.maximum(np.linalg.norm(arr, axis=-1, keepdims=True), 1e-8)
    centered = normalized - normalized.mean(axis=0, keepdims=True)
    dim_std_mean = float(centered.std(axis=0).mean())
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    energy = singular_values**2
    if float(energy.sum()) > 0:
        prob = energy / float(energy.sum())
        effective_rank = float(np.exp(-np.sum(prob * np.log(prob + 1e-12))))
    else:
        effective_rank = 0.0
    sim = normalized @ normalized.T
    if sim.shape[0] > 1:
        off_diag = sim[~np.eye(sim.shape[0], dtype=bool)]
        mean_pairwise_cos = float(off_diag.mean())
        nn = np.argmax(sim - np.eye(sim.shape[0], dtype=np.float32) * 2.0, axis=1)
        top1_concentration = float(np.max(np.bincount(nn, minlength=sim.shape[0])) / max(sim.shape[0], 1))
    else:
        mean_pairwise_cos = 1.0
        top1_concentration = 1.0
    return {
        f"{prefix}_dim_std_mean": dim_std_mean,
        f"{prefix}_effective_rank": effective_rank,
        f"{prefix}_mean_pairwise_cos": mean_pairwise_cos,
        f"{prefix}_top1_concentration": top1_concentration,
    }


def batch_forward(
    model: Stage1AlignmentModel,
    batch: list[Stage1Example],
    max_instances: int,
    max_text_evidence: int,
    device: torch.device,
    text_evidence_sampling: str,
    max_neighbors: int,
    max_confusions: int,
    text_prior_temperature: float,
    text_diversity_weight: float,
    set_fusion: str,
    residual_scale: float,
    neighbor_agg_alpha: float,
    instance_score_alpha: float,
    instance_score_mode: str,
    sampling_seed: int | None = None,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    neighbor_distributions = None
    neighbor_infos = None
    neighbor_evidence_sets = None
    if max_neighbors > 0:
        neighbor_distributions = [[n.dist for n in e.neighbors[:max_neighbors]] for e in batch]
        neighbor_evidence_sets = [[n.evidence for n in e.neighbors[:max_neighbors]] for e in batch]
        neighbor_infos = [[n.info for n in e.neighbors[:max_neighbors]] for e in batch]
    confusion_evidence_sets = None
    if max_confusions > 0:
        confusion_evidence_sets = [[n.evidence for n in e.confusions[:max_confusions]] for e in batch]
    return model(
        [e.dist for e in batch],
        [e.evidence for e in batch],
        max_instances=max_instances,
        max_text_evidence=max_text_evidence,
        device=device,
        text_evidence_sampling=text_evidence_sampling,
        text_prior_temperature=text_prior_temperature,
        text_diversity_weight=text_diversity_weight,
        neighbor_distributions=neighbor_distributions,
        neighbor_evidence_sets=neighbor_evidence_sets,
        neighbor_infos=neighbor_infos,
        max_neighbors=max_neighbors,
        neighbor_agg_alpha=neighbor_agg_alpha,
        instance_score_alpha=instance_score_alpha,
        instance_score_mode=instance_score_mode,
        confusion_evidence_sets=confusion_evidence_sets,
        max_confusions=max_confusions,
        set_fusion=set_fusion,
        residual_scale=residual_scale,
        sampling_seed=sampling_seed,
    )


def memory_bank_align_loss(
    cell_z: torch.Tensor,
    text_z: torch.Tensor,
    labels: list[str],
    bank: ContrastiveMemoryBank | None,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = cell_z.new_tensor(0.0)
    if bank is None or bank.cell is None or bank.text is None or len(bank) == 0:
        return zero, {
            "loss_memory": zero.detach(),
            "memory_bank_size": zero.detach(),
            "memory_effective_negatives": zero.detach(),
            "memory_cell_acc": zero.detach(),
            "memory_text_acc": zero.detach(),
        }

    labels_arr = np.asarray(labels, dtype=object)
    bank_labels = np.asarray(bank.labels, dtype=object)
    valid_neg = torch.tensor(labels_arr[:, None] != bank_labels[None, :], dtype=torch.bool, device=cell_z.device)

    cell_z = F.normalize(cell_z, dim=-1)
    text_z = F.normalize(text_z, dim=-1)
    bank_cell = bank.cell.to(device=cell_z.device, dtype=cell_z.dtype)
    bank_text = bank.text.to(device=text_z.device, dtype=text_z.dtype)
    tau = max(float(temperature), 1e-6)

    pos_cell = torch.sum(cell_z * text_z, dim=-1, keepdim=True) / tau
    neg_cell = (cell_z @ bank_text.T) / tau
    neg_cell = neg_cell.masked_fill(~valid_neg, -1e9)
    logits_cell = torch.cat([pos_cell, neg_cell], dim=1)

    pos_text = torch.sum(text_z * cell_z, dim=-1, keepdim=True) / tau
    neg_text = (text_z @ bank_cell.T) / tau
    neg_text = neg_text.masked_fill(~valid_neg, -1e9)
    logits_text = torch.cat([pos_text, neg_text], dim=1)

    targets = torch.zeros(cell_z.shape[0], dtype=torch.long, device=cell_z.device)
    loss = 0.5 * (F.cross_entropy(logits_cell, targets) + F.cross_entropy(logits_text, targets))
    cell_acc = (logits_cell.argmax(dim=1) == 0).to(cell_z.dtype).mean()
    text_acc = (logits_text.argmax(dim=1) == 0).to(cell_z.dtype).mean()
    return loss, {
        "loss_memory": loss.detach(),
        "memory_bank_size": cell_z.new_tensor(float(len(bank))).detach(),
        "memory_effective_negatives": valid_neg.sum(dim=1).to(cell_z.dtype).mean().detach(),
        "memory_cell_acc": cell_acc.detach(),
        "memory_text_acc": text_acc.detach(),
    }


def _neighbor_soft_loss(
    batch: list[Stage1Example],
    out: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Build soft-target CE loss with neighbours as partial positives."""
    device = out["cell_set"].device  # type: ignore[union-attr]
    zero = torch.tensor(0.0, device=device)
    if args.lambda_neighbor_soft <= 0 or args.max_neighbors <= 0:
        return zero, {"loss_neighbor_soft": zero, "neighbor_soft_coverage": zero}

    # Map genes to batch indices.
    gene_to_idx: dict[str, list[int]] = {}
    for i, ex in enumerate(batch):
        gene_to_idx.setdefault(ex.dist.gene, []).append(i)
    B = len(batch)

    # Build soft-target matrix: self=1.0, neighbours=cosine_score * soft_weight
    raw = torch.zeros(B, B, device=device)
    for i, ex in enumerate(batch):
        raw[i, i] = 1.0
        for n in ex.neighbors[: args.max_neighbors]:
            n_gene = n.info.neighbor_gene
            if n_gene in gene_to_idx:
                w = max(0.0, float(n.info.cosine_score)) * args.neighbor_soft_weight
                for j in gene_to_idx[n_gene]:
                    raw[i, j] += w

    # Row-normalise to sum 1
    row_sum = raw.sum(dim=1, keepdim=True).clamp_min(1e-8)
    soft_targets = raw / row_sum

    # Validate: self weight must be > 0
    diag_mean = float(soft_targets.diag().mean())
    if diag_mean < 0.1:
        return zero, {"loss_neighbor_soft": zero, "neighbor_soft_coverage": zero}

    loss_set = soft_positive_ce_loss(
        out["cell_set"],  # type: ignore[arg-type]
        out["text_set"],  # type: ignore[arg-type]
        soft_targets,
        temperature=args.tau_ce,
    )
    loss_single = soft_positive_ce_loss(
        out["cell_single"],  # type: ignore[arg-type]
        out["text_single"],  # type: ignore[arg-type]
        soft_targets,
        temperature=args.tau_ce,
    )
    coverage = float((raw.sum(dim=1) > 1.01).float().mean())
    return 0.5 * (loss_single + loss_set), {
        "loss_neighbor_soft": (loss_single + loss_set).detach() * 0.5,
        "neighbor_soft_coverage": torch.tensor(coverage, device=device),
    }


def _expr_stratified_loss(
    model: Stage1AlignmentModel,
    out: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """T3: expression-stratified self-supervision.

    Splits each gene's cell instances into high-expr and low-expr halves,
    encodes each subset through the cell_set encoder, and penalises MSE
    between the two representations. Forces the encoder to produce stable
    gene-level signals regardless of which expression stratum is sampled.
    Purely cell-side; no cross-modal signal required.
    """
    device = out["cell_set"].device  # type: ignore[union-attr]
    zero = torch.tensor(0.0, device=device)
    if args.lambda_expr_strat <= 0:
        return zero, {"loss_expr_strat": zero}

    cell_emb = out["cell_embeddings"]  # type: ignore[union-attr]  # (B, M, D)
    cell_expr = out["cell_expr"]        # type: ignore[union-attr]  # (B, M)
    cell_ranks = out["cell_ranks"]      # type: ignore[union-attr]  # (B, M)
    cell_mask = out["cell_mask"]        # type: ignore[union-attr]  # (B, M)
    occurrence = out["cell_occurrence"] # type: ignore[union-attr]  # (B,)

    B, M, D = cell_emb.shape

    # Build high/low masks per gene
    high_mask = torch.zeros(B, M, dtype=torch.bool, device=device)
    low_mask = torch.zeros(B, M, dtype=torch.bool, device=device)
    for i in range(B):
        valid = cell_mask[i]
        if valid.sum() < 4:
            continue
        expr_valid = cell_expr[i, valid]
        median_val = expr_valid.median()
        valid_idx = torch.where(valid)[0]
        high_mask[i, valid_idx[expr_valid >= median_val]] = True
        low_mask[i, valid_idx[expr_valid < median_val]] = True

    # Encode high and low subsets
    with torch.no_grad() if args.expr_strat_detach else torch.enable_grad():
        aux_high = model.cell_set(cell_emb, cell_ranks, cell_expr, high_mask, occurrence)
        aux_low = model.cell_set(cell_emb, cell_ranks, cell_expr, low_mask, occurrence)

    loss = F.mse_loss(aux_high["z"], aux_low["z"])
    parts = {
        "loss_expr_strat": loss.detach(),
    }
    return loss, parts


def gene_neighbor_prior_loss(
    batch: list[Stage1Example],
    out: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    prior_store: GeneNeighborPriorStore | None,
    prior_proj: nn.Module | None,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = out["cell_set"].device  # type: ignore[union-attr]
    zero = torch.tensor(0.0, device=device)
    if (
        prior_store is None
        or prior_proj is None
        or (args.lambda_neighbor_prior_cell <= 0 and args.lambda_neighbor_prior_text <= 0)
    ):
        return zero, {
            "loss_neighbor_prior": zero,
            "loss_neighbor_prior_cell": zero,
            "loss_neighbor_prior_text": zero,
            "neighbor_prior_coverage": zero,
        }
    genes = [example.dist.gene for example in batch]
    prior_vec, prior_mask, prior_counts = batch_gene_neighbor_prior(prior_store, genes, device)
    if args.gene_neighbor_prior_min_count > 1:
        prior_mask = prior_mask & (prior_counts >= float(args.gene_neighbor_prior_min_count))
    coverage = prior_mask.to(torch.float32).mean() if prior_mask.numel() else zero
    if not bool(prior_mask.any()):
        return zero, {
            "loss_neighbor_prior": zero,
            "loss_neighbor_prior_cell": zero,
            "loss_neighbor_prior_text": zero,
            "neighbor_prior_coverage": coverage.detach(),
            "neighbor_prior_mean_count": zero,
            "neighbor_prior_mean_weight": zero,
        }
    prior_z = F.normalize(prior_proj(prior_vec), dim=-1)
    cell_z = out["cell_set"]  # type: ignore[assignment]
    text_z = out["text_set"]  # type: ignore[assignment]
    loss_cell_vec = 1.0 - F.cosine_similarity(cell_z, prior_z, dim=-1)
    loss_text_vec = 1.0 - F.cosine_similarity(text_z, prior_z, dim=-1)
    valid_counts = prior_counts[prior_mask].clamp_min(1.0)
    if args.gene_neighbor_prior_count_weighting == "none":
        valid_weight = torch.ones_like(valid_counts)
    elif args.gene_neighbor_prior_count_weighting == "log":
        valid_weight = torch.log1p(valid_counts)
    elif args.gene_neighbor_prior_count_weighting == "sqrt":
        valid_weight = torch.sqrt(valid_counts)
    elif args.gene_neighbor_prior_count_weighting == "linear":
        valid_weight = valid_counts
    else:
        raise ValueError(f"Unsupported gene_neighbor_prior_count_weighting={args.gene_neighbor_prior_count_weighting}")
    valid_weight = valid_weight / valid_weight.mean().clamp_min(1e-8)
    loss_cell = (loss_cell_vec[prior_mask] * valid_weight).sum() / valid_weight.sum().clamp_min(1e-8)
    loss_text = (loss_text_vec[prior_mask] * valid_weight).sum() / valid_weight.sum().clamp_min(1e-8)
    total = args.lambda_neighbor_prior_cell * loss_cell + args.lambda_neighbor_prior_text * loss_text
    return total, {
        "loss_neighbor_prior": total.detach(),
        "loss_neighbor_prior_cell": loss_cell.detach(),
        "loss_neighbor_prior_text": loss_text.detach(),
        "neighbor_prior_coverage": coverage.detach(),
        "neighbor_prior_mean_count": valid_counts.mean().detach(),
        "neighbor_prior_mean_weight": valid_weight.mean().detach(),
    }


def gene_neighbor_prior_margin_loss(
    batch: list[Stage1Example],
    out: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    prior_store: GeneNeighborPriorStore | None,
    negative_prior_store: GeneNeighborPriorStore | None,
    prior_proj: nn.Module | None,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = out["cell_set"].device  # type: ignore[union-attr]
    zero = torch.tensor(0.0, device=device)
    if (
        args.lambda_neighbor_prior_margin <= 0
        or prior_store is None
        or negative_prior_store is None
        or prior_proj is None
    ):
        return zero, {
            "loss_neighbor_prior_margin": zero,
            "neighbor_prior_margin_coverage": zero,
            "neighbor_prior_margin_acc": zero,
            "neighbor_prior_margin_gap": zero,
        }

    genes = [example.dist.gene for example in batch]
    pos_vec, pos_mask, pos_counts = batch_gene_neighbor_prior(prior_store, genes, device)
    neg_vec, neg_mask, neg_counts = batch_gene_neighbor_prior(negative_prior_store, genes, device)
    valid = pos_mask & neg_mask
    if args.gene_neighbor_prior_min_count > 1:
        threshold = float(args.gene_neighbor_prior_min_count)
        valid = valid & (pos_counts >= threshold) & (neg_counts >= threshold)
    coverage = valid.to(torch.float32).mean() if valid.numel() else zero
    if not bool(valid.any()):
        return zero, {
            "loss_neighbor_prior_margin": zero,
            "neighbor_prior_margin_coverage": coverage.detach(),
            "neighbor_prior_margin_acc": zero,
            "neighbor_prior_margin_gap": zero,
        }

    cell_z = F.normalize(out["cell_set"], dim=-1)  # type: ignore[arg-type]
    pos_z = F.normalize(prior_proj(pos_vec), dim=-1)
    neg_z = F.normalize(prior_proj(neg_vec), dim=-1)
    pos_sim = torch.sum(cell_z * pos_z, dim=-1)
    neg_sim = torch.sum(cell_z * neg_z, dim=-1)
    gap = pos_sim - neg_sim
    margin_loss = F.relu(float(args.gene_neighbor_prior_margin) - gap[valid]).mean()
    loss = args.lambda_neighbor_prior_margin * margin_loss
    return loss, {
        "loss_neighbor_prior_margin": margin_loss.detach(),
        "neighbor_prior_margin_coverage": coverage.detach(),
        "neighbor_prior_margin_acc": (gap[valid] > 0).to(torch.float32).mean().detach(),
        "neighbor_prior_margin_gap": gap[valid].mean().detach(),
    }


def apply_gene_neighbor_prior_cell_attention(
    batch: list[Stage1Example],
    out: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    prior_store: GeneNeighborPriorStore | None,
    prior_cell_query_proj: nn.Module | None,
    prior_cell_adapter: nn.Module | None,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    device = out["cell_set"].device  # type: ignore[union-attr]
    zero = torch.tensor(0.0, device=device)
    if (
        prior_store is None
        or prior_cell_query_proj is None
        or prior_cell_adapter is None
        or (args.lambda_neighbor_prior_cell_attn <= 0 and args.gene_neighbor_prior_cell_attn_fusion_alpha <= 0)
    ):
        return {
            "neighbor_prior_cell_attn_coverage": zero,
            "neighbor_prior_cell_attn_entropy": zero,
            "neighbor_prior_cell_attn_alpha": zero,
        }

    genes = [example.dist.gene for example in batch]
    prior_vec, prior_mask, prior_counts = batch_gene_neighbor_prior(prior_store, genes, device)
    if args.gene_neighbor_prior_min_count > 1:
        prior_mask = prior_mask & (prior_counts >= float(args.gene_neighbor_prior_min_count))
    coverage = prior_mask.to(torch.float32).mean() if prior_mask.numel() else zero
    if not bool(prior_mask.any()):
        out["prior_cell_set"] = out["cell_set"]  # type: ignore[assignment]
        out["prior_cell_mask"] = prior_mask
        return {
            "neighbor_prior_cell_attn_coverage": coverage.detach(),
            "neighbor_prior_cell_attn_entropy": zero,
            "neighbor_prior_cell_attn_alpha": zero,
        }

    cell_emb = out["cell_embeddings"]  # type: ignore[assignment]
    cell_mask = out["cell_mask"]  # type: ignore[assignment]
    query = F.normalize(prior_cell_query_proj(prior_vec), dim=-1)
    cell_norm = F.normalize(cell_emb, dim=-1)
    logits = torch.sum(cell_norm * query.unsqueeze(1), dim=-1)
    logits = logits / max(float(args.gene_neighbor_prior_cell_attn_temperature), 1e-6)
    logits = logits.masked_fill(~cell_mask, -1e9)
    weights = torch.softmax(logits, dim=-1).masked_fill(~cell_mask, 0.0)
    pooled = torch.sum(cell_emb * weights.unsqueeze(-1), dim=1)
    prior_cell_z = F.normalize(prior_cell_adapter(pooled), dim=-1)
    out["prior_cell_set"] = prior_cell_z
    out["prior_cell_mask"] = prior_mask

    alpha = float(args.gene_neighbor_prior_cell_attn_fusion_alpha)
    if alpha > 0:
        base = F.normalize(out["cell_set"], dim=-1)  # type: ignore[arg-type]
        alpha_scale = torch.ones_like(prior_mask, dtype=torch.float32, device=device)
        if args.gene_neighbor_prior_cell_attn_gate == "agreement":
            agreement = torch.sum(base * prior_cell_z, dim=-1)
            alpha_scale = torch.sigmoid(
                (agreement - float(args.gene_neighbor_prior_cell_attn_gate_threshold))
                / max(float(args.gene_neighbor_prior_cell_attn_gate_temperature), 1e-6)
            )
            alpha_scale = torch.where(prior_mask, alpha_scale, torch.zeros_like(alpha_scale))
            out["prior_cell_gate"] = alpha_scale
        fused = F.normalize(base + (alpha * alpha_scale).unsqueeze(-1) * prior_cell_z, dim=-1)
        out["cell_set"] = torch.where(prior_mask.unsqueeze(-1), fused, base)

    valid_weights = weights[prior_mask]
    entropy = (
        -(valid_weights * torch.log(valid_weights.clamp_min(1e-8))).sum(dim=-1).mean()
        if valid_weights.numel()
        else zero
    )
    return {
        "neighbor_prior_cell_attn_coverage": coverage.detach(),
        "neighbor_prior_cell_attn_entropy": entropy.detach(),
        "neighbor_prior_cell_attn_alpha": torch.tensor(alpha, device=device),
        "neighbor_prior_cell_attn_gate_mean": (
            out["prior_cell_gate"][prior_mask].mean().detach()  # type: ignore[index]
            if "prior_cell_gate" in out and bool(prior_mask.any())
            else zero
        ),
    }


def gene_neighbor_prior_cell_attention_loss(
    out: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = out["cell_set"].device  # type: ignore[union-attr]
    zero = torch.tensor(0.0, device=device)
    if args.lambda_neighbor_prior_cell_attn <= 0 or "prior_cell_set" not in out or "prior_cell_mask" not in out:
        return zero, {"loss_neighbor_prior_cell_attn": zero}
    mask = out["prior_cell_mask"]  # type: ignore[assignment]
    if not bool(mask.any()) or int(mask.to(torch.int64).sum().item()) < 2:
        return zero, {"loss_neighbor_prior_cell_attn": zero}
    prior_cell_z = out["prior_cell_set"][mask]  # type: ignore[index]
    text_z = out["text_set"][mask]  # type: ignore[index]
    loss = set_align_loss(
        prior_cell_z,
        text_z,
        temperature_nce=args.tau_nce,
        temperature_ce=args.tau_ce,
    )
    return args.lambda_neighbor_prior_cell_attn * loss, {
        "loss_neighbor_prior_cell_attn": loss.detach(),
    }


def apply_gene_neighbor_prior_fusion(
    batch: list[Stage1Example],
    out: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    prior_store: GeneNeighborPriorStore | None,
    prior_proj: nn.Module | None,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    device = out["cell_set"].device  # type: ignore[union-attr]
    zero = torch.tensor(0.0, device=device)
    if (
        prior_store is None
        or prior_proj is None
        or args.gene_neighbor_prior_fusion == "none"
        or args.gene_neighbor_prior_fusion_alpha <= 0
    ):
        return {
            "neighbor_prior_fusion_coverage": zero,
            "neighbor_prior_fusion_alpha": zero,
        }

    genes = [example.dist.gene for example in batch]
    prior_vec, prior_mask, prior_counts = batch_gene_neighbor_prior(prior_store, genes, device)
    if args.gene_neighbor_prior_min_count > 1:
        prior_mask = prior_mask & (prior_counts >= float(args.gene_neighbor_prior_min_count))
    coverage = prior_mask.to(torch.float32).mean() if prior_mask.numel() else zero
    if not bool(prior_mask.any()):
        return {
            "neighbor_prior_fusion_coverage": coverage.detach(),
            "neighbor_prior_fusion_alpha": zero,
        }

    prior_z = F.normalize(prior_proj(prior_vec), dim=-1)
    alpha = float(args.gene_neighbor_prior_fusion_alpha)
    alpha_mask = prior_mask.to(torch.float32).unsqueeze(-1) * alpha

    def _fuse(base: torch.Tensor) -> torch.Tensor:
        base_z = F.normalize(base, dim=-1)
        fused = F.normalize(base_z + alpha_mask * prior_z, dim=-1)
        return torch.where(prior_mask.unsqueeze(-1), fused, base_z)

    if args.gene_neighbor_prior_fusion in {"cell", "both"}:
        out["cell_set"] = _fuse(out["cell_set"])  # type: ignore[arg-type]
    if args.gene_neighbor_prior_fusion in {"text", "both"}:
        out["text_set"] = _fuse(out["text_set"])  # type: ignore[arg-type]

    return {
        "neighbor_prior_fusion_coverage": coverage.detach(),
        "neighbor_prior_fusion_alpha": torch.tensor(alpha, device=device),
    }


def _soft_positive_ce_from_scores(
    scores: torch.Tensor,
    positive_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    logits = scores / float(temperature)
    positive_mask = positive_mask.to(device=logits.device, dtype=logits.dtype)
    targets = positive_mask / positive_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    log_probs = F.log_softmax(logits, dim=1)
    return -(targets * log_probs).sum(dim=1).mean()


def _uses_text_aware_set(args: argparse.Namespace) -> bool:
    return bool(
        args.lambda_text_aware_set > 0
        or args.lambda_fused_set > 0
        or args.fused_set_beta > 0
    )


def _score_matrix_alignment_loss(
    scores: torch.Tensor,
    batch: list[Stage1Example],
    args: argparse.Namespace,
    temperature: float,
) -> torch.Tensor:
    if args.align_loss == "multi_positive":
        labels = np.asarray([example.dist.gene for example in batch], dtype=object)
        positive_mask = torch.tensor(labels[:, None] == labels[None, :], dtype=torch.bool, device=scores.device)
        return 0.5 * (
            _soft_positive_ce_from_scores(scores, positive_mask, temperature)
            + _soft_positive_ce_from_scores(scores.T, positive_mask.T, temperature)
        )

    target = torch.arange(scores.shape[0], dtype=torch.long, device=scores.device)
    logits = scores / float(temperature)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


def text_aware_set_loss(
    model: Stage1AlignmentModel,
    batch: list[Stage1Example],
    out: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = out["cell_set"].new_tensor(0.0)  # type: ignore[union-attr]
    if args.lambda_text_aware_set <= 0:
        return zero, {
            "loss_text_aware_set": zero.detach(),
            "text_aware_attn_entropy": zero.detach(),
        }
    scores, weights = model.text_aware_set_scorer(
        out["cell_instance_set"],  # type: ignore[arg-type]
        out["cell_mask"],  # type: ignore[arg-type]
        out["text_set"],  # type: ignore[arg-type]
        instance_scores=out["cell_instance_scores"],  # type: ignore[arg-type]
        alpha=args.text_aware_instance_alpha,
    )
    if args.align_loss == "multi_positive":
        labels = np.asarray([example.dist.gene for example in batch], dtype=object)
        positive_mask = torch.tensor(labels[:, None] == labels[None, :], dtype=torch.bool, device=scores.device)
        loss = _soft_positive_ce_from_scores(scores, positive_mask, args.text_aware_temperature)
    else:
        target = torch.arange(scores.shape[0], dtype=torch.long, device=scores.device)
        loss = F.cross_entropy(scores / float(args.text_aware_temperature), target)
    entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1).mean()
    return loss, {
        "loss_text_aware_set": loss.detach(),
        "text_aware_attn_entropy": entropy.detach(),
    }


def fused_set_loss(
    model: Stage1AlignmentModel,
    batch: list[Stage1Example],
    out: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = out["cell_set"].new_tensor(0.0)  # type: ignore[union-attr]
    if args.lambda_fused_set <= 0:
        return zero, {
            "loss_fused_set": zero.detach(),
            "fused_text_aware_attn_entropy": zero.detach(),
        }

    cell_set = F.normalize(out["cell_set"], dim=-1)  # type: ignore[arg-type]
    text_set = F.normalize(out["text_set"], dim=-1)  # type: ignore[arg-type]
    set_scores = cell_set @ text_set.T
    text_aware_scores, weights = model.text_aware_set_scorer(
        out["cell_instance_set"],  # type: ignore[arg-type]
        out["cell_mask"],  # type: ignore[arg-type]
        out["text_set"],  # type: ignore[arg-type]
        instance_scores=out["cell_instance_scores"],  # type: ignore[arg-type]
        alpha=args.text_aware_instance_alpha,
    )
    fused_scores = set_scores + float(args.fused_set_beta) * text_aware_scores
    loss = _score_matrix_alignment_loss(fused_scores, batch, args, args.fused_set_temperature)
    entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1).mean()
    return loss, {
        "loss_fused_set": loss.detach(),
        "fused_text_aware_attn_entropy": entropy.detach(),
    }


def text_noise_conf_loss(
    batch: list[Stage1Example],
    out: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    conf = out["text_aux"]["conf"]  # type: ignore[index]
    text_indices = out["text_indices"]  # type: ignore[assignment]
    zero = conf.new_tensor(0.0)
    zero_parts = {
        "loss_text_noise_conf": zero.detach(),
        "text_noise_conf_pairs": zero.detach(),
        "text_noise_conf_groups": zero.detach(),
        "text_noise_conf_margin_acc": zero.detach(),
    }
    if args.lambda_text_noise_conf <= 0:
        return zero, zero_parts

    text_indices_cpu = text_indices.detach().cpu().numpy()
    losses: list[torch.Tensor] = []
    acc_values: list[torch.Tensor] = []
    pair_count = 0
    group_count = 0
    margin = float(args.text_noise_conf_margin)
    for row_idx, example in enumerate(batch):
        clean_pos: list[int] = []
        noise_pos: list[int] = []
        for local_idx, evidence_idx_raw in enumerate(text_indices_cpu[row_idx]):
            evidence_idx = int(evidence_idx_raw)
            if evidence_idx < 0:
                continue
            metadata = (
                example.evidence.metadata[evidence_idx]
                if evidence_idx < len(example.evidence.metadata)
                else {}
            )
            if str(metadata.get("is_noise", "0")) == "1":
                noise_pos.append(local_idx)
            else:
                clean_pos.append(local_idx)
        if not clean_pos or not noise_pos:
            continue
        clean_conf = conf[row_idx, torch.as_tensor(clean_pos, dtype=torch.long, device=conf.device)]
        noise_conf = conf[row_idx, torch.as_tensor(noise_pos, dtype=torch.long, device=conf.device)]
        pair_gap = clean_conf[:, None] - noise_conf[None, :]
        losses.append(F.relu(margin - pair_gap).mean())
        acc_values.append((pair_gap > 0).to(conf.dtype).mean())
        pair_count += int(pair_gap.numel())
        group_count += 1

    if not losses:
        return zero, zero_parts

    loss = torch.stack(losses).mean()
    margin_acc = torch.stack(acc_values).mean()
    return loss, {
        "loss_text_noise_conf": loss.detach(),
        "text_noise_conf_pairs": conf.new_tensor(float(pair_count)),
        "text_noise_conf_groups": conf.new_tensor(float(group_count)),
        "text_noise_conf_margin_acc": margin_acc.detach(),
    }


def compute_losses(
    model: Stage1AlignmentModel,
    batch: list[Stage1Example],
    out: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    args: argparse.Namespace,
    epoch: int,
    memory_bank: ContrastiveMemoryBank | None = None,
    out_b: dict[str, torch.Tensor | dict[str, torch.Tensor]] | None = None,
    prior_store: GeneNeighborPriorStore | None = None,
    negative_prior_store: GeneNeighborPriorStore | None = None,
    prior_proj: nn.Module | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if args.align_loss == "multi_positive":
        labels = np.asarray([example.dist.gene for example in batch], dtype=object)
        positive_mask = torch.tensor(labels[:, None] == labels[None, :], dtype=torch.bool, device=out["cell_single"].device)  # type: ignore[union-attr]
        loss_single = multi_positive_align_loss(
            out["cell_single"],  # type: ignore[arg-type]
            out["text_single"],  # type: ignore[arg-type]
            positive_mask,
            temperature_nce=args.tau_nce,
            temperature_ce=args.tau_ce,
        )
        loss_set = multi_positive_align_loss(
            out["cell_set"],  # type: ignore[arg-type]
            out["text_set"],  # type: ignore[arg-type]
            positive_mask,
            temperature_nce=args.tau_nce,
            temperature_ce=args.tau_ce,
        )
    else:
        loss_single = single_align_loss(
            out["cell_single"],  # type: ignore[arg-type]
            out["text_single"],  # type: ignore[arg-type]
            temperature_nce=args.tau_nce,
            temperature_ce=args.tau_ce,
        )
        loss_set = set_align_loss(
            out["cell_set"],  # type: ignore[arg-type]
            out["text_set"],  # type: ignore[arg-type]
            temperature_nce=args.tau_nce,
            temperature_ce=args.tau_ce,
        )
    loss_denoise, denoise_parts = denoise_loss(
        out["cell_aux"],  # type: ignore[arg-type]
        out["text_aux"],  # type: ignore[arg-type]
        out["cell_mask"],  # type: ignore[arg-type]
        out["text_mask"],  # type: ignore[arg-type]
        lambda_prior=args.lambda_prior,
        lambda_entropy=args.lambda_entropy,
        lambda_cell=args.lambda_cell_denoise,
        lambda_text=args.lambda_text_denoise,
    )
    loss_struct, struct_parts = structure_consistency_loss(
        out["cell_embeddings"],  # type: ignore[arg-type]
        out["text_embeddings"],  # type: ignore[arg-type]
        out["cell_aux"]["weights"],  # type: ignore[index]
        out["text_aux"]["weights"],  # type: ignore[index]
        out["cell_mask"],  # type: ignore[arg-type]
        out["text_mask"],  # type: ignore[arg-type]
        epsilon=args.struct_epsilon,
        max_iter=args.struct_iter,
        tau_cell=args.tau_struct_cell,
        tau_text=args.tau_struct_text,
        max_pairs=args.struct_max_pairs,
        huber_delta=args.struct_huber_delta,
        loss_mode=args.struct_mode,
        rank_margin=args.struct_rank_margin,
    )
    loss_cons, cons_parts = augmentation_consistency_loss(model, out, args)
    loss_anchor, anchor_parts = anchor_consistency_loss(out, args)
    loss_neighbor_struct, neighbor_parts = neighbor_structure_loss(out, args)
    loss_neighbor_contrast, neighbor_contrast_parts = neighbor_contrastive_loss(out, args)
    labels_list = [example.dist.gene for example in batch]
    loss_memory, memory_parts = memory_bank_align_loss(
        out["cell_set"],  # type: ignore[arg-type]
        out["text_set"],  # type: ignore[arg-type]
        labels_list,
        memory_bank,
        args.tau_memory,
    )
    loss_multi_view, multi_view_parts = (
        multi_view_consistency_loss(
            out["cell_set"],  # type: ignore[arg-type]
            out_b["cell_set"],  # type: ignore[arg-type]
            out["text_set"],  # type: ignore[arg-type]
            out_b["text_set"],  # type: ignore[arg-type]
        )
        if out_b is not None
        else (out["cell_set"].new_tensor(0.0),  # type: ignore[union-attr]
              {"loss_multi_view": out["cell_set"].new_tensor(0.0),  # type: ignore[union-attr]
               "loss_mv_cell": out["cell_set"].new_tensor(0.0),  # type: ignore[union-attr]
               "loss_mv_text": out["cell_set"].new_tensor(0.0)})  # type: ignore[union-attr]
    )
    loss_neighbor_soft, neighbor_soft_parts = _neighbor_soft_loss(
        batch, out, args
    )
    loss_expr_strat, expr_parts = _expr_stratified_loss(model, out, args)
    loss_neighbor_prior, neighbor_prior_parts = gene_neighbor_prior_loss(
        batch,
        out,
        prior_store=prior_store,
        prior_proj=prior_proj,
        args=args,
    )
    loss_neighbor_prior_margin, neighbor_prior_margin_parts = gene_neighbor_prior_margin_loss(
        batch,
        out,
        prior_store=prior_store,
        negative_prior_store=negative_prior_store,
        prior_proj=prior_proj,
        args=args,
    )
    loss_neighbor_prior_cell_attn, neighbor_prior_cell_attn_parts = gene_neighbor_prior_cell_attention_loss(
        out,
        args,
    )
    loss_stage3_confusion, stage3_confusion_parts = stage3_confusion_loss(
        out,
        lambda_confusion=args.lambda_stage3_confusion,
        margin=args.stage3_confusion_margin,
        mode=args.stage3_confusion_mode,
        anchor_mode=args.stage3_confusion_anchor,
    )
    loss_text_aware_set, text_aware_parts = text_aware_set_loss(model, batch, out, args)
    loss_fused_set, fused_set_parts = fused_set_loss(model, batch, out, args)
    loss_text_noise_conf, text_noise_conf_parts = text_noise_conf_loss(batch, out, args)
    warmup_scale = min(1.0, float(epoch + 1) / max(float(args.warmup_epochs), 1.0))
    total = (
        args.lambda_single * loss_single
        + args.lambda_set * loss_set
        + args.lambda_text_aware_set * loss_text_aware_set
        + args.lambda_fused_set * loss_fused_set
        + args.lambda_text_noise_conf * loss_text_noise_conf
        + warmup_scale * args.lambda_struct * loss_struct
        + args.lambda_denoise * loss_denoise
        + warmup_scale * loss_cons
        + args.lambda_anchor * loss_anchor
        + warmup_scale * args.lambda_neighbor_struct * loss_neighbor_struct
        + warmup_scale * args.lambda_neighbor_contrast * loss_neighbor_contrast
        + warmup_scale * args.lambda_memory * loss_memory
        + warmup_scale * args.lambda_multi_view * loss_multi_view
        + warmup_scale * args.lambda_neighbor_soft * loss_neighbor_soft
        + warmup_scale * args.lambda_expr_strat * loss_expr_strat
        + warmup_scale * loss_neighbor_prior
        + warmup_scale * loss_neighbor_prior_margin
        + warmup_scale * loss_neighbor_prior_cell_attn
        + warmup_scale * loss_stage3_confusion
    )
    parts = {
        "loss_total": float(total.detach().cpu()),
        "loss_single": float(loss_single.detach().cpu()),
        "loss_set": float(loss_set.detach().cpu()),
        "loss_text_aware_set": float(loss_text_aware_set.detach().cpu()),
        "loss_fused_set": float(loss_fused_set.detach().cpu()),
        "loss_text_noise_conf": float(loss_text_noise_conf.detach().cpu()),
        "loss_struct": float(loss_struct.detach().cpu()),
        "loss_denoise": float(loss_denoise.detach().cpu()),
        "lambda_cell_denoise": float(args.lambda_cell_denoise),
        "lambda_text_denoise": float(args.lambda_text_denoise),
        "loss_consistency": float(loss_cons.detach().cpu()),
        "loss_anchor": float(loss_anchor.detach().cpu()),
        "loss_neighbor_struct": float(loss_neighbor_struct.detach().cpu()),
        "loss_neighbor_contrast": float(loss_neighbor_contrast.detach().cpu()),
        "loss_neighbor_prior": float(loss_neighbor_prior.detach().cpu()),
        "loss_neighbor_prior_margin": float(loss_neighbor_prior_margin.detach().cpu()),
        "loss_neighbor_prior_cell_attn": float(loss_neighbor_prior_cell_attn.detach().cpu()),
        "loss_stage3_confusion": float(loss_stage3_confusion.detach().cpu()),
        "loss_memory": float(loss_memory.detach().cpu()),
        "warmup_scale": float(warmup_scale),
        **{k: float(v.cpu()) for k, v in denoise_parts.items()},
        **{k: float(v.cpu()) for k, v in struct_parts.items()},
        **{k: float(v.cpu()) for k, v in cons_parts.items()},
        **{k: float(v.cpu()) for k, v in anchor_parts.items()},
        **{k: float(v.cpu()) for k, v in neighbor_parts.items()},
        **{k: float(v.cpu()) for k, v in neighbor_contrast_parts.items()},
        **{k: float(v.cpu()) for k, v in memory_parts.items()},
        **{k: float(v.cpu()) for k, v in multi_view_parts.items()},
        **{k: float(v.cpu()) for k, v in neighbor_soft_parts.items()},
        **{k: float(v.cpu()) for k, v in expr_parts.items()},
        **{k: float(v.cpu()) for k, v in neighbor_prior_parts.items()},
        **{k: float(v.cpu()) for k, v in neighbor_prior_margin_parts.items()},
        **{k: float(v.cpu()) for k, v in neighbor_prior_cell_attn_parts.items()},
        **{k: float(v.cpu()) for k, v in stage3_confusion_parts.items()},
        **{k: float(v.cpu()) for k, v in text_aware_parts.items()},
        **{k: float(v.cpu()) for k, v in fused_set_parts.items()},
        **{k: float(v.cpu()) for k, v in text_noise_conf_parts.items()},
    }
    # Log per-step mean confidence for cell / text set encoders
    cell_aux = out.get("cell_aux")
    if isinstance(cell_aux, dict) and "conf" in cell_aux:
        cell_conf = cell_aux["conf"]  # type: ignore[index]
        cell_mask = out.get("cell_mask")
        if cell_mask is not None:
            parts["cell_conf_mean"] = float(cell_conf[cell_mask].mean().detach().cpu())  # type: ignore[index]
        else:
            parts["cell_conf_mean"] = float(cell_conf.mean().detach().cpu())  # type: ignore[index]
    text_aux = out.get("text_aux")
    if isinstance(text_aux, dict) and "conf" in text_aux:
        text_conf = text_aux["conf"]  # type: ignore[index]
        text_mask = out.get("text_mask")
        if text_mask is not None:
            parts["text_conf_mean"] = float(text_conf[text_mask].mean().detach().cpu())  # type: ignore[index]
        else:
            parts["text_conf_mean"] = float(text_conf.mean().detach().cpu())  # type: ignore[index]
    return total, parts


def _dropout_mask(mask: torch.Tensor, drop_prob: float) -> torch.Tensor:
    if drop_prob <= 0:
        return mask
    keep = torch.rand(mask.shape, device=mask.device) >= drop_prob
    out = mask & keep
    empty = out.sum(dim=1) == 0
    if bool(empty.any()):
        first_valid = mask.float().argmax(dim=1)
        rows = torch.where(empty)[0]
        out[rows, first_valid[rows]] = True
    return out


def augmentation_consistency_loss(
    model: Stage1AlignmentModel,
    out: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cell_loss = out["cell_embeddings"].new_tensor(0.0)  # type: ignore[union-attr]
    text_loss = out["cell_embeddings"].new_tensor(0.0)  # type: ignore[union-attr]

    if args.lambda_cons_cell > 0:
        cell_mask_aug = _dropout_mask(out["cell_mask"], args.cell_dropout)  # type: ignore[arg-type]
        expr_aug = out["cell_expr"]  # type: ignore[assignment]
        if args.expr_noise_std > 0:
            expr_aug = expr_aug + torch.randn_like(expr_aug) * args.expr_noise_std
        cell_aug = model.cell_set(
            out["cell_embeddings"],  # type: ignore[arg-type]
            out["cell_ranks"],  # type: ignore[arg-type]
            expr_aug,  # type: ignore[arg-type]
            cell_mask_aug,
            out["cell_occurrence"],  # type: ignore[arg-type]
        )
        cell_aug_z = F.normalize(model.cell_set_proj(cell_aug["z"]), dim=-1)
        cell_loss = F.mse_loss(cell_aug_z, out["cell_set_base"].detach())  # type: ignore[arg-type]

    if args.lambda_cons_text > 0:
        text_mask_aug = _dropout_mask(out["text_mask"], args.text_dropout)  # type: ignore[arg-type]
        text_aug = model.text_set(
            out["text_embeddings"],  # type: ignore[arg-type]
            out["text_prior"],  # type: ignore[arg-type]
            text_mask_aug,
            out["text_metadata"],  # type: ignore[arg-type]
        )
        text_aug_z = F.normalize(model.text_set_proj(text_aug["z"]), dim=-1)
        text_loss = F.mse_loss(text_aug_z, out["text_set_base"].detach())  # type: ignore[arg-type]

    loss = args.lambda_cons_cell * cell_loss + args.lambda_cons_text * text_loss
    return loss, {"loss_cons_cell": cell_loss.detach(), "loss_cons_text": text_loss.detach()}


def anchor_consistency_loss(
    out: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if args.lambda_anchor <= 0:
        zero = out["cell_set"].new_tensor(0.0)  # type: ignore[union-attr]
        return zero, {"loss_anchor_cell": zero.detach(), "loss_anchor_text": zero.detach()}

    cell_target = out["cell_single"]  # type: ignore[assignment]
    text_target = out["text_single"]  # type: ignore[assignment]
    if not args.anchor_no_detach:
        cell_target = cell_target.detach()
        text_target = text_target.detach()
    cell_loss = 1.0 - F.cosine_similarity(out["cell_set"], cell_target, dim=-1).mean()  # type: ignore[arg-type]
    text_loss = 1.0 - F.cosine_similarity(out["text_set"], text_target, dim=-1).mean()  # type: ignore[arg-type]
    return cell_loss + text_loss, {"loss_anchor_cell": cell_loss.detach(), "loss_anchor_text": text_loss.detach()}


def neighbor_structure_loss(
    out: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if args.lambda_neighbor_struct <= 0 or "neighbor_mask" not in out or "neighbor_text_set" not in out:
        zero = out["cell_set"].new_tensor(0.0)  # type: ignore[union-attr]
        return zero, {
            "neighbor_rows": zero.detach(),
            "neighbor_coverage": zero.detach(),
            "neighbor_cell_entropy": zero.detach(),
            "neighbor_text_entropy": zero.detach(),
            "neighbor_prior_std": zero.detach(),
        }

    mask = out["neighbor_mask"]  # type: ignore[assignment]
    valid_rows = mask.any(dim=1)
    if not bool(valid_rows.any()):
        zero = out["cell_set"].new_tensor(0.0)  # type: ignore[union-attr]
        return zero, {
            "neighbor_rows": zero.detach(),
            "neighbor_coverage": zero.detach(),
            "neighbor_cell_entropy": zero.detach(),
            "neighbor_text_entropy": zero.detach(),
            "neighbor_prior_std": zero.detach(),
        }

    target_cell = out["cell_set_base"]  # type: ignore[assignment]
    target_text = out["text_set_base"]  # type: ignore[assignment]
    neighbor_cell = out["neighbor_cell_set"]  # type: ignore[assignment]
    neighbor_text = out["neighbor_text_set"]  # type: ignore[assignment]
    cell_logits = torch.sum(target_cell.unsqueeze(1) * neighbor_cell, dim=-1) / max(args.tau_neighbor_struct, 1e-6)
    text_logits = torch.sum(target_text.unsqueeze(1) * neighbor_text, dim=-1) / max(args.tau_neighbor_struct, 1e-6)
    neighbor_prior = out.get("neighbor_prior_logits")
    if neighbor_prior is not None:
        prior_bias = neighbor_prior.to(cell_logits.dtype) / max(args.tau_neighbor_prior, 1e-6)  # type: ignore[union-attr]
        cell_logits = cell_logits + prior_bias
        text_logits = text_logits + prior_bias
    cell_prob = _masked_softmax(cell_logits, mask)
    text_prob = _masked_softmax(text_logits, mask)
    cell_prob_valid = cell_prob[valid_rows]
    text_prob_valid = text_prob[valid_rows]
    kl_cell_text = torch.sum(cell_prob_valid * (torch.log(cell_prob_valid + 1e-8) - torch.log(text_prob_valid + 1e-8)), dim=-1)
    kl_text_cell = torch.sum(text_prob_valid * (torch.log(text_prob_valid + 1e-8) - torch.log(cell_prob_valid + 1e-8)), dim=-1)
    loss = 0.5 * (kl_cell_text + kl_text_cell).mean()
    cell_entropy = -torch.sum(cell_prob_valid * torch.log(cell_prob_valid + 1e-8), dim=-1).mean()
    text_entropy = -torch.sum(text_prob_valid * torch.log(text_prob_valid + 1e-8), dim=-1).mean()
    return loss, {
        "neighbor_rows": mask.sum().to(target_cell.dtype).detach(),
        "neighbor_coverage": valid_rows.to(target_cell.dtype).mean().detach(),
        "neighbor_cell_entropy": cell_entropy.detach(),
        "neighbor_text_entropy": text_entropy.detach(),
        "neighbor_prior_std": (
            neighbor_prior[mask].std(unbiased=False).detach()
            if neighbor_prior is not None and bool(mask.any())
            else target_cell.new_tensor(0.0)
        ),
    }


def neighbor_contrastive_loss(
    out: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = out["cell_set"].new_tensor(0.0)  # type: ignore[union-attr]
    if args.lambda_neighbor_contrast <= 0 or "neighbor_mask" not in out or "neighbor_text_set" not in out:
        return zero, {
            "neighbor_contrast_rows": zero.detach(),
            "neighbor_contrast_cell_acc": zero.detach(),
            "neighbor_contrast_text_acc": zero.detach(),
        }

    mask = out["neighbor_mask"]  # type: ignore[assignment]
    valid_rows = mask.any(dim=1)
    if not bool(valid_rows.any()):
        return zero, {
            "neighbor_contrast_rows": zero.detach(),
            "neighbor_contrast_cell_acc": zero.detach(),
            "neighbor_contrast_text_acc": zero.detach(),
        }

    tau = max(float(args.tau_neighbor_contrast), 1e-6)
    cell_set = out["cell_set"]  # type: ignore[assignment]
    text_set = out["text_set"]  # type: ignore[assignment]
    neighbor_cell = out["neighbor_cell_set"].detach()  # type: ignore[union-attr]
    neighbor_text = out["neighbor_text_set"].detach()  # type: ignore[union-attr]

    pos_cell_raw = torch.sum(cell_set * text_set, dim=-1, keepdim=True)
    neg_cell_raw = torch.sum(cell_set.unsqueeze(1) * neighbor_text, dim=-1).masked_fill(~mask, -1e9)
    pos_text_raw = torch.sum(text_set * cell_set, dim=-1, keepdim=True)
    neg_text_raw = torch.sum(text_set.unsqueeze(1) * neighbor_cell, dim=-1).masked_fill(~mask, -1e9)

    if args.neighbor_contrast_mode == "margin":
        margin = float(args.neighbor_contrast_margin)
        cell_hinge = F.relu(margin + neg_cell_raw[valid_rows] - pos_cell_raw[valid_rows])
        text_hinge = F.relu(margin + neg_text_raw[valid_rows] - pos_text_raw[valid_rows])
        cell_hinge = cell_hinge.masked_fill(~mask[valid_rows], 0.0)
        text_hinge = text_hinge.masked_fill(~mask[valid_rows], 0.0)
        denom = mask[valid_rows].sum().to(cell_set.dtype).clamp_min(1.0)
        loss = 0.5 * (cell_hinge.sum() / denom + text_hinge.sum() / denom)
        cell_acc = (pos_cell_raw[valid_rows] > neg_cell_raw[valid_rows].amax(dim=1, keepdim=True)).to(cell_set.dtype).mean()
        text_acc = (pos_text_raw[valid_rows] > neg_text_raw[valid_rows].amax(dim=1, keepdim=True)).to(cell_set.dtype).mean()
    else:
        pos_cell = pos_cell_raw / tau
        neg_cell = neg_cell_raw / tau
        logits_cell = torch.cat([pos_cell, neg_cell], dim=1)

        pos_text = pos_text_raw / tau
        neg_text = neg_text_raw / tau
        logits_text = torch.cat([pos_text, neg_text], dim=1)

        logits_cell_valid = logits_cell[valid_rows]
        logits_text_valid = logits_text[valid_rows]
        targets = torch.zeros(logits_cell_valid.shape[0], dtype=torch.long, device=cell_set.device)
        loss = 0.5 * (F.cross_entropy(logits_cell_valid, targets) + F.cross_entropy(logits_text_valid, targets))
        cell_acc = (logits_cell_valid.argmax(dim=1) == 0).to(cell_set.dtype).mean()
        text_acc = (logits_text_valid.argmax(dim=1) == 0).to(cell_set.dtype).mean()
    return loss, {
        "neighbor_contrast_rows": valid_rows.to(cell_set.dtype).sum().detach(),
        "neighbor_contrast_cell_acc": cell_acc.detach(),
        "neighbor_contrast_text_acc": text_acc.detach(),
    }


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_logits = logits.masked_fill(~mask, -1e9)
    probs = torch.softmax(masked_logits, dim=-1)
    return probs.masked_fill(~mask, 0.0)


def _build_conf_filter_mask(
    conf: torch.Tensor,
    mask: torch.Tensor,
    mode: str,
    frac: float,
    min_keep: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    if mode == "none":
        kept = mask.clone()
    elif mode not in {"top_frac", "positive"}:
        raise ValueError(f"Unsupported eval text conf filter mode: {mode}")
    else:
        kept = torch.zeros_like(mask)
        batch_size = int(mask.shape[0])
        for row_idx in range(batch_size):
            valid = torch.nonzero(mask[row_idx], as_tuple=False).flatten()
            n = int(valid.numel())
            if n <= 0:
                continue
            if n <= min_keep:
                keep_idx = valid
            elif mode == "top_frac":
                keep_n = int(np.ceil(n * float(frac)))
                keep_n = max(keep_n, int(min_keep))
                keep_n = min(keep_n, n)
                scores = conf[row_idx, valid]
                local = torch.topk(scores, k=keep_n, largest=True).indices
                keep_idx = valid[local]
            else:
                scores = conf[row_idx, valid]
                positive_idx = valid[scores > 0]
                if int(positive_idx.numel()) >= min_keep:
                    keep_idx = positive_idx
                else:
                    keep_n = min(max(int(min_keep), int(positive_idx.numel())), n)
                    local = torch.topk(scores, k=keep_n, largest=True).indices
                    keep_idx = valid[local]
            kept[row_idx, keep_idx] = True

    valid_counts = mask.sum(dim=1).to(torch.float32)
    keep_counts = kept.sum(dim=1).to(torch.float32)
    drop_counts = (valid_counts - keep_counts).clamp_min(0.0)
    nonempty = valid_counts > 0
    if bool(nonempty.any()):
        mean_keep = float(keep_counts[nonempty].mean().detach().cpu())
        mean_drop = float(drop_counts[nonempty].mean().detach().cpu())
        mean_keep_fraction = float((keep_counts[nonempty] / valid_counts[nonempty].clamp_min(1.0)).mean().detach().cpu())
    else:
        mean_keep = 0.0
        mean_drop = 0.0
        mean_keep_fraction = 0.0
    return kept, {
        "conf_filter_mean_keep": mean_keep,
        "conf_filter_mean_drop": mean_drop,
        "conf_filter_mean_keep_fraction": mean_keep_fraction,
    }


def evaluate(
    model: Stage1AlignmentModel,
    examples: list[Stage1Example],
    args: argparse.Namespace,
    device: torch.device,
    split_name: str,
    prior_store: GeneNeighborPriorStore | None = None,
    prior_proj: nn.Module | None = None,
    prior_cell_query_proj: nn.Module | None = None,
    prior_cell_adapter: nn.Module | None = None,
    collect_rows: bool = True,
    cell_noise_metadata: CellNoiseMetadata | None = None,
) -> tuple[dict[str, float], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    model.eval()
    single_cell: list[np.ndarray] = []
    single_text: list[np.ndarray] = []
    set_cell: list[np.ndarray] = []
    set_text: list[np.ndarray] = []
    set_conf_filtered_text: list[np.ndarray] = []
    conf_filter_parts: list[dict[str, float]] = []
    ta_cell_features: list[torch.Tensor] = []
    ta_cell_masks: list[torch.Tensor] = []
    ta_instance_scores: list[torch.Tensor] = []
    weight_rows: list[dict[str, object]] = []
    cell_weight_rows: list[dict[str, object]] = []

    with torch.no_grad():
        for start in range(0, len(examples), args.batch_size):
            batch = examples[start : start + args.batch_size]
            out = batch_forward(
                model,
                batch,
                args.max_instances,
                args.max_text_evidence,
                device,
                args.text_evidence_sampling,
                args.max_neighbors,
                args.stage3_confusion_max_negatives,
                args.text_prior_temperature,
                args.text_diversity_weight,
                args.set_fusion,
                args.residual_scale,
                args.neighbor_agg_alpha,
                args.instance_score_alpha,
                args.instance_score_mode,
            )
            apply_gene_neighbor_prior_fusion(
                batch,
                out,
                prior_store=prior_store,
                prior_proj=prior_proj,
                args=args,
            )
            apply_gene_neighbor_prior_cell_attention(
                batch,
                out,
                prior_store=prior_store,
                prior_cell_query_proj=prior_cell_query_proj,
                prior_cell_adapter=prior_cell_adapter,
                args=args,
            )
            single_cell.append(out["cell_single"].cpu().numpy())  # type: ignore[union-attr]
            single_text.append(out["text_single"].cpu().numpy())  # type: ignore[union-attr]
            set_cell.append(out["cell_set"].cpu().numpy())  # type: ignore[union-attr]
            set_text.append(out["text_set"].cpu().numpy())  # type: ignore[union-attr]
            conf_filter_keep_mask_cpu: np.ndarray | None = None
            if args.eval_text_conf_filter_mode != "none":
                text_aux_for_filter = out["text_aux"]  # type: ignore[assignment]
                text_mask_for_filter = out["text_mask"]  # type: ignore[assignment]
                keep_mask, filter_part = _build_conf_filter_mask(
                    text_aux_for_filter["conf"],  # type: ignore[index]
                    text_mask_for_filter,  # type: ignore[arg-type]
                    args.eval_text_conf_filter_mode,
                    args.eval_text_conf_filter_frac,
                    args.eval_text_conf_filter_min_keep,
                )
                filtered_text_aux = model.text_set(
                    out["text_embeddings"],  # type: ignore[arg-type]
                    out["text_prior"],  # type: ignore[arg-type]
                    keep_mask,
                    out["text_metadata"],  # type: ignore[arg-type]
                )
                if args.set_fusion == "residual":
                    text_single_hidden = model.text_single(
                        out["text_embeddings"],  # type: ignore[arg-type]
                        out["text_mask"],  # type: ignore[arg-type]
                    )
                    filtered_text_hidden = text_single_hidden + float(args.residual_scale) * filtered_text_aux["z"]
                else:
                    filtered_text_hidden = filtered_text_aux["z"]
                filtered_text_set = F.normalize(model.text_set_proj(filtered_text_hidden), dim=-1)
                set_conf_filtered_text.append(filtered_text_set.cpu().numpy())
                conf_filter_parts.append(filter_part)
                if collect_rows:
                    conf_filter_keep_mask_cpu = keep_mask.cpu().numpy()
            if _uses_text_aware_set(args):
                ta_cell_features.append(out["cell_instance_set"].cpu())  # type: ignore[union-attr]
                ta_cell_masks.append(out["cell_mask"].cpu())  # type: ignore[union-attr]
                ta_instance_scores.append(out["cell_instance_scores"].cpu())  # type: ignore[union-attr]
            if collect_rows:
                text_aux = out["text_aux"]  # type: ignore[assignment]
                cell_aux = out["cell_aux"]  # type: ignore[assignment]
                weights = text_aux["weights"].cpu().numpy()
                prior = text_aux["prior"].cpu().numpy()
                text_conf_gate = text_aux["conf_gate"].cpu().numpy()
                text_gated_prior = text_aux["gated_prior"].cpu().numpy()
                text_conf = text_aux["conf"].cpu().numpy()
                text_content_logit = text_aux["content_logit"].cpu().numpy()
                text_prior_logit = text_aux["prior_logit"].cpu().numpy()
                text_conf_logit = text_aux["conf_logit"].cpu().numpy()
                text_attn_logit = text_aux["attn_logit"].cpu().numpy()
                text_indices = out["text_indices"].cpu().numpy()  # type: ignore[union-attr]
                cell_weights = cell_aux["weights"].cpu().numpy()
                cell_prior = cell_aux["prior"].cpu().numpy()
                cell_conf = cell_aux["conf"].cpu().numpy()
                cell_content_logit = cell_aux["content_logit"].cpu().numpy()
                cell_prior_logit = cell_aux["prior_logit"].cpu().numpy()
                cell_conf_logit = cell_aux["conf_logit"].cpu().numpy()
                cell_instance_score_logit = cell_aux["instance_score_logit"].cpu().numpy()
                cell_attn_logit = cell_aux["attn_logit"].cpu().numpy()
                cell_instance_scores = out["cell_instance_scores"].cpu().numpy()  # type: ignore[union-attr]
                cell_selected_indices = out["cell_selected_indices"].cpu().numpy()  # type: ignore[union-attr]
                for row_idx, example in enumerate(batch):
                    cell_limit = int((cell_selected_indices[row_idx] >= 0).sum())
                    for local_idx in range(cell_limit):
                        instance_idx = int(cell_selected_indices[row_idx, local_idx])
                        noise_meta = (
                            cell_noise_metadata.get((example.dist.key, instance_idx), {})
                            if cell_noise_metadata is not None
                            else {}
                        )
                        cell_weight_rows.append(
                            {
                                "gene": example.dist.gene,
                                "context": example.dist.context,
                                "local_index": local_idx,
                                "instance_index": instance_idx,
                                "split": split_name,
                                "weight": float(cell_weights[row_idx, local_idx]),
                                "prior": float(cell_prior[row_idx, local_idx]),
                                "conf": float(cell_conf[row_idx, local_idx]),
                                "content_logit": float(cell_content_logit[row_idx, local_idx]),
                                "prior_logit": float(cell_prior_logit[row_idx, local_idx]),
                                "conf_logit": float(cell_conf_logit[row_idx, local_idx]),
                                "instance_score_logit": float(cell_instance_score_logit[row_idx, local_idx]),
                                "attn_logit": float(cell_attn_logit[row_idx, local_idx]),
                                "instance_score": float(cell_instance_scores[row_idx, local_idx]),
                                "rank": float(example.dist.ranks[instance_idx]),
                                "expr": float(example.dist.expr[instance_idx]),
                                "is_cell_noise": noise_meta.get("is_cell_noise", "0"),
                                "cell_noise_kind": noise_meta.get("cell_noise_kind", "clean"),
                                "source_gene": noise_meta.get("source_gene", example.dist.gene),
                                "source_context": noise_meta.get("source_context", example.dist.context),
                                "source_instance_index": noise_meta.get("source_instance_index", instance_idx),
                                "target_gene": noise_meta.get("target_gene", example.dist.gene),
                                "target_context": noise_meta.get("target_context", example.dist.context),
                            }
                        )
                    limit = int((text_indices[row_idx] >= 0).sum())
                    for local_idx in range(limit):
                        evidence_idx = int(text_indices[row_idx, local_idx])
                        metadata = (
                            example.evidence.metadata[evidence_idx]
                            if evidence_idx < len(example.evidence.metadata)
                            else {}
                        )
                        weight_rows.append(
                            {
                                "gene": example.dist.gene,
                                "context": example.dist.context,
                                "text_id": example.evidence.text_ids[evidence_idx],
                                "evidence_index": evidence_idx,
                                "local_index": local_idx,
                                "split": split_name,
                                "text": _metadata_text(metadata),
                                "weight": float(weights[row_idx, local_idx]),
                                "prior": float(prior[row_idx, local_idx]),
                                "conf_gate": float(text_conf_gate[row_idx, local_idx]),
                                "gated_prior": float(text_gated_prior[row_idx, local_idx]),
                                "conf": float(text_conf[row_idx, local_idx]),
                                "content_logit": float(text_content_logit[row_idx, local_idx]),
                                "prior_logit": float(text_prior_logit[row_idx, local_idx]),
                                "conf_logit": float(text_conf_logit[row_idx, local_idx]),
                                "attn_logit": float(text_attn_logit[row_idx, local_idx]),
                                "is_noise": metadata.get("is_noise", "0"),
                                "noise_kind": metadata.get("noise_kind", "clean"),
                                "source_gene": metadata.get("source_gene", example.dist.gene),
                                "source_context": metadata.get("source_context", example.dist.context),
                                "source_text_id": metadata.get("source_text_id", example.evidence.text_ids[evidence_idx]),
                                "target_gene": metadata.get("target_gene", example.dist.gene),
                                "target_context": metadata.get("target_context", example.dist.context),
                                "category": metadata.get("category", ""),
                                "conf_filter_keep": (
                                    int(bool(conf_filter_keep_mask_cpu[row_idx, local_idx]))
                                    if conf_filter_keep_mask_cpu is not None
                                    else ""
                                ),
                            }
                        )

    single_cell_arr = np.concatenate(single_cell, axis=0)
    single_text_arr = np.concatenate(single_text, axis=0)
    set_cell_arr = np.concatenate(set_cell, axis=0)
    set_text_arr = np.concatenate(set_text, axis=0)
    set_conf_filtered_text_arr = (
        np.concatenate(set_conf_filtered_text, axis=0)
        if set_conf_filtered_text
        else None
    )
    gene_labels = [example.dist.gene for example in examples]
    metrics = {f"single_{k}": v for k, v in retrieval_metrics_by_label(single_cell_arr, single_text_arr, gene_labels, gene_labels).items()}
    metrics.update({f"set_{k}": v for k, v in retrieval_metrics_by_label(set_cell_arr, set_text_arr, gene_labels, gene_labels).items()})
    if set_conf_filtered_text_arr is not None:
        metrics.update(
            {
                f"set_conf_filtered_{k}": v
                for k, v in retrieval_metrics_by_label(
                    set_cell_arr,
                    set_conf_filtered_text_arr,
                    gene_labels,
                    gene_labels,
                ).items()
            }
        )
        metrics.update(
            {
                f"set_conf_filtered_diag_{k}": v
                for k, v in retrieval_metrics(set_cell_arr, set_conf_filtered_text_arr).items()
            }
        )
        if conf_filter_parts:
            for key in ("conf_filter_mean_keep", "conf_filter_mean_drop", "conf_filter_mean_keep_fraction"):
                metrics[key] = float(np.mean([part[key] for part in conf_filter_parts]))
        metrics["conf_filter_mode"] = args.eval_text_conf_filter_mode
        metrics["conf_filter_frac"] = float(args.eval_text_conf_filter_frac)
        metrics["conf_filter_min_keep"] = float(args.eval_text_conf_filter_min_keep)
    metrics.update({f"single_diag_{k}": v for k, v in retrieval_metrics(single_cell_arr, single_text_arr).items()})
    metrics.update({f"set_diag_{k}": v for k, v in retrieval_metrics(set_cell_arr, set_text_arr).items()})
    metrics["cell_single_set_cos"] = _mean_pairwise_cosine(single_cell_arr, set_cell_arr)
    metrics["text_single_set_cos"] = _mean_pairwise_cosine(single_text_arr, set_text_arr)
    metrics.update(embedding_diagnostics("cell_set", set_cell_arr))
    metrics.update(embedding_diagnostics("text_set", set_text_arr))
    neighbor_counts = np.asarray([min(args.max_neighbors, len(example.neighbors)) for example in examples], dtype=np.float32)
    metrics["eval_neighbor_coverage"] = float(np.mean(neighbor_counts > 0)) if neighbor_counts.size else 0.0
    metrics["eval_mean_neighbors"] = float(np.mean(neighbor_counts)) if neighbor_counts.size else 0.0
    set_cell_norm = set_cell_arr / np.maximum(np.linalg.norm(set_cell_arr, axis=-1, keepdims=True), 1e-8)
    set_text_norm = set_text_arr / np.maximum(np.linalg.norm(set_text_arr, axis=-1, keepdims=True), 1e-8)
    set_scores = set_cell_norm @ set_text_norm.T

    text_aware_scores: np.ndarray | None = None
    fused_set_scores: np.ndarray | None = None
    if _uses_text_aware_set(args) and ta_cell_features:
        text_set_all = torch.as_tensor(set_text_arr, dtype=torch.float32, device=device)
        score_chunks: list[np.ndarray] = []
        with torch.no_grad():
            for cell_feat_cpu, cell_mask_cpu, instance_scores_cpu in zip(ta_cell_features, ta_cell_masks, ta_instance_scores):
                scores_ta, _ = model.text_aware_set_scorer(
                    cell_feat_cpu.to(device),
                    cell_mask_cpu.to(device),
                    text_set_all,
                    instance_scores=instance_scores_cpu.to(device),
                    alpha=args.text_aware_instance_alpha,
                )
                score_chunks.append(scores_ta.cpu().numpy())
        text_aware_scores = np.concatenate(score_chunks, axis=0)
        metrics.update(
            {
                f"text_aware_set_{k}": v
                for k, v in _retrieval_metrics_by_label_from_scores(
                    text_aware_scores,
                    gene_labels,
                    gene_labels,
                ).items()
            }
        )
        metrics.update(
            {
                f"text_aware_set_diag_{k}": v
                for k, v in _retrieval_metrics_from_scores(text_aware_scores).items()
            }
        )
        if args.fused_set_beta > 0:
            fused_set_scores = set_scores + float(args.fused_set_beta) * text_aware_scores
            metrics.update(
                {
                    f"fused_set_{k}": v
                    for k, v in _retrieval_metrics_by_label_from_scores(
                        fused_set_scores,
                        gene_labels,
                        gene_labels,
                    ).items()
                }
            )
            metrics.update(
                {
                    f"fused_set_diag_{k}": v
                    for k, v in _retrieval_metrics_from_scores(fused_set_scores).items()
                }
            )

    prediction_rows: list[dict[str, object]] = []
    if collect_rows:
        if fused_set_scores is not None:
            score_source = "fused_set"
            scores = fused_set_scores
        elif text_aware_scores is not None:
            score_source = "text_aware_set"
            scores = text_aware_scores
        else:
            score_source = "set"
            scores = set_scores
        order = np.argsort(-scores, axis=1)
        for i, example in enumerate(examples):
            positives = np.flatnonzero(np.asarray(gene_labels, dtype=object) == example.dist.gene)
            positive_positions = np.flatnonzero(np.isin(order[i], positives))
            rank = int(positive_positions[0]) + 1 if positive_positions.size else int(np.where(order[i] == i)[0][0]) + 1
            row: dict[str, object] = {
                "split": split_name,
                "key": example.dist.key,
                "gene": example.dist.gene,
                "context": example.dist.context,
                "gold_rank": rank,
                "diag_rank": int(np.where(order[i] == i)[0][0]) + 1,
                "gold_score": float(scores[i, i]),
                "score_source": score_source,
                "num_text_evidence": len(example.evidence.text_ids),
            }
            for top_idx in range(min(5, order.shape[1])):
                pred_idx = int(order[i, top_idx])
                row[f"top{top_idx + 1}_gene"] = examples[pred_idx].dist.gene
                row[f"top{top_idx + 1}_context"] = examples[pred_idx].dist.context
                row[f"top{top_idx + 1}_score"] = float(scores[i, pred_idx])
            prediction_rows.append(row)
    return metrics, prediction_rows, weight_rows, cell_weight_rows


def prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


EPOCH_EVAL_SUMMARY_FIELDNAMES = [
    "epoch",
    "epoch_train_elapsed_sec",
    "epoch_eval_elapsed_sec",
    "loss_total",
    "loss_single",
    "loss_set",
    "loss_text_aware_set",
    "loss_fused_set",
    "loss_text_noise_conf",
    "text_noise_conf_margin_acc",
    "text_noise_conf_groups",
    "text_noise_conf_pairs",
    "eval_set_recall_at_1",
    "eval_set_mrr",
    "eval_set_conf_filtered_recall_at_1",
    "eval_set_conf_filtered_mrr",
    "eval_set_diag_recall_at_1",
    "eval_set_diag_mrr",
    "eval_fused_set_recall_at_1",
    "eval_fused_set_mrr",
    "eval_fused_set_diag_recall_at_1",
    "eval_fused_set_diag_mrr",
    "eval_single_recall_at_1",
    "eval_single_mrr",
    "eval_single_diag_recall_at_1",
    "eval_single_diag_mrr",
    "train_set_recall_at_1",
    "train_set_mrr",
    "train_set_conf_filtered_recall_at_1",
    "train_set_conf_filtered_mrr",
    "train_set_diag_recall_at_1",
    "train_set_diag_mrr",
    "train_fused_set_recall_at_1",
    "train_fused_set_mrr",
    "train_fused_set_diag_recall_at_1",
    "train_fused_set_diag_mrr",
    "train_single_recall_at_1",
    "train_single_mrr",
    "train_single_diag_recall_at_1",
    "train_single_diag_mrr",
    "train_eval_set_recall_at_1_gap",
    "train_eval_set_mrr_gap",
    "train_eval_fused_set_recall_at_1_gap",
    "train_eval_fused_set_mrr_gap",
    "train_eval_single_recall_at_1_gap",
    "train_eval_single_mrr_gap",
]


def epoch_eval_summary_row(row: dict[str, float | int]) -> dict[str, float | int | str]:
    summary: dict[str, float | int | str] = {}
    for field in EPOCH_EVAL_SUMMARY_FIELDNAMES:
        if field.endswith("_gap"):
            continue
        value = row.get(field)
        summary[field] = value if value is not None else ""

    def _gap(train_key: str, eval_key: str) -> float | str:
        train_value = row.get(train_key)
        eval_value = row.get(eval_key)
        if train_value is None or eval_value is None:
            return ""
        return float(train_value) - float(eval_value)

    summary["train_eval_set_recall_at_1_gap"] = _gap("train_set_recall_at_1", "eval_set_recall_at_1")
    summary["train_eval_set_mrr_gap"] = _gap("train_set_mrr", "eval_set_mrr")
    summary["train_eval_fused_set_recall_at_1_gap"] = _gap("train_fused_set_recall_at_1", "eval_fused_set_recall_at_1")
    summary["train_eval_fused_set_mrr_gap"] = _gap("train_fused_set_mrr", "eval_fused_set_mrr")
    summary["train_eval_single_recall_at_1_gap"] = _gap("train_single_recall_at_1", "eval_single_recall_at_1")
    summary["train_eval_single_mrr_gap"] = _gap("train_single_mrr", "eval_single_mrr")
    return summary


PREDICTION_FIELDNAMES = [
    "split",
    "key",
    "gene",
    "context",
    "gold_rank",
    "diag_rank",
    "gold_score",
    "score_source",
    "top1_gene",
    "top1_context",
    "top1_score",
    "top2_gene",
    "top2_context",
    "top2_score",
    "top3_gene",
    "top3_context",
    "top3_score",
    "top4_gene",
    "top4_context",
    "top4_score",
    "top5_gene",
    "top5_context",
    "top5_score",
    "num_text_evidence",
]


EVIDENCE_WEIGHT_FIELDNAMES = [
    "gene",
    "context",
    "text_id",
    "evidence_index",
    "local_index",
    "split",
    "text",
    "weight",
    "prior",
    "conf_gate",
    "gated_prior",
    "conf",
    "content_logit",
    "prior_logit",
    "conf_logit",
    "attn_logit",
    "is_noise",
    "noise_kind",
    "source_gene",
    "source_context",
    "source_text_id",
    "target_gene",
    "target_context",
    "category",
    "conf_filter_keep",
]


def _format_float(value: object) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return ""


def _markdown_cell(value: object, max_chars: int | None = None) -> str:
    text = str(value if value is not None else "")
    text = " ".join(text.replace("\r", " ").replace("\n", " ").split())
    if max_chars is not None and len(text) > max_chars:
        text = text[: max(0, max_chars - 3)].rstrip() + "..."
    return text.replace("\\", "\\\\").replace("|", "\\|")


def write_evidence_weights_by_gene_md(
    path: str | Path,
    rows: list[dict[str, object]],
    max_text_chars: int = 400,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row.get("gene", "")), str(row.get("context", "")), str(row.get("split", "")))
        grouped.setdefault(key, []).append(row)

    fields = [
        "idx",
        "is_noise",
        "noise_kind",
        "source_gene",
        "category",
        "weight",
        "conf",
        "prior",
        "conf_gate",
        "gated_prior",
        "attn_logit",
        "text",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("# Evidence Weights By Gene\n\n")
        for gene, context, split in sorted(grouped):
            group = sorted(
                grouped[(gene, context, split)],
                key=lambda row: (
                    int(str(row.get("is_noise", "0") or "0")),
                    int(str(row.get("local_index", "0") or "0")),
                    int(str(row.get("evidence_index", "0") or "0")),
                ),
            )
            f.write(f"## gene={_markdown_cell(gene)} | context={_markdown_cell(context)} | split={_markdown_cell(split)}\n\n")
            f.write("| " + " | ".join(fields) + " |\n")
            f.write("| " + " | ".join(["---"] * len(fields)) + " |\n")
            for row in group:
                values = [
                    str(row.get("local_index", "")),
                    str(row.get("is_noise", "")),
                    _markdown_cell(row.get("noise_kind", "")),
                    _markdown_cell(row.get("source_gene", "")),
                    _markdown_cell(row.get("category", "")),
                    _format_float(row.get("weight", "")),
                    _format_float(row.get("conf", "")),
                    _format_float(row.get("prior", "")),
                    _format_float(row.get("conf_gate", "")),
                    _format_float(row.get("gated_prior", "")),
                    _format_float(row.get("attn_logit", "")),
                    _markdown_cell(row.get("text", ""), max_chars=max_text_chars),
                ]
                f.write("| " + " | ".join(values) + " |\n")
            f.write("\n")


CELL_WEIGHT_FIELDNAMES = [
    "gene",
    "context",
    "local_index",
    "instance_index",
    "split",
    "weight",
    "prior",
    "conf",
    "content_logit",
    "prior_logit",
    "conf_logit",
    "instance_score_logit",
    "attn_logit",
    "instance_score",
    "rank",
    "expr",
    "is_cell_noise",
    "cell_noise_kind",
    "source_gene",
    "source_context",
    "source_instance_index",
    "target_gene",
    "target_context",
]


def _mean_pairwise_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-8)
    return float(np.mean(np.sum(a * b, axis=-1)))


def _retrieval_metrics_from_scores(scores: np.ndarray, ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    order = np.argsort(-scores, axis=1)
    ranks: list[int] = []
    for i in range(order.shape[0]):
        rank = int(np.where(order[i] == i)[0][0]) + 1
        ranks.append(rank)
    ranks_arr = np.asarray(ranks)
    out = {f"recall_at_{k}": float(np.mean(ranks_arr <= k)) for k in ks}
    out["mrr"] = float(np.mean(1.0 / ranks_arr))
    out["mean_rank"] = float(np.mean(ranks_arr))
    out["median_rank"] = float(np.median(ranks_arr))
    return out


def _retrieval_metrics_by_label_from_scores(
    scores: np.ndarray,
    pred_labels: list[str],
    target_labels: list[str],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    order = np.argsort(-scores, axis=1)
    target_labels_arr = np.asarray(target_labels, dtype=object)
    ranks: list[int] = []
    for i, label in enumerate(pred_labels):
        positives = np.flatnonzero(target_labels_arr == label)
        if positives.size == 0:
            continue
        positive_positions = np.flatnonzero(np.isin(order[i], positives))
        if positive_positions.size == 0:
            continue
        ranks.append(int(positive_positions[0]) + 1)
    if not ranks:
        return {f"recall_at_{k}": 0.0 for k in ks} | {"mrr": 0.0, "mean_rank": 0.0, "median_rank": 0.0}
    ranks_arr = np.asarray(ranks)
    out = {f"recall_at_{k}": float(np.mean(ranks_arr <= k)) for k in ks}
    out["mrr"] = float(np.mean(1.0 / ranks_arr))
    out["mean_rank"] = float(np.mean(ranks_arr))
    out["median_rank"] = float(np.median(ranks_arr))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scGATES Stage 1 dual-set alignment.")
    parser.add_argument("--distribution-npz", default="")
    parser.add_argument("--distribution-store", default="")
    parser.add_argument("--eval-distribution-store", default="",
                        help="If set, load eval distributions from a different store. "
                             "Enables cell-split: train on store_A, eval on store_B with same genes but different cells.")
    parser.add_argument("--text-evidence-npz", default="")
    parser.add_argument("--text-evidence-store", default="")
    parser.add_argument(
        "--eval-text-evidence-npz",
        default="",
        help="Optional held-out eval text evidence NPZ. Defaults to --text-evidence-npz.",
    )
    parser.add_argument(
        "--eval-text-evidence-store",
        default="",
        help="Optional held-out eval text evidence store. Defaults to --text-evidence-store.",
    )
    parser.add_argument(
        "--gene-allowlist",
        default="",
        help="Optional text file with one gene per line. Train/eval distributions and text evidence are restricted to these genes.",
    )
    parser.add_argument(
        "--paired-eval-all-genes",
        action="store_true",
        help=(
            "When --eval-distribution-store is provided, disable the historical gene-level "
            "train/eval split and evaluate the same matched genes with eval donor distributions "
            "and held-out eval text evidence. This matches paired donor/text split experiments."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--align-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cell-encoder", choices=["distribution_mha", "attention_pool"], default="distribution_mha")
    parser.add_argument("--cell-num-heads", type=int, default=4)
    parser.add_argument("--cell-num-layers", type=int, default=2)
    parser.add_argument(
        "--cell-denoise-mode",
        choices=["normal", "none"],
        default="normal",
        help="Cell-side denoising mode. none removes cell prior/conf attention bias while keeping instance-score bias available.",
    )
    parser.add_argument("--max-instances", type=int, default=64)
    parser.add_argument("--max-text-evidence", type=int, default=16)
    parser.add_argument(
        "--text-evidence-sampling",
        choices=["prior_diverse", "top_prior", "random_prior", "first"],
        default="prior_diverse",
    )
    parser.add_argument("--text-prior-temperature", type=float, default=1.0)
    parser.add_argument(
        "--text-conf-beta-init",
        type=float,
        default=0.0,
        help="Initialize TextSetEncoder beta, which scales conf(t) in text attention logits. Default keeps the historical beta=0.",
    )
    parser.add_argument(
        "--text-conf-mode",
        choices=["text_only", "text_prototype", "text_prototype_loo"],
        default="text_only",
        help=(
            "Controls the evidence-level confidence input. text_only preserves the historical "
            "conf(t)=MLP([text; prior metadata]) behavior. text_prototype adds the masked mean "
            "text prototype for the current gene; text_prototype_loo uses a leave-one-out prototype."
        ),
    )
    parser.add_argument(
        "--text-conf-gate-mode",
        choices=["none", "prior_sigmoid"],
        default="none",
        help=(
            "Optional soft gate that uses conf(t) to modify text evidence prior before attention. "
            "none preserves historical behavior; prior_sigmoid uses prior * sigmoid(scale * conf)."
        ),
    )
    parser.add_argument(
        "--text-conf-gate-scale",
        type=float,
        default=1.0,
        help="Scale used by --text-conf-gate-mode prior_sigmoid.",
    )
    parser.add_argument(
        "--text-diversity-weight",
        type=float,
        default=0.7,
        help="Weight for embedding novelty in prior_diverse evidence sampling. 0 means prior-only, 1 means diversity-only.",
    )
    parser.add_argument(
        "--text-noise-ratio",
        type=float,
        default=0.0,
        help="Append this fraction of cross-gene text evidence as marked synthetic noise. Default disables noise.",
    )
    parser.add_argument(
        "--text-noise-max",
        type=int,
        default=0,
        help="Maximum number of synthetic noisy text evidence items appended per gene. 0 means no cap.",
    )
    parser.add_argument(
        "--text-noise-splits",
        choices=["train", "eval", "both"],
        default="eval",
        help="Which text evidence split receives synthetic noise when --text-noise-ratio > 0.",
    )
    parser.add_argument(
        "--text-noise-prior-mode",
        choices=["uniform", "source"],
        default="uniform",
        help="Prior assigned after noise injection. uniform avoids leaking noise identity through prior values.",
    )
    parser.add_argument(
        "--text-noise-kind",
        choices=["cross_gene_text", "easy_random_text"],
        default="cross_gene_text",
        help="Synthetic text noise source. cross_gene_text preserves the historical behavior.",
    )
    parser.add_argument(
        "--text-noise-train-bank",
        default="",
        help="NPZ easy-noise bank used for train split when --text-noise-kind easy_random_text.",
    )
    parser.add_argument(
        "--text-noise-eval-bank",
        default="",
        help="NPZ easy-noise bank used for eval split when --text-noise-kind easy_random_text.",
    )
    parser.add_argument("--text-noise-seed", type=int, default=0)
    parser.add_argument(
        "--eval-text-conf-filter-mode",
        choices=["none", "top_frac", "positive"],
        default="none",
        help="Eval-only diagnostic: recompute Set text embeddings after filtering evidence by per-gene conf(t).",
    )
    parser.add_argument("--eval-text-conf-filter-frac", type=float, default=0.5)
    parser.add_argument("--eval-text-conf-filter-min-keep", type=int, default=4)
    parser.add_argument("--set-fusion", choices=["direct", "residual"], default="direct")
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--align-loss", choices=["multi_positive", "diagonal"], default="multi_positive")
    parser.add_argument(
        "--paper-main",
        action="store_true",
        help="Use the paper-consistent Stage 1 objective: Lsingle + Lset + Lstruct + Ldenoise, and disable experimental losses.",
    )
    parser.add_argument(
        "--context-filter",
        default="",
        help="Optional exact context value. Use to train/evaluate one cell context independently.",
    )
    parser.add_argument("--tau-nce", type=float, default=0.07)
    parser.add_argument("--tau-ce", type=float, default=0.07)
    parser.add_argument("--lambda-single", type=float, default=0.0)
    parser.add_argument("--lambda-set", type=float, default=1.0)
    parser.add_argument("--lambda-text-aware-set", type=float, default=0.0)
    parser.add_argument("--text-aware-instance-alpha", type=float, default=0.0)
    parser.add_argument("--text-aware-temperature", type=float, default=0.07)
    parser.add_argument(
        "--lambda-text-noise-conf",
        type=float,
        default=0.0,
        help="Diagnostic loss weight for supervised clean/noise text confidence margin. Default disables it.",
    )
    parser.add_argument(
        "--text-noise-conf-margin",
        type=float,
        default=0.5,
        help="Margin for clean_conf > noise_conf in the diagnostic text noise confidence loss.",
    )
    parser.add_argument(
        "--fused-set-beta",
        type=float,
        default=0.0,
        help=(
            "Teacher-free Stage3 evaluation score: static set score plus beta times "
            "text-aware set score. Use 0 to disable fused scoring."
        ),
    )
    parser.add_argument(
        "--lambda-fused-set",
        type=float,
        default=0.0,
        help=(
            "Teacher-free Stage3 training loss on the fused set score. This does not "
            "use the single branch as a teacher."
        ),
    )
    parser.add_argument("--fused-set-temperature", type=float, default=0.07)
    parser.add_argument("--lambda-denoise", type=float, default=0.01)
    parser.add_argument(
        "--lambda-cell-denoise",
        type=float,
        default=1.0,
        help="Relative cell-side weight inside denoise_loss. The whole denoise loss is still scaled by --lambda-denoise.",
    )
    parser.add_argument(
        "--lambda-text-denoise",
        type=float,
        default=1.0,
        help="Relative text-side weight inside denoise_loss. The whole denoise loss is still scaled by --lambda-denoise.",
    )
    parser.add_argument("--lambda-prior", type=float, default=1.0)
    parser.add_argument("--lambda-entropy", type=float, default=0.001)
    parser.add_argument("--lambda-struct", type=float, default=0.1)
    parser.add_argument("--lambda-neighbor-struct", type=float, default=0.05)
    parser.add_argument(
        "--lambda-neighbor-contrast",
        type=float,
        default=0.0,
        help="[Experimental] Contrast gold pairs against retrieved neighbor hard negatives. Not part of the paper main objective.",
    )
    parser.add_argument("--neighbor-contrast-mode", choices=["ce", "margin"], default="ce")
    parser.add_argument("--neighbor-contrast-margin", type=float, default=0.05)
    parser.add_argument("--tau-neighbor-contrast", type=float, default=0.07)
    parser.add_argument(
        "--lambda-memory",
        type=float,
        default=0.0,
        help="[Experimental] Extra bidirectional contrastive loss against a detached memory bank of prior batches. Not part of the paper main objective.",
    )
    parser.add_argument("--tau-memory", type=float, default=0.07)
    parser.add_argument("--memory-bank-size", type=int, default=0)
    parser.add_argument(
        "--allow-experimental-negative-mining",
        action="store_true",
        help="Allow experimental neighbor hard-negative / memory-bank losses. Keep off for paper-consistent runs.",
    )
    parser.add_argument("--struct-epsilon", type=float, default=0.08)
    parser.add_argument("--struct-iter", type=int, default=30)
    parser.add_argument("--struct-max-pairs", type=int, default=64)
    parser.add_argument("--struct-huber-delta", type=float, default=0.1)
    parser.add_argument("--struct-mode", choices=["huber", "rank"], default="huber",
                        help="T4: 'huber' (original) or 'rank' (triplet ordering). 'rank' only requires "
                             "order consistency, robust to cross-modal scale differences.")
    parser.add_argument("--struct-rank-margin", type=float, default=0.1,
                        help="T4: margin for triplet ranking in rank mode.")
    parser.add_argument("--tau-struct-cell", type=float, default=0.1)
    parser.add_argument("--tau-struct-text", type=float, default=0.1)
    parser.add_argument("--tau-neighbor-struct", type=float, default=0.1)
    parser.add_argument("--tau-neighbor-prior", type=float, default=1.0)
    parser.add_argument("--neighbor-csv", default="")
    parser.add_argument("--neighbor-output-csv", default="")
    parser.add_argument("--build-neighbors", action="store_true")
    parser.add_argument("--max-neighbors", type=int, default=0)
    parser.add_argument("--neighbor-top-k", type=int, default=8)
    parser.add_argument("--neighbor-ann-k", type=int, default=64)
    parser.add_argument("--neighbor-sinkhorn-epsilon", type=float, default=0.08)
    parser.add_argument("--neighbor-sinkhorn-iter", type=int, default=20)
    parser.add_argument("--neighbor-max-instances", type=int, default=128)
    parser.add_argument("--neighbor-skip-sinkhorn", action="store_true", help="Use cosine-ranked neighbors when building neighbors.")
    parser.add_argument("--allow-cross-context-neighbors", action="store_true")
    parser.add_argument(
        "--neighbor-agg-alpha",
        type=float,
        default=1.0,
        help=(
            "Strength of neighbor aggregation injected into the main cell-set representation. "
            "Use 0.0 to keep neighbor graph only as an auxiliary structural loss."
        ),
    )
    parser.add_argument(
        "--instance-score-npz",
        default="",
        help=(
            "Optional NPZ with per-instance cell-side scores for the training distribution store. "
            "Scores are injected as a cell SetEncoder attention bias."
        ),
    )
    parser.add_argument(
        "--eval-instance-score-npz",
        default="",
        help=(
            "Optional NPZ with per-instance scores for --eval-distribution-store. "
            "Use this when train/eval stores have different instance counts."
        ),
    )
    parser.add_argument(
        "--instance-score-alpha",
        type=float,
        default=0.0,
        help="Strength of per-instance score attention bias. Use 0 to disable.",
    )
    parser.add_argument(
        "--instance-score-mode",
        choices=["signed", "positive"],
        default="signed",
        help=(
            "How per-instance scores are used as attention bias. 'signed' uses the "
            "z-scored value directly. 'positive' uses max(score, 0), rewarding "
            "high-confidence instances without actively suppressing low-score ones."
        ),
    )
    parser.add_argument(
        "--cell-noise-ratio",
        type=float,
        default=0.0,
        help="Append cross-gene cell embeddings as cell-side noise at this ratio per distribution.",
    )
    parser.add_argument(
        "--cell-noise-max",
        type=int,
        default=0,
        help="Maximum number of cell-side noise instances appended per distribution. 0 means no explicit cap.",
    )
    parser.add_argument(
        "--cell-noise-splits",
        choices=["train", "eval", "both"],
        default="both",
        help="Apply cell-side noise to train, eval, or both splits when --cell-noise-ratio > 0.",
    )
    parser.add_argument(
        "--cell-noise-source",
        choices=["same_context", "global"],
        default="same_context",
        help="Source pool for cross-gene cell noise.",
    )
    parser.add_argument(
        "--cell-noise-instance-score-mode",
        choices=["zero", "source", "uniform"],
        default="zero",
        help="Instance score assigned to synthetic cell noise instances.",
    )
    parser.add_argument(
        "--cell-noise-seed",
        type=int,
        default=0,
        help="Random seed for cell-side noise. Defaults to --seed when 0.",
    )
    parser.add_argument("--lambda-cons-cell", type=float, default=0.05)
    parser.add_argument("--lambda-cons-text", type=float, default=0.05)
    parser.add_argument("--lambda-anchor", type=float, default=0.0)
    parser.add_argument(
        "--anchor-no-detach",
        action="store_true",
        help="Let anchor consistency update both single and set branches. Default only pulls set toward single anchors.",
    )
    parser.add_argument("--cell-dropout", type=float, default=0.1)
    parser.add_argument("--text-dropout", type=float, default=0.1)
    parser.add_argument("--lambda-multi-view", type=float, default=0.0,
                        help="Weight for cross-view consistency loss. Forces gene-level stable signal "
                             "by penalising deviation between two differently-sampled views of Dg and Pg. "
                             "Recommended starting value: 0.1-0.5.")
    parser.add_argument("--lambda-neighbor-soft", type=float, default=0.0,
                        help="Weight for neighbor soft-positive CE loss. Neighbour genes receive "
                             "partial positive weight proportional to their cosine similarity, forcing the "
                             "model to learn distributional semantics instead of gene-ID memorisation. "
                             "Recommended starting value: 0.3-1.0.")
    parser.add_argument("--neighbor-soft-weight", type=float, default=0.5,
                        help="Multiplier applied to neighbour cosine_score when building soft targets. "
                             "Default 0.5 means a neighbour with cos=0.8 gets weight 0.4 relative to self=1.0.")
    parser.add_argument(
        "--gene-neighbor-prior-npz",
        default="",
        help="Frozen Stage1 Geneformer neighbor prior NPZ exported by stage1_neighbor_prior.",
    )
    parser.add_argument(
        "--gene-neighbor-prior-negative-npz",
        default="",
        help=(
            "Optional shuffled or otherwise mismatched Stage1 prior NPZ used as an explicit "
            "negative control in the prior-margin loss."
        ),
    )
    parser.add_argument(
        "--lambda-neighbor-prior-cell",
        type=float,
        default=0.0,
        help="Weight for aligning Stage1 cell-set embedding to frozen Stage0 gene-neighbor prior.",
    )
    parser.add_argument(
        "--lambda-neighbor-prior-text",
        type=float,
        default=0.0,
        help="Weight for aligning Stage1 text-set embedding to frozen Stage0 gene-neighbor prior.",
    )
    parser.add_argument(
        "--gene-neighbor-prior-min-count",
        type=int,
        default=1,
        help="Use Stage1 neighbor priors only for genes with counts >= this value.",
    )
    parser.add_argument(
        "--gene-neighbor-prior-count-weighting",
        choices=["none", "log", "sqrt", "linear"],
        default="none",
        help="Weight Stage1 neighbor-prior loss by prior observation counts.",
    )
    parser.add_argument(
        "--lambda-neighbor-prior-margin",
        type=float,
        default=0.0,
        help=(
            "Weight for shuffled-aware Stage3 margin loss. It enforces "
            "cos(cell_set, real_prior) > cos(cell_set, negative_prior) + margin."
        ),
    )
    parser.add_argument(
        "--gene-neighbor-prior-margin",
        type=float,
        default=0.05,
        help="Margin used by --lambda-neighbor-prior-margin.",
    )
    parser.add_argument(
        "--lambda-neighbor-prior-cell-attn",
        type=float,
        default=0.0,
        help=(
            "Weight for Stage3-v2 prior-guided cell attention loss. A projected "
            "Stage1 prior queries cell instances inside Dg, and the resulting "
            "prior-guided cell representation is aligned to Pg."
        ),
    )
    parser.add_argument(
        "--gene-neighbor-prior-cell-attn-fusion-alpha",
        type=float,
        default=0.0,
        help="Residual strength for fusing prior-guided cell representation into the main cell_set.",
    )
    parser.add_argument(
        "--gene-neighbor-prior-cell-attn-temperature",
        type=float,
        default=0.1,
        help="Softmax temperature for prior-guided cell-instance attention.",
    )
    parser.add_argument(
        "--gene-neighbor-prior-cell-attn-gate",
        choices=["none", "agreement"],
        default="none",
        help=(
            "Reliability gate for Stage3-v2 cell-attention fusion. 'agreement' scales "
            "the fusion alpha by the cosine agreement between the original cell_set "
            "and the prior-guided cell representation."
        ),
    )
    parser.add_argument(
        "--gene-neighbor-prior-cell-attn-gate-threshold",
        type=float,
        default=0.0,
        help="Cosine-agreement threshold for --gene-neighbor-prior-cell-attn-gate agreement.",
    )
    parser.add_argument(
        "--gene-neighbor-prior-cell-attn-gate-temperature",
        type=float,
        default=0.1,
        help="Sigmoid temperature for --gene-neighbor-prior-cell-attn-gate agreement.",
    )
    parser.add_argument(
        "--stage3-confusion-csv",
        default="",
        help=(
            "Neighbor-format CSV of Stage2 confusable genes. These are encoded as "
            "negative text evidence sets for Stage3 confusion-aware training and do "
            "not affect the main neighbor aggregator."
        ),
    )
    parser.add_argument("--lambda-stage3-confusion", type=float, default=0.0)
    parser.add_argument("--stage3-confusion-margin", type=float, default=0.05)
    parser.add_argument("--stage3-confusion-max-negatives", type=int, default=0)
    parser.add_argument(
        "--stage3-confusion-mode",
        choices=["margin", "softplus"],
        default="margin",
    )
    parser.add_argument(
        "--stage3-confusion-anchor",
        choices=["prior", "final", "both"],
        default="prior",
        help=(
            "Anchor used by the Stage3 confusion-aware loss. 'prior' preserves the "
            "original behavior, 'final' directly constrains the evaluated cell-set "
            "embedding, and 'both' constrains both representations."
        ),
    )
    parser.add_argument(
        "--gene-neighbor-prior-fusion",
        choices=["none", "cell", "text", "both"],
        default="none",
        help=(
            "Fuse projected Stage1 neighbor prior into the main Stage2 set embedding before "
            "alignment loss/evaluation. This makes the prior part of the primary Dg-Pg "
            "matching objective instead of only an auxiliary regularizer."
        ),
    )
    parser.add_argument(
        "--gene-neighbor-prior-fusion-alpha",
        type=float,
        default=0.0,
        help="Residual strength for --gene-neighbor-prior-fusion. Recommended first values: 0.05 or 0.1.",
    )
    parser.add_argument("--expr-noise-std", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--proj-warmup-epochs", type=int, default=0,
                        help="T1: freeze SetEncoder/TextSetEncoder and train only projection heads for N epochs.")
    parser.add_argument("--lambda-expr-strat", type=float, default=0.0,
                        help="T3: weight for expression-stratified self-supervision. Splits Dg into "
                             "high/low expr halves and penalises MSE between the two encodings. "
                             "Purely cell-side. Recommended starting: 0.05-0.2.")
    parser.add_argument("--expr-strat-detach", action="store_true",
                        help="T3: detach the cell_set encoder during expression stratification, "
                             "so gradients only flow through one branch. Recommended for stabilising early training.")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--eval-train-split",
        action="store_true",
        help="After training, also evaluate the training split and write train_* diagnostic outputs.",
    )
    parser.add_argument(
        "--eval-every-epoch",
        action="store_true",
        help="Evaluate the eval split after every epoch and write epoch_eval_metrics.csv.",
    )
    parser.add_argument(
        "--eval-every-epoch-train",
        action="store_true",
        help="With --eval-every-epoch, also evaluate the train split after every epoch to diagnose overfitting.",
    )
    parser.add_argument("--max-distributions", type=int, default=0)
    parser.add_argument(
        "--sample-distributions",
        action="store_true",
        help="Randomly sample --max-distributions after filtering instead of taking the first sorted rows.",
    )
    parser.add_argument("--distribution-sample-seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true", default=True)
    args = parser.parse_args()

    if bool(args.distribution_npz) == bool(args.distribution_store):
        raise SystemExit("Pass exactly one of --distribution-npz or --distribution-store")
    if bool(args.text_evidence_npz) == bool(args.text_evidence_store):
        raise SystemExit("Pass exactly one of --text-evidence-npz or --text-evidence-store")
    if bool(args.eval_text_evidence_npz) and bool(args.eval_text_evidence_store):
        raise SystemExit("Pass at most one of --eval-text-evidence-npz or --eval-text-evidence-store")
    if args.paper_main:
        if not cli_flag_was_passed("--align-loss"):
            args.align_loss = "diagonal"
        if not cli_flag_was_passed("--lambda-single"):
            args.lambda_single = 0.0
        if not cli_flag_was_passed("--lambda-set"):
            args.lambda_set = 1.0
        if not cli_flag_was_passed("--lambda-neighbor-struct"):
            args.lambda_neighbor_struct = 0.0
        if not cli_flag_was_passed("--lambda-cons-cell"):
            args.lambda_cons_cell = 0.05
        if not cli_flag_was_passed("--lambda-cons-text"):
            args.lambda_cons_text = 0.05
        if not cli_flag_was_passed("--lambda-anchor"):
            args.lambda_anchor = 0.0
        if not cli_flag_was_passed("--proj-warmup-epochs"):
            args.proj_warmup_epochs = 3
        if args.lambda_neighbor_contrast > 0 or args.lambda_memory > 0 or args.memory_bank_size > 0:
            raise SystemExit(
                "--paper-main is incompatible with experimental negative mining. "
                "Set --lambda-neighbor-contrast 0, --lambda-memory 0, and --memory-bank-size 0."
            )
    if args.max_text_evidence < 0:
        raise SystemExit("--max-text-evidence must be >= 0; use 0 to keep all evidence in each batch")
    if args.paired_eval_all_genes and not args.eval_distribution_store:
        raise SystemExit("--paired-eval-all-genes requires --eval-distribution-store")
    if args.text_prior_temperature <= 0:
        raise SystemExit("--text-prior-temperature must be > 0")
    if args.text_conf_beta_init < 0:
        raise SystemExit("--text-conf-beta-init must be >= 0")
    if args.text_conf_gate_scale < 0:
        raise SystemExit("--text-conf-gate-scale must be >= 0")
    if args.lambda_cell_denoise < 0 or args.lambda_text_denoise < 0:
        raise SystemExit("--lambda-cell-denoise and --lambda-text-denoise must be >= 0")
    if not 0.0 <= args.text_diversity_weight <= 1.0:
        raise SystemExit("--text-diversity-weight must be in [0, 1]")
    if args.text_noise_ratio < 0:
        raise SystemExit("--text-noise-ratio must be >= 0")
    if args.text_noise_max < 0:
        raise SystemExit("--text-noise-max must be >= 0")
    if not 0.0 < args.eval_text_conf_filter_frac <= 1.0:
        raise SystemExit("--eval-text-conf-filter-frac must be in (0, 1]")
    if args.eval_text_conf_filter_min_keep < 1:
        raise SystemExit("--eval-text-conf-filter-min-keep must be >= 1")
    if args.lambda_text_noise_conf < 0:
        raise SystemExit("--lambda-text-noise-conf must be >= 0")
    if args.text_noise_conf_margin < 0:
        raise SystemExit("--text-noise-conf-margin must be >= 0")
    if args.text_noise_ratio > 0 and args.text_noise_kind == "easy_random_text":
        if args.text_noise_splits in {"train", "both"} and not args.text_noise_train_bank:
            raise SystemExit("--text-noise-kind easy_random_text requires --text-noise-train-bank for train/both splits")
        if args.text_noise_splits in {"eval", "both"} and not args.text_noise_eval_bank:
            raise SystemExit("--text-noise-kind easy_random_text requires --text-noise-eval-bank for eval/both splits")
    if args.lambda_text_noise_conf > 0:
        if args.text_noise_ratio <= 0:
            raise SystemExit("--lambda-text-noise-conf requires --text-noise-ratio > 0")
        if args.text_noise_splits == "eval":
            raise SystemExit("--lambda-text-noise-conf requires --text-noise-splits train or both")
    if args.tau_neighbor_prior <= 0:
        raise SystemExit("--tau-neighbor-prior must be > 0")
    if args.text_aware_temperature <= 0:
        raise SystemExit("--text-aware-temperature must be > 0")
    if args.fused_set_temperature <= 0:
        raise SystemExit("--fused-set-temperature must be > 0")
    if args.fused_set_beta < 0 or args.lambda_fused_set < 0:
        raise SystemExit("--fused-set-beta and --lambda-fused-set must be >= 0")
    if args.lambda_fused_set > 0 and args.fused_set_beta <= 0:
        raise SystemExit("Set --fused-set-beta > 0 when enabling --lambda-fused-set")
    if args.tau_neighbor_contrast <= 0:
        raise SystemExit("--tau-neighbor-contrast must be > 0")
    if args.neighbor_contrast_margin < 0:
        raise SystemExit("--neighbor-contrast-margin must be >= 0")
    if args.tau_memory <= 0:
        raise SystemExit("--tau-memory must be > 0")
    if args.lambda_memory > 0 and args.memory_bank_size <= 0:
        raise SystemExit("--memory-bank-size must be > 0 when --lambda-memory > 0")
    if args.memory_bank_size < 0:
        raise SystemExit("--memory-bank-size must be >= 0")
    if args.lambda_neighbor_prior_cell < 0 or args.lambda_neighbor_prior_text < 0:
        raise SystemExit("--lambda-neighbor-prior-cell/text must be >= 0")
    if args.lambda_neighbor_prior_margin < 0:
        raise SystemExit("--lambda-neighbor-prior-margin must be >= 0")
    if args.gene_neighbor_prior_margin < 0:
        raise SystemExit("--gene-neighbor-prior-margin must be >= 0")
    if args.lambda_neighbor_prior_cell_attn < 0:
        raise SystemExit("--lambda-neighbor-prior-cell-attn must be >= 0")
    if args.gene_neighbor_prior_cell_attn_fusion_alpha < 0:
        raise SystemExit("--gene-neighbor-prior-cell-attn-fusion-alpha must be >= 0")
    if args.gene_neighbor_prior_cell_attn_temperature <= 0:
        raise SystemExit("--gene-neighbor-prior-cell-attn-temperature must be > 0")
    if args.gene_neighbor_prior_cell_attn_gate_temperature <= 0:
        raise SystemExit("--gene-neighbor-prior-cell-attn-gate-temperature must be > 0")
    if args.lambda_stage3_confusion < 0:
        raise SystemExit("--lambda-stage3-confusion must be >= 0")
    if args.stage3_confusion_margin < 0:
        raise SystemExit("--stage3-confusion-margin must be >= 0")
    if args.stage3_confusion_max_negatives < 0:
        raise SystemExit("--stage3-confusion-max-negatives must be >= 0")
    if args.lambda_stage3_confusion > 0 and (not args.stage3_confusion_csv or args.stage3_confusion_max_negatives <= 0):
        raise SystemExit("Pass --stage3-confusion-csv and --stage3-confusion-max-negatives > 0 when enabling --lambda-stage3-confusion")
    if args.gene_neighbor_prior_fusion_alpha < 0:
        raise SystemExit("--gene-neighbor-prior-fusion-alpha must be >= 0")
    if args.gene_neighbor_prior_min_count < 1:
        raise SystemExit("--gene-neighbor-prior-min-count must be >= 1")
    if (
        args.lambda_neighbor_prior_cell > 0
        or args.lambda_neighbor_prior_text > 0
        or args.lambda_neighbor_prior_margin > 0
        or args.lambda_neighbor_prior_cell_attn > 0
        or args.gene_neighbor_prior_cell_attn_fusion_alpha > 0
        or (args.lambda_stage3_confusion > 0 and args.stage3_confusion_anchor != "final")
        or (args.gene_neighbor_prior_fusion != "none" and args.gene_neighbor_prior_fusion_alpha > 0)
    ) and not args.gene_neighbor_prior_npz:
        raise SystemExit("Pass --gene-neighbor-prior-npz when enabling Stage1 neighbor prior losses or fusion")
    if args.lambda_neighbor_prior_margin > 0 and not args.gene_neighbor_prior_negative_npz:
        raise SystemExit("Pass --gene-neighbor-prior-negative-npz when enabling --lambda-neighbor-prior-margin")
    if (args.lambda_neighbor_contrast > 0 or args.lambda_memory > 0) and not args.allow_experimental_negative_mining:
        raise SystemExit(
            "neighbor hard-negative / memory-bank losses are experimental and not paper-consistent. "
            "Pass --allow-experimental-negative-mining only for exploratory debugging."
        )
    if args.residual_scale < 0:
        raise SystemExit("--residual-scale must be >= 0")
    if args.cell_encoder == "distribution_mha" and args.hidden_dim % args.cell_num_heads != 0:
        raise SystemExit("--hidden-dim must be divisible by --cell-num-heads for distribution_mha")
    if args.max_neighbors < 0:
        raise SystemExit("--max-neighbors must be >= 0")
    if not 0.0 <= args.neighbor_agg_alpha <= 1.0:
        raise SystemExit("--neighbor-agg-alpha must be in [0, 1]")
    if args.instance_score_alpha < 0:
        raise SystemExit("--instance-score-alpha must be >= 0")
    if args.instance_score_alpha > 0 and not args.instance_score_npz:
        raise SystemExit("Pass --instance-score-npz when --instance-score-alpha > 0")
    if args.cell_noise_ratio < 0:
        raise SystemExit("--cell-noise-ratio must be >= 0")
    if args.cell_noise_max < 0:
        raise SystemExit("--cell-noise-max must be >= 0")
    if args.neighbor_ann_k < args.neighbor_top_k:
        raise SystemExit("--neighbor-ann-k must be >= --neighbor-top-k")
    if args.lambda_neighbor_struct > 0 and args.max_neighbors <= 0:
        raise SystemExit("Set --max-neighbors > 0 when enabling --lambda-neighbor-struct")
    if args.max_neighbors > 0 and not args.neighbor_csv and not args.build_neighbors:
        raise SystemExit("Pass --neighbor-csv or --build-neighbors when --max-neighbors > 0")

    start_time = time.time()
    set_seed(args.seed)
    out_dir = Path(args.output_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    log_step(f"run_id={args.run_id}")
    eval_text_evidence_store = args.eval_text_evidence_store or args.text_evidence_store
    eval_text_evidence_npz = args.eval_text_evidence_npz or args.text_evidence_npz

    step_time = log_step("loading train text evidence gene index")
    if args.text_evidence_store:
        evidence_genes = load_text_evidence_genes_store(args.text_evidence_store)
    else:
        evidence_genes = load_text_evidence_genes_npz(args.text_evidence_npz)
    step_time = log_step(f"loaded train text evidence gene index: {len(evidence_genes)} genes", step_time)

    eval_evidence_genes = evidence_genes
    if args.eval_text_evidence_store or args.eval_text_evidence_npz:
        step_time = log_step("loading eval text evidence gene index")
        if eval_text_evidence_store:
            eval_evidence_genes = load_text_evidence_genes_store(eval_text_evidence_store)
        else:
            eval_evidence_genes = load_text_evidence_genes_npz(eval_text_evidence_npz)
        step_time = log_step(f"loaded eval text evidence gene index: {len(eval_evidence_genes)} genes", step_time)

    requested_allowlist_genes: set[str] | None = None
    gene_allowlist_summary: dict[str, object] = {
        "enabled": False,
        "path": "",
        "requested_allowlist_genes": 0,
        "matched_allowlist_genes": 0,
        "dropped_allowlist_genes": 0,
        "dropped_allowlist_gene_names": [],
    }
    if args.gene_allowlist:
        step_time = log_step(f"loading gene allowlist: {args.gene_allowlist}")
        requested_allowlist_genes = read_gene_allowlist(args.gene_allowlist)
        available_text_genes = evidence_genes & eval_evidence_genes
        matched_allowlist_genes = requested_allowlist_genes & available_text_genes
        dropped_allowlist_genes = sorted(requested_allowlist_genes - matched_allowlist_genes)
        if not matched_allowlist_genes:
            raise SystemExit(f"No allowlisted genes are present in both train/eval text evidence: {args.gene_allowlist}")
        evidence_genes = evidence_genes & matched_allowlist_genes
        eval_evidence_genes = eval_evidence_genes & matched_allowlist_genes
        gene_allowlist_summary = {
            "enabled": True,
            "path": args.gene_allowlist,
            "requested_allowlist_genes": len(requested_allowlist_genes),
            "matched_allowlist_genes": len(matched_allowlist_genes),
            "dropped_allowlist_genes": len(dropped_allowlist_genes),
            "dropped_allowlist_gene_names": dropped_allowlist_genes,
        }
        step_time = log_step(
            f"applied gene allowlist: requested={len(requested_allowlist_genes)} "
            f"matched={len(matched_allowlist_genes)} dropped={len(dropped_allowlist_genes)}",
            step_time,
        )

    step_time = log_step("loading cell distributions")
    if args.distribution_store:
        distributions = load_distributions_store(
            args.distribution_store,
            gene_allowlist=evidence_genes,
            context_filter=args.context_filter,
            max_distributions=0 if args.sample_distributions else args.max_distributions,
        )
    else:
        distributions = load_distributions_npz(args.distribution_npz)
        distributions = [d for d in distributions if d.gene in evidence_genes]
        if args.context_filter:
            distributions = [d for d in distributions if d.context == args.context_filter]
        if args.max_distributions and not args.sample_distributions:
            distributions = distributions[: args.max_distributions]
    if args.context_filter and not distributions:
        raise SystemExit(f"No distributions matched --context-filter: {args.context_filter}")
    distributions = limit_distributions(
        distributions,
        max_distributions=args.max_distributions,
        sample=args.sample_distributions,
        seed=args.distribution_sample_seed,
    )
    step_time = log_step(
        f"loaded cell distributions: {len(distributions)} distributions, {len({d.gene for d in distributions})} genes",
        step_time,
    )
    instance_score_summary: dict[str, object] = {}
    eval_instance_score_summary: dict[str, object] = {}
    if args.instance_score_npz:
        step_time = log_step(f"loading train instance scores: {args.instance_score_npz}")
        train_score_map = load_instance_score_npz(args.instance_score_npz)
        distributions, instance_score_summary = attach_instance_scores(distributions, train_score_map, strict=False)
        step_time = log_step(
            f"attached train instance scores: coverage={float(instance_score_summary['coverage']):.3f}",
            step_time,
        )
    write_csv(
        out_dir / "selected_distributions.csv",
        [
            {
                "key": d.key,
                "gene": d.gene,
                "context": d.context,
                "num_instances": int(d.embeddings.shape[0]),
                "occurrence_rate": float(d.occurrence_rate),
            }
            for d in distributions
        ],
        ["key", "gene", "context", "num_instances", "occurrence_rate"],
    )

    neighbor_map: dict[str, list[Neighbor]] | None = None
    if args.max_neighbors > 0:
        if args.neighbor_csv:
            step_time = log_step(f"loading neighbor csv: {args.neighbor_csv}")
            neighbor_map = load_neighbor_csv(args.neighbor_csv)
            step_time = log_step(f"loaded neighbor csv: {sum(len(v) for v in neighbor_map.values())} rows", step_time)
        elif args.build_neighbors:
            step_time = log_step("building neighbors")
            neighbor_map = retrieve_neighbors(
                distributions,
                top_k=args.neighbor_top_k,
                ann_k=args.neighbor_ann_k,
                sinkhorn_epsilon=args.neighbor_sinkhorn_epsilon,
                sinkhorn_iter=args.neighbor_sinkhorn_iter,
                same_context_only=not args.allow_cross_context_neighbors,
                max_instances=args.neighbor_max_instances,
                skip_sinkhorn=args.neighbor_skip_sinkhorn,
            )
            step_time = log_step(f"built neighbors: {sum(len(v) for v in neighbor_map.values())} rows", step_time)
            neighbor_output_csv = args.neighbor_output_csv or str(out_dir / "neighbors.csv")
            step_time = log_step(f"writing neighbor csv: {neighbor_output_csv}")
            write_neighbor_csv(neighbor_output_csv, neighbor_map)
            step_time = log_step("wrote neighbor csv", step_time)

    confusion_map: dict[str, list[Neighbor]] | None = None
    if args.stage3_confusion_csv:
        step_time = log_step(f"loading Stage3 confusion csv: {args.stage3_confusion_csv}")
        confusion_map = load_neighbor_csv(args.stage3_confusion_csv)
        step_time = log_step(f"loaded Stage3 confusion csv: {sum(len(v) for v in confusion_map.values())} rows", step_time)

    selected_genes = {d.gene for d in distributions}
    if requested_allowlist_genes is not None:
        final_matched_allowlist_genes = requested_allowlist_genes & selected_genes & eval_evidence_genes
        final_dropped_allowlist_genes = sorted(requested_allowlist_genes - final_matched_allowlist_genes)
        gene_allowlist_summary = {
            **gene_allowlist_summary,
            "matched_allowlist_genes": len(final_matched_allowlist_genes),
            "dropped_allowlist_genes": len(final_dropped_allowlist_genes),
            "dropped_allowlist_gene_names": final_dropped_allowlist_genes,
        }
        log_step(
            f"final gene allowlist match after distribution/context filtering: "
            f"matched={len(final_matched_allowlist_genes)} dropped={len(final_dropped_allowlist_genes)}"
        )
    step_time = log_step(f"loading selected train text evidence sets: {len(selected_genes)} genes")
    if args.text_evidence_store:
        evidence_sets = load_text_evidence_sets_store(args.text_evidence_store, gene_allowlist=selected_genes)
    else:
        evidence_sets = load_text_evidence_sets_npz(args.text_evidence_npz, gene_allowlist=selected_genes)
    step_time = log_step(f"loaded selected train text evidence sets: {len(evidence_sets)} sets", step_time)

    eval_evidence_sets = evidence_sets
    if args.eval_text_evidence_store or args.eval_text_evidence_npz:
        step_time = log_step(f"loading selected eval text evidence sets: {len(selected_genes)} genes")
        if eval_text_evidence_store:
            eval_evidence_sets = load_text_evidence_sets_store(eval_text_evidence_store, gene_allowlist=selected_genes)
        else:
            eval_evidence_sets = load_text_evidence_sets_npz(eval_text_evidence_npz, gene_allowlist=selected_genes)
        step_time = log_step(f"loaded selected eval text evidence sets: {len(eval_evidence_sets)} sets", step_time)

    train_text_noise_summary: dict[str, object] = {
        "enabled": False,
        "kind": args.text_noise_kind,
        "sets": len(evidence_sets),
        "clean_evidence": sum(len(e.text_ids) for e in evidence_sets),
        "noise_evidence": 0,
        "noise_fraction": 0.0,
    }
    eval_text_noise_summary: dict[str, object] = {
        "enabled": False,
        "kind": args.text_noise_kind,
        "sets": len(eval_evidence_sets),
        "clean_evidence": sum(len(e.text_ids) for e in eval_evidence_sets),
        "noise_evidence": 0,
        "noise_fraction": 0.0,
    }
    if args.text_noise_ratio > 0:
        base_noise_seed = int(args.text_noise_seed or args.seed)
        train_easy_noise_bank = None
        eval_easy_noise_bank = None
        if args.text_noise_kind == "easy_random_text":
            if args.text_noise_splits in {"train", "both"}:
                step_time = log_step(f"loading train easy text noise bank: {args.text_noise_train_bank}")
                train_easy_noise_bank = load_easy_noise_bank(args.text_noise_train_bank)
                step_time = log_step(
                    f"loaded train easy text noise bank: {len(train_easy_noise_bank.noise_ids)} texts",
                    step_time,
                )
            if args.text_noise_splits in {"eval", "both"}:
                step_time = log_step(f"loading eval easy text noise bank: {args.text_noise_eval_bank}")
                eval_easy_noise_bank = load_easy_noise_bank(args.text_noise_eval_bank)
                step_time = log_step(
                    f"loaded eval easy text noise bank: {len(eval_easy_noise_bank.noise_ids)} texts",
                    step_time,
                )
        if args.text_noise_splits in {"train", "both"}:
            step_time = log_step(
                f"injecting train text noise: kind={args.text_noise_kind}, ratio={args.text_noise_ratio}, "
                f"max={args.text_noise_max}, prior={args.text_noise_prior_mode}"
            )
            if args.text_noise_kind == "easy_random_text":
                if train_easy_noise_bank is None:
                    raise RuntimeError("train easy noise bank was not loaded")
                evidence_sets, train_text_noise_summary = inject_easy_text_evidence_noise(
                    evidence_sets,
                    bank=train_easy_noise_bank,
                    ratio=args.text_noise_ratio,
                    max_noise=args.text_noise_max,
                    seed=base_noise_seed,
                    prior_mode=args.text_noise_prior_mode,
                )
            else:
                evidence_sets, train_text_noise_summary = inject_text_evidence_noise(
                    evidence_sets,
                    ratio=args.text_noise_ratio,
                    max_noise=args.text_noise_max,
                    seed=base_noise_seed,
                    prior_mode=args.text_noise_prior_mode,
                )
            step_time = log_step(
                f"injected train text noise: noise={train_text_noise_summary['noise_evidence']} "
                f"fraction={float(train_text_noise_summary['noise_fraction']):.3f}",
                step_time,
            )
        if args.text_noise_splits in {"eval", "both"}:
            step_time = log_step(
                f"injecting eval text noise: kind={args.text_noise_kind}, ratio={args.text_noise_ratio}, "
                f"max={args.text_noise_max}, prior={args.text_noise_prior_mode}"
            )
            if args.text_noise_kind == "easy_random_text":
                if eval_easy_noise_bank is None:
                    raise RuntimeError("eval easy noise bank was not loaded")
                eval_evidence_sets, eval_text_noise_summary = inject_easy_text_evidence_noise(
                    eval_evidence_sets,
                    bank=eval_easy_noise_bank,
                    ratio=args.text_noise_ratio,
                    max_noise=args.text_noise_max,
                    seed=base_noise_seed + 100_003,
                    prior_mode=args.text_noise_prior_mode,
                )
            else:
                eval_evidence_sets, eval_text_noise_summary = inject_text_evidence_noise(
                    eval_evidence_sets,
                    ratio=args.text_noise_ratio,
                    max_noise=args.text_noise_max,
                    seed=base_noise_seed + 100_003,
                    prior_mode=args.text_noise_prior_mode,
                )
            step_time = log_step(
                f"injected eval text noise: noise={eval_text_noise_summary['noise_evidence']} "
                f"fraction={float(eval_text_noise_summary['noise_fraction']):.3f}",
                step_time,
            )

    step_time = log_step("building matched examples")
    examples = build_examples(distributions, evidence_sets, neighbor_map=neighbor_map, confusion_map=confusion_map)
    if args.eval_distribution_store and args.paired_eval_all_genes:
        train_examples = examples
        eval_examples: list[Stage1Example] = []
        step_time = log_step(
            f"built paired train examples without gene split: train={len(train_examples)}, matched={len(examples)}",
            step_time,
        )
    else:
        train_examples, eval_examples = split_examples(examples, args.seed)
        if (args.eval_text_evidence_store or args.eval_text_evidence_npz) and not args.eval_distribution_store:
            eval_dists_for_text_split = [example.dist for example in eval_examples]
            eval_examples = build_examples(
                eval_dists_for_text_split,
                eval_evidence_sets,
                neighbor_map=neighbor_map,
                confusion_map=confusion_map,
            )
        step_time = log_step(
            f"built examples: train={len(train_examples)}, eval={len(eval_examples)}, matched={len(examples)}",
            step_time,
        )

    # Cell-split / context-split: reload eval from a separate store or context
    if args.eval_distribution_store:
        train_genes = {e.dist.gene for e in train_examples}
        eval_dists = load_distributions_store(
            args.eval_distribution_store,
            gene_allowlist=train_genes & eval_evidence_genes,
            context_filter=args.context_filter,
            max_distributions=0,
        )
        if args.eval_instance_score_npz:
            step_time = log_step(f"loading eval instance scores: {args.eval_instance_score_npz}")
            eval_score_map = load_instance_score_npz(args.eval_instance_score_npz)
            eval_dists, eval_instance_score_summary = attach_instance_scores(eval_dists, eval_score_map, strict=False)
            step_time = log_step(
                f"attached eval instance scores: coverage={float(eval_instance_score_summary['coverage']):.3f}",
                step_time,
            )
        if eval_dists:
            eval_examples = build_examples(eval_dists, eval_evidence_sets, neighbor_map=neighbor_map, confusion_map=confusion_map)
        step_time = log_step(
            f"cell/context split: eval examples={len(eval_examples)}",
            step_time,
        )

    train_cell_noise_summary: dict[str, object] = {
        "enabled": False,
        "sets": len(train_examples),
        "clean_instances": sum(int(example.dist.embeddings.shape[0]) for example in train_examples),
        "noise_instances": 0,
        "noise_fraction": 0.0,
        "ratio": float(args.cell_noise_ratio),
        "max_noise": int(args.cell_noise_max),
        "source": args.cell_noise_source,
        "instance_score_mode": args.cell_noise_instance_score_mode,
        "seed": int(args.cell_noise_seed or args.seed),
    }
    eval_cell_noise_summary: dict[str, object] = {
        "enabled": False,
        "sets": len(eval_examples),
        "clean_instances": sum(int(example.dist.embeddings.shape[0]) for example in eval_examples),
        "noise_instances": 0,
        "noise_fraction": 0.0,
        "ratio": float(args.cell_noise_ratio),
        "max_noise": int(args.cell_noise_max),
        "source": args.cell_noise_source,
        "instance_score_mode": args.cell_noise_instance_score_mode,
        "seed": int(args.cell_noise_seed or args.seed),
    }
    train_cell_noise_metadata: CellNoiseMetadata = {}
    eval_cell_noise_metadata: CellNoiseMetadata = {}
    if args.cell_noise_ratio > 0:
        base_cell_noise_seed = int(args.cell_noise_seed or args.seed)
        if args.cell_noise_splits in {"train", "both"}:
            step_time = log_step(
                f"injecting train cell noise: ratio={args.cell_noise_ratio}, max={args.cell_noise_max}, "
                f"source={args.cell_noise_source}, instance_score_mode={args.cell_noise_instance_score_mode}"
            )
            train_examples, train_cell_noise_summary, train_cell_noise_metadata = inject_cell_distribution_noise(
                train_examples,
                ratio=args.cell_noise_ratio,
                max_noise=args.cell_noise_max,
                seed=base_cell_noise_seed,
                source_mode=args.cell_noise_source,
                instance_score_mode=args.cell_noise_instance_score_mode,
            )
            step_time = log_step(
                f"injected train cell noise: noise={train_cell_noise_summary['noise_instances']} "
                f"fraction={float(train_cell_noise_summary['noise_fraction']):.3f}",
                step_time,
            )
        if args.cell_noise_splits in {"eval", "both"}:
            step_time = log_step(
                f"injecting eval cell noise: ratio={args.cell_noise_ratio}, max={args.cell_noise_max}, "
                f"source={args.cell_noise_source}, instance_score_mode={args.cell_noise_instance_score_mode}"
            )
            eval_examples, eval_cell_noise_summary, eval_cell_noise_metadata = inject_cell_distribution_noise(
                eval_examples,
                ratio=args.cell_noise_ratio,
                max_noise=args.cell_noise_max,
                seed=base_cell_noise_seed + 100_003,
                source_mode=args.cell_noise_source,
                instance_score_mode=args.cell_noise_instance_score_mode,
            )
            step_time = log_step(
                f"injected eval cell noise: noise={eval_cell_noise_summary['noise_instances']} "
                f"fraction={float(eval_cell_noise_summary['noise_fraction']):.3f}",
                step_time,
            )
    if requested_allowlist_genes is not None:
        final_train_genes = {example.dist.gene for example in train_examples}
        final_eval_genes = {example.dist.gene for example in eval_examples}
        final_matched_allowlist_genes = requested_allowlist_genes & final_train_genes & final_eval_genes
        final_dropped_allowlist_genes = sorted(requested_allowlist_genes - final_matched_allowlist_genes)
        gene_allowlist_summary = {
            **gene_allowlist_summary,
            "matched_allowlist_genes": len(final_matched_allowlist_genes),
            "dropped_allowlist_genes": len(final_dropped_allowlist_genes),
            "dropped_allowlist_gene_names": final_dropped_allowlist_genes,
        }
        log_step(
            f"final gene allowlist match after train/eval example construction: "
            f"matched={len(final_matched_allowlist_genes)} dropped={len(final_dropped_allowlist_genes)}"
        )
    text_metadata_dim = 0
    for evidence in [*evidence_sets, *eval_evidence_sets]:
        if evidence.metadata_features is not None:
            text_metadata_dim = int(evidence.metadata_features.shape[1])
            break
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    log_step(f"using device={device}")
    prior_store: GeneNeighborPriorStore | None = None
    negative_prior_store: GeneNeighborPriorStore | None = None
    prior_proj: nn.Module | None = None
    prior_cell_query_proj: nn.Module | None = None
    prior_cell_adapter: nn.Module | None = None
    if args.gene_neighbor_prior_npz:
        step_time = log_step(f"loading Stage0 gene neighbor prior: {args.gene_neighbor_prior_npz}")
        prior_store = load_gene_neighbor_prior_npz(args.gene_neighbor_prior_npz)
        step_time = log_step(
            f"loaded Stage0 prior: {len(prior_store.genes)} genes, dim={prior_store.dim}",
            step_time,
        )
    if args.gene_neighbor_prior_negative_npz:
        step_time = log_step(f"loading Stage0 negative gene neighbor prior: {args.gene_neighbor_prior_negative_npz}")
        negative_prior_store = load_gene_neighbor_prior_npz(args.gene_neighbor_prior_negative_npz)
        if prior_store is not None and negative_prior_store.dim != prior_store.dim:
            raise SystemExit(
                f"Negative prior dim mismatch: {negative_prior_store.dim} != {prior_store.dim}"
            )
        step_time = log_step(
            f"loaded Stage0 negative prior: {len(negative_prior_store.genes)} genes, dim={negative_prior_store.dim}",
            step_time,
        )
    step_time = log_step("initializing model")
    model = Stage1AlignmentModel(
        cell_dim=examples[0].dist.embeddings.shape[1],
        text_dim=examples[0].evidence.embeddings.shape[1],
        hidden_dim=args.hidden_dim,
        align_dim=args.align_dim,
        dropout=args.dropout,
        text_metadata_dim=text_metadata_dim,
        cell_encoder=args.cell_encoder,
        cell_num_heads=args.cell_num_heads,
        cell_num_layers=args.cell_num_layers,
        cell_denoise_mode=args.cell_denoise_mode,
        text_conf_beta_init=args.text_conf_beta_init,
        text_conf_mode=args.text_conf_mode,
        text_conf_gate_mode=args.text_conf_gate_mode,
        text_conf_gate_scale=args.text_conf_gate_scale,
    ).to(device)
    if prior_store is not None:
        prior_proj = nn.Sequential(
            nn.LayerNorm(prior_store.dim),
            nn.Linear(prior_store.dim, args.align_dim),
        ).to(device)
        if args.lambda_neighbor_prior_cell_attn > 0 or args.gene_neighbor_prior_cell_attn_fusion_alpha > 0:
            cell_dim = int(examples[0].dist.embeddings.shape[1])
            prior_cell_query_proj = nn.Sequential(
                nn.LayerNorm(prior_store.dim),
                nn.Linear(prior_store.dim, cell_dim),
            ).to(device)
            prior_cell_adapter = nn.Sequential(
                nn.LayerNorm(cell_dim),
                nn.Linear(cell_dim, args.align_dim),
            ).to(device)
    trainable_params = (
        list(model.parameters())
        + (list(prior_proj.parameters()) if prior_proj is not None else [])
        + (list(prior_cell_query_proj.parameters()) if prior_cell_query_proj is not None else [])
        + (list(prior_cell_adapter.parameters()) if prior_cell_adapter is not None else [])
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    log_step("initialized model", step_time)

    # Projection-head warmup: freeze encoders, train only projection layers.
    warmup_encoders = (
        [model.cell_set, model.cell_single, model.text_set, model.text_single, model.neighbor_agg]
        if args.proj_warmup_epochs > 0
        else []
    )
    if warmup_encoders:
        for m in warmup_encoders:
            for p in m.parameters():
                p.requires_grad = False
        proj_params = [p for n, p in model.named_parameters() if "proj" in n and p.requires_grad]
        if _uses_text_aware_set(args):
            proj_params.extend([p for p in model.text_aware_set_scorer.parameters() if p.requires_grad])
        if prior_proj is not None:
            proj_params.extend([p for p in prior_proj.parameters() if p.requires_grad])
        if prior_cell_query_proj is not None:
            proj_params.extend([p for p in prior_cell_query_proj.parameters() if p.requires_grad])
        if prior_cell_adapter is not None:
            proj_params.extend([p for p in prior_cell_adapter.parameters() if p.requires_grad])
        warmup_optimizer = torch.optim.AdamW(proj_params, lr=args.lr * 2.0, weight_decay=args.weight_decay)
        log_step(f"T1: freezing encoders for {args.proj_warmup_epochs} warmup epochs, lr={args.lr * 2.0:.1e}")

    last_parts: dict[str, float] = {}
    epoch_rows: list[dict[str, float | int]] = []
    epoch_eval_rows: list[dict[str, float | int]] = []
    epoch_eval_summary_rows: list[dict[str, float | int | str]] = []
    memory_bank = ContrastiveMemoryBank(args.memory_bank_size) if args.memory_bank_size > 0 else None
    for epoch in range(args.epochs):
        epoch_time = log_step(f"starting epoch {epoch + 1}/{args.epochs}")
        # T1: unfreeze encoders after warmup
        if warmup_encoders and epoch == args.proj_warmup_epochs:
            for m in warmup_encoders:
                for p in m.parameters():
                    p.requires_grad = True
            log_step("T1: unfreezing encoders, switching to full optimizer")
            warmup_encoders = []  # prevent re-entry
        use_opt = warmup_optimizer if warmup_encoders else optimizer
        random.shuffle(train_examples)
        model.train()
        epoch_parts: list[dict[str, float]] = []
        for start in range(0, len(train_examples), args.batch_size):
            batch = train_examples[start : start + args.batch_size]
            if len(batch) < 2:
                continue
            out = batch_forward(
                model,
                batch,
                args.max_instances,
                args.max_text_evidence,
                device,
                args.text_evidence_sampling,
                args.max_neighbors,
                args.stage3_confusion_max_negatives,
                args.text_prior_temperature,
                args.text_diversity_weight,
                args.set_fusion,
                args.residual_scale,
                args.neighbor_agg_alpha,
                args.instance_score_alpha,
                args.instance_score_mode,
            )
            fusion_parts = apply_gene_neighbor_prior_fusion(
                batch,
                out,
                prior_store=prior_store,
                prior_proj=prior_proj,
                args=args,
            )
            cell_attn_parts = apply_gene_neighbor_prior_cell_attention(
                batch,
                out,
                prior_store=prior_store,
                prior_cell_query_proj=prior_cell_query_proj,
                prior_cell_adapter=prior_cell_adapter,
                args=args,
            )
            out_b = None
            if args.lambda_multi_view > 0:
                out_b = batch_forward(
                    model,
                    batch,
                    args.max_instances,
                    args.max_text_evidence,
                    device,
                    args.text_evidence_sampling,
                    args.max_neighbors,
                    args.stage3_confusion_max_negatives,
                    args.text_prior_temperature,
                    args.text_diversity_weight,
                    args.set_fusion,
                    args.residual_scale,
                    args.neighbor_agg_alpha,
                    args.instance_score_alpha,
                    args.instance_score_mode,
                    sampling_seed=-(epoch * 1000 + start + 1),  # Negative seed enables random mode for cell stratification and text evidence
                )
            if out_b is not None:
                apply_gene_neighbor_prior_fusion(
                    batch,
                    out_b,
                    prior_store=prior_store,
                    prior_proj=prior_proj,
                    args=args,
                )
                apply_gene_neighbor_prior_cell_attention(
                    batch,
                    out_b,
                    prior_store=prior_store,
                    prior_cell_query_proj=prior_cell_query_proj,
                    prior_cell_adapter=prior_cell_adapter,
                    args=args,
                )
            loss, parts = compute_losses(
                model,
                batch,
                out,
                args,
                epoch,
                memory_bank=memory_bank,
                out_b=out_b,
                prior_store=prior_store,
                negative_prior_store=negative_prior_store,
                prior_proj=prior_proj,
            )
            parts.update({k: float(v.detach().cpu()) for k, v in fusion_parts.items()})
            parts.update({k: float(v.detach().cpu()) for k, v in cell_attn_parts.items()})
            use_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            use_opt.step()
            if memory_bank is not None:
                memory_bank.enqueue(
                    out["cell_set"],  # type: ignore[arg-type]
                    out["text_set"],  # type: ignore[arg-type]
                    [example.dist.gene for example in batch],
                )
            epoch_parts.append(parts)
        if epoch_parts:
            last_parts = {key: float(np.mean([p[key] for p in epoch_parts])) for key in epoch_parts[0]}
        elapsed_epoch = time.time() - epoch_time
        if last_parts:
            epoch_rows.append({"epoch": epoch + 1, "epoch_elapsed_sec": elapsed_epoch, **last_parts})
            write_csv(
                out_dir / "epoch_metrics.csv",
                epoch_rows,
                ["epoch", "epoch_elapsed_sec", *sorted(last_parts)],
            )
        if args.eval_every_epoch:
            epoch_eval_time = log_step(f"starting per-epoch eval {epoch + 1}/{args.epochs}")
            epoch_eval_metrics, _, _, _ = evaluate(
                model,
                eval_examples,
                args,
                device,
                "eval",
                prior_store=prior_store,
                prior_proj=prior_proj,
                prior_cell_query_proj=prior_cell_query_proj,
                prior_cell_adapter=prior_cell_adapter,
                collect_rows=False,
            )
            eval_row: dict[str, float | int] = {
                "epoch": epoch + 1,
                "epoch_train_elapsed_sec": elapsed_epoch,
                **last_parts,
                **prefix_metrics(epoch_eval_metrics, "eval"),
            }
            if args.eval_every_epoch_train:
                epoch_train_eval_time = log_step(f"starting per-epoch train eval {epoch + 1}/{args.epochs}")
                epoch_train_metrics, _, _, _ = evaluate(
                    model,
                    train_examples,
                    args,
                    device,
                    "train",
                    prior_store=prior_store,
                    prior_proj=prior_proj,
                    prior_cell_query_proj=prior_cell_query_proj,
                    prior_cell_adapter=prior_cell_adapter,
                    collect_rows=False,
                )
                eval_row.update(prefix_metrics(epoch_train_metrics, "train"))
                log_step(f"finished per-epoch train eval {epoch + 1}/{args.epochs}", epoch_train_eval_time)
            eval_row["epoch_eval_elapsed_sec"] = time.time() - epoch_eval_time
            epoch_eval_rows.append(eval_row)
            eval_fieldnames = [
                "epoch",
                *sorted({key for row in epoch_eval_rows for key in row if key != "epoch"}),
            ]
            write_csv(out_dir / "epoch_eval_metrics.csv", epoch_eval_rows, eval_fieldnames)
            epoch_eval_summary_rows.append(epoch_eval_summary_row(eval_row))
            write_csv(
                out_dir / "epoch_eval_summary.csv",
                epoch_eval_summary_rows,
                EPOCH_EVAL_SUMMARY_FIELDNAMES,
            )
            log_step(f"finished per-epoch eval {epoch + 1}/{args.epochs}", epoch_eval_time)
        log_step(f"finished epoch {epoch + 1}/{args.epochs}", epoch_time)

    train_metrics: dict[str, float] = {}
    train_prediction_rows: list[dict[str, object]] = []
    train_evidence_weight_rows: list[dict[str, object]] = []
    train_cell_weight_rows: list[dict[str, object]] = []
    if args.eval_train_split:
        train_eval_time = log_step("starting train split evaluation")
        (
            train_metrics,
            train_prediction_rows,
            train_evidence_weight_rows,
            train_cell_weight_rows,
        ) = evaluate(
            model,
            train_examples,
            args,
            device,
            "train",
            prior_store=prior_store,
            prior_proj=prior_proj,
            prior_cell_query_proj=prior_cell_query_proj,
            prior_cell_adapter=prior_cell_adapter,
            cell_noise_metadata=train_cell_noise_metadata,
        )
        log_step("finished train split evaluation", train_eval_time)

    eval_time = log_step("starting eval split evaluation")
    eval_metrics, prediction_rows, evidence_weight_rows, cell_weight_rows = evaluate(
        model,
        eval_examples,
        args,
        device,
        "eval",
        prior_store=prior_store,
        prior_proj=prior_proj,
        prior_cell_query_proj=prior_cell_query_proj,
        prior_cell_adapter=prior_cell_adapter,
        cell_noise_metadata=eval_cell_noise_metadata,
    )
    log_step("finished eval split evaluation", eval_time)
    metrics = {
        **last_parts,
        **eval_metrics,
        **prefix_metrics(eval_metrics, "eval"),
        **prefix_metrics(train_metrics, "train"),
        "run_id": args.run_id,
        "stage": "stage1_alignment",
        "elapsed_sec": time.time() - start_time,
        "wandb": False,
        "num_train_examples": len(train_examples),
        "num_eval_examples": len(eval_examples),
        "num_matched_examples": len(examples),
        "paired_eval_all_genes": bool(args.paired_eval_all_genes),
        "example_split_mode": "paired_eval_all_genes" if args.paired_eval_all_genes else "gene_split",
        "gene_allowlist": args.gene_allowlist,
        "gene_allowlist_enabled": bool(gene_allowlist_summary.get("enabled", False)),
        "gene_allowlist_size": int(gene_allowlist_summary.get("requested_allowlist_genes", 0)),
        "requested_allowlist_genes": int(gene_allowlist_summary.get("requested_allowlist_genes", 0)),
        "matched_allowlist_genes": int(gene_allowlist_summary.get("matched_allowlist_genes", 0)),
        "dropped_allowlist_genes": int(gene_allowlist_summary.get("dropped_allowlist_genes", 0)),
        "num_text_evidence_sets": len(evidence_sets),
        "num_neighbor_rows": sum(len(example.neighbors) for example in examples),
        "num_examples_with_neighbors": sum(1 for example in examples if example.neighbors),
        "num_stage3_confusion_rows": sum(len(example.confusions) for example in examples),
        "num_examples_with_stage3_confusions": sum(1 for example in examples if example.confusions),
        "single_hidden_trained": bool(args.lambda_single > 0 or args.lambda_anchor > 0 or args.set_fusion == "residual"),
        "single_projection_trained": bool(args.lambda_single > 0),
        "proj_warmup_epochs": args.proj_warmup_epochs,
        "stage0_neighbor_prior_enabled": bool(prior_store is not None),
        "stage0_neighbor_prior_genes": len(prior_store.genes) if prior_store is not None else 0,
        "stage0_negative_neighbor_prior_enabled": bool(negative_prior_store is not None),
        "stage0_negative_neighbor_prior_genes": len(negative_prior_store.genes) if negative_prior_store is not None else 0,
        "stage3_prior_cell_attention_enabled": bool(prior_cell_query_proj is not None and prior_cell_adapter is not None),
        "instance_score_enabled": bool(args.instance_score_alpha > 0 and args.instance_score_npz),
        "instance_score_alpha": float(args.instance_score_alpha),
        "instance_score_mode": args.instance_score_mode,
        "cell_denoise_mode": args.cell_denoise_mode,
        "lambda_denoise": float(args.lambda_denoise),
        "lambda_cell_denoise": float(args.lambda_cell_denoise),
        "lambda_text_denoise": float(args.lambda_text_denoise),
        "text_aware_set_enabled": _uses_text_aware_set(args),
        "text_conf_mode": args.text_conf_mode,
        "text_conf_gate_mode": args.text_conf_gate_mode,
        "text_conf_gate_scale": float(args.text_conf_gate_scale),
        "text_conf_beta_init": float(args.text_conf_beta_init),
        "text_conf_beta_final": float(model.text_set.beta.detach().cpu()),
        "lambda_text_aware_set": float(args.lambda_text_aware_set),
        "text_aware_instance_alpha": float(args.text_aware_instance_alpha),
        "text_aware_temperature": float(args.text_aware_temperature),
        "lambda_text_noise_conf": float(args.lambda_text_noise_conf),
        "text_noise_conf_margin": float(args.text_noise_conf_margin),
        "fused_set_enabled": bool(args.fused_set_beta > 0),
        "fused_set_beta": float(args.fused_set_beta),
        "lambda_fused_set": float(args.lambda_fused_set),
        "fused_set_temperature": float(args.fused_set_temperature),
        "text_noise_ratio": float(args.text_noise_ratio),
        "text_noise_kind": args.text_noise_kind,
        "text_noise_splits": args.text_noise_splits,
        "text_noise_prior_mode": args.text_noise_prior_mode,
        "text_noise_train_bank": args.text_noise_train_bank,
        "text_noise_eval_bank": args.text_noise_eval_bank,
        "text_noise_train_noise_evidence": int(train_text_noise_summary.get("noise_evidence", 0)),
        "text_noise_train_noise_fraction": float(train_text_noise_summary.get("noise_fraction", 0.0)),
        "text_noise_train_bank_size": int(train_text_noise_summary.get("bank_size", 0)),
        "text_noise_train_replacement": bool(train_text_noise_summary.get("replacement", False)),
        "text_noise_eval_noise_evidence": int(eval_text_noise_summary.get("noise_evidence", 0)),
        "text_noise_eval_noise_fraction": float(eval_text_noise_summary.get("noise_fraction", 0.0)),
        "text_noise_eval_bank_size": int(eval_text_noise_summary.get("bank_size", 0)),
        "text_noise_eval_replacement": bool(eval_text_noise_summary.get("replacement", False)),
        "cell_noise_ratio": float(args.cell_noise_ratio),
        "cell_noise_max": int(args.cell_noise_max),
        "cell_noise_splits": args.cell_noise_splits,
        "cell_noise_source": args.cell_noise_source,
        "cell_noise_instance_score_mode": args.cell_noise_instance_score_mode,
        "cell_noise_train_noise_instances": int(train_cell_noise_summary.get("noise_instances", 0)),
        "cell_noise_train_noise_fraction": float(train_cell_noise_summary.get("noise_fraction", 0.0)),
        "cell_noise_eval_noise_instances": int(eval_cell_noise_summary.get("noise_instances", 0)),
        "cell_noise_eval_noise_fraction": float(eval_cell_noise_summary.get("noise_fraction", 0.0)),
        "eval_text_conf_filter_mode": args.eval_text_conf_filter_mode,
        "eval_text_conf_filter_frac": float(args.eval_text_conf_filter_frac),
        "eval_text_conf_filter_min_keep": int(args.eval_text_conf_filter_min_keep),
        "instance_score_train_coverage": float(instance_score_summary.get("coverage", 0.0)) if instance_score_summary else 0.0,
        "instance_score_eval_coverage": float(eval_instance_score_summary.get("coverage", 0.0)) if eval_instance_score_summary else 0.0,
    }
    write_json(out_dir / "metrics.json", metrics)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "prior_proj_state_dict": prior_proj.state_dict() if prior_proj is not None else None,
            "prior_cell_query_proj_state_dict": prior_cell_query_proj.state_dict() if prior_cell_query_proj is not None else None,
            "prior_cell_adapter_state_dict": prior_cell_adapter.state_dict() if prior_cell_adapter is not None else None,
            "run_config": vars(args),
            "metrics": metrics,
        },
        out_dir / "checkpoint.pt",
    )
    write_json(
        out_dir / "run_config.json",
        {
            **vars(args),
            "text_evidence_input": args.text_evidence_store or args.text_evidence_npz,
            "distribution_summary": summarize_distributions(distributions),
            "text_evidence_summary": {
                "num_sets": len(evidence_sets),
                "min_evidence": min(len(e.text_ids) for e in evidence_sets),
                "max_evidence": max(len(e.text_ids) for e in evidence_sets),
                "text_dim": int(examples[0].evidence.embeddings.shape[1]),
                "metadata_dim": int(text_metadata_dim),
            },
            "gene_allowlist_summary": gene_allowlist_summary,
            "neighbor_summary": {
                "enabled": bool(args.max_neighbors > 0),
                "neighbor_csv": args.neighbor_csv,
                "built_neighbors": bool(args.build_neighbors),
                "num_neighbor_rows": sum(len(example.neighbors) for example in examples),
                "num_examples_with_neighbors": sum(1 for example in examples if example.neighbors),
            },
            "stage3_confusion_summary": {
                "enabled": bool(args.lambda_stage3_confusion > 0),
                "confusion_csv": args.stage3_confusion_csv,
                "lambda": args.lambda_stage3_confusion,
                "margin": args.stage3_confusion_margin,
                "mode": args.stage3_confusion_mode,
                "anchor": args.stage3_confusion_anchor,
                "max_negatives": args.stage3_confusion_max_negatives,
                "num_confusion_rows": sum(len(example.confusions) for example in examples),
                "num_examples_with_confusions": sum(1 for example in examples if example.confusions),
            },
            "stage0_neighbor_prior_summary": {
                "enabled": bool(prior_store is not None),
                "input": args.gene_neighbor_prior_npz,
                "num_genes": len(prior_store.genes) if prior_store is not None else 0,
                "dim": prior_store.dim if prior_store is not None else 0,
                "metadata": prior_store.metadata if prior_store is not None else {},
            },
            "stage0_negative_neighbor_prior_summary": {
                "enabled": bool(negative_prior_store is not None),
                "input": args.gene_neighbor_prior_negative_npz,
                "num_genes": len(negative_prior_store.genes) if negative_prior_store is not None else 0,
                "dim": negative_prior_store.dim if negative_prior_store is not None else 0,
                "metadata": negative_prior_store.metadata if negative_prior_store is not None else {},
            },
            "stage3_prior_cell_attention_summary": {
                "enabled": bool(prior_cell_query_proj is not None and prior_cell_adapter is not None),
                "lambda": args.lambda_neighbor_prior_cell_attn,
                "fusion_alpha": args.gene_neighbor_prior_cell_attn_fusion_alpha,
                "temperature": args.gene_neighbor_prior_cell_attn_temperature,
                "gate": args.gene_neighbor_prior_cell_attn_gate,
                "gate_threshold": args.gene_neighbor_prior_cell_attn_gate_threshold,
                "gate_temperature": args.gene_neighbor_prior_cell_attn_gate_temperature,
            },
            "instance_score_summary": {
                "enabled": bool(args.instance_score_alpha > 0 and args.instance_score_npz),
                "alpha": args.instance_score_alpha,
                "mode": args.instance_score_mode,
                "train": instance_score_summary,
                "eval": eval_instance_score_summary,
            },
            "cell_noise_summary": {
                "enabled": bool(args.cell_noise_ratio > 0),
                "ratio": args.cell_noise_ratio,
                "max_noise": args.cell_noise_max,
                "splits": args.cell_noise_splits,
                "source": args.cell_noise_source,
                "instance_score_mode": args.cell_noise_instance_score_mode,
                "train": train_cell_noise_summary,
                "eval": eval_cell_noise_summary,
            },
        },
    )
    write_csv(
        out_dir / "predictions.csv",
        prediction_rows,
        PREDICTION_FIELDNAMES,
    )
    if args.eval_train_split:
        write_csv(
            out_dir / "train_predictions.csv",
            train_prediction_rows,
            PREDICTION_FIELDNAMES,
        )
        write_csv(
            out_dir / "train_evidence_weights.csv",
            train_evidence_weight_rows,
            EVIDENCE_WEIGHT_FIELDNAMES,
        )
        write_evidence_weights_by_gene_md(
            out_dir / "train_evidence_weights.by_gene.md",
            train_evidence_weight_rows,
        )
        write_csv(
            out_dir / "train_cell_weights.csv",
            train_cell_weight_rows,
            CELL_WEIGHT_FIELDNAMES,
        )
    write_csv(
        out_dir / "evidence_weights.csv",
        evidence_weight_rows,
        EVIDENCE_WEIGHT_FIELDNAMES,
    )
    write_evidence_weights_by_gene_md(
        out_dir / "evidence_weights.by_gene.md",
        evidence_weight_rows,
    )
    write_csv(
        out_dir / "cell_weights.csv",
        cell_weight_rows,
        CELL_WEIGHT_FIELDNAMES,
    )
    print(metrics)


if __name__ == "__main__":
    main()

