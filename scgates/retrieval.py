from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scgates.io import GeneDistribution
from scgates.ot import l2_normalize, sinkhorn_cost
from scgates.sampling import stratified_instance_indices


@dataclass(frozen=True)
class Neighbor:
    query_key: str
    neighbor_key: str
    query_gene: str
    neighbor_gene: str
    context: str
    rank: int
    cosine_score: float
    sinkhorn_cost: float


def distribution_mean(dist: GeneDistribution, max_instances: int = 0) -> np.ndarray:
    selected = stratified_instance_indices(dist.ranks, dist.expr, max_instances)
    embeddings = dist.embeddings[selected]
    return np.asarray(embeddings, dtype=np.float32).mean(axis=0).astype(np.float32)


def _distribution_embeddings(dist: GeneDistribution, max_instances: int = 0) -> np.ndarray:
    selected = stratified_instance_indices(dist.ranks, dist.expr, max_instances)
    embeddings = dist.embeddings[selected]
    return np.asarray(embeddings, dtype=np.float32)


def retrieve_neighbors(
    distributions: list[GeneDistribution],
    top_k: int = 20,
    ann_k: int = 200,
    sinkhorn_epsilon: float = 0.08,
    sinkhorn_iter: int = 50,
    same_context_only: bool = True,
    max_instances: int = 0,
    skip_sinkhorn: bool = False,
) -> dict[str, list[Neighbor]]:
    if not distributions:
        return {}
    means = np.stack([distribution_mean(d, max_instances=max_instances) for d in distributions], axis=0)
    means = l2_normalize(means)
    sim = means @ means.T
    contexts = np.asarray([d.context for d in distributions], dtype=object)
    by_key: dict[str, list[Neighbor]] = {}
    candidate_count = max(top_k, ann_k)

    for i, query in enumerate(distributions):
        scores = sim[i].copy()
        scores[i] = -np.inf
        if same_context_only:
            scores[contexts != query.context] = -np.inf
        valid_idx = np.flatnonzero(np.isfinite(scores))
        if valid_idx.size == 0:
            by_key[query.key] = []
            continue
        if valid_idx.size > candidate_count:
            selected = valid_idx[np.argpartition(-scores[valid_idx], candidate_count - 1)[:candidate_count]]
        else:
            selected = valid_idx
        selected = selected[np.argsort(-scores[selected])]
        candidates = [(float(scores[j]), int(j)) for j in selected]

        reranked: list[tuple[float, float, int]] = []
        if skip_sinkhorn:
            for cos_score, j in candidates:
                reranked.append((1.0 - cos_score, cos_score, j))
            reranked.sort(key=lambda item: item[0])
        else:
            for cos_score, j in candidates:
                cost = sinkhorn_cost(
                    _distribution_embeddings(query, max_instances=max_instances),
                    _distribution_embeddings(distributions[j], max_instances=max_instances),
                    epsilon=sinkhorn_epsilon,
                    max_iter=sinkhorn_iter,
                )
                reranked.append((cost, cos_score, j))
            reranked.sort(key=lambda item: item[0])

        neighbors: list[Neighbor] = []
        for rank, (cost, cos_score, j) in enumerate(reranked[:top_k], start=1):
            candidate = distributions[j]
            neighbors.append(
                Neighbor(
                    query_key=query.key,
                    neighbor_key=candidate.key,
                    query_gene=query.gene,
                    neighbor_gene=candidate.gene,
                    context=query.context,
                    rank=rank,
                    cosine_score=cos_score,
                    sinkhorn_cost=cost,
                )
            )
        by_key[query.key] = neighbors
    return by_key


def load_neighbor_csv(path: str | Path) -> dict[str, list[Neighbor]]:
    by_key: dict[str, list[Neighbor]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_key = str(row["query_key"])
            neighbor = Neighbor(
                query_key=query_key,
                neighbor_key=str(row["neighbor_key"]),
                query_gene=str(row["query_gene"]),
                neighbor_gene=str(row["neighbor_gene"]),
                context=str(row["context"]),
                rank=int(row["rank"]),
                cosine_score=float(row["cosine_score"]),
                sinkhorn_cost=float(row["sinkhorn_cost"]),
            )
            by_key.setdefault(query_key, []).append(neighbor)
    for neighbors in by_key.values():
        neighbors.sort(key=lambda item: item.rank)
    return by_key


def write_neighbor_csv(path: str | Path, neighbor_map: dict[str, list[Neighbor]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_key",
        "neighbor_key",
        "query_gene",
        "neighbor_gene",
        "context",
        "rank",
        "cosine_score",
        "sinkhorn_cost",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in neighbor_rows(neighbor_map):
            writer.writerow(row)


def neighbor_rows(neighbor_map: dict[str, list[Neighbor]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for neighbors in neighbor_map.values():
        for n in neighbors:
            rows.append(
                {
                    "query_key": n.query_key,
                    "neighbor_key": n.neighbor_key,
                    "query_gene": n.query_gene,
                    "neighbor_gene": n.neighbor_gene,
                    "context": n.context,
                    "rank": n.rank,
                    "cosine_score": n.cosine_score,
                    "sinkhorn_cost": n.sinkhorn_cost,
                }
            )
    return rows
