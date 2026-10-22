from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scgates.io import GeneDistribution
from scgates.metrics import retrieval_metrics
from scgates.models import AlignmentHead, MeanEncoder, NeighborAggregator, NoiseAwareSetEncoder
from scgates.retrieval import Neighbor


@dataclass
class Example:
    dist: GeneDistribution
    text_vec: np.ndarray
    text_id: str
    neighbors: list[GeneDistribution]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_by_gene(distributions: list[GeneDistribution], seed: int = 42) -> dict[str, set[str]]:
    genes = sorted({d.gene for d in distributions})
    rng = random.Random(seed)
    rng.shuffle(genes)
    n = len(genes)
    n_train = max(1, int(0.7 * n))
    n_val = max(1, int(0.15 * n)) if n >= 6 else 0
    return {
        "train": set(genes[:n_train]),
        "val": set(genes[n_train : n_train + n_val]),
        "test": set(genes[n_train + n_val :]),
    }


def pad_distributions(
    distributions: Sequence[GeneDistribution],
    max_instances: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(distributions)
    input_dim = distributions[0].embeddings.shape[1]
    emb = torch.zeros(batch_size, max_instances, input_dim, dtype=torch.float32, device=device)
    ranks = torch.zeros(batch_size, max_instances, dtype=torch.float32, device=device)
    expr = torch.zeros(batch_size, max_instances, dtype=torch.float32, device=device)
    mask = torch.zeros(batch_size, max_instances, dtype=torch.bool, device=device)
    for i, dist in enumerate(distributions):
        n = min(max_instances, dist.embeddings.shape[0])
        emb[i, :n] = torch.from_numpy(dist.embeddings[:n]).to(device)
        ranks[i, :n] = torch.from_numpy(_normalize_1d(dist.ranks[:n])).to(device)
        expr[i, :n] = torch.from_numpy(_normalize_1d(dist.expr[:n])).to(device)
        mask[i, :n] = True
    return emb, ranks, expr, mask


def _normalize_1d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    if x.size == 0:
        return x
    lo = float(np.min(x))
    hi = float(np.max(x))
    if math.isclose(lo, hi):
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo)


class MVPModel(nn.Module):
    def __init__(self, input_dim: int, text_dim: int, hidden_dim: int, variant: str, dropout: float = 0.1):
        super().__init__()
        self.variant = variant
        if variant == "full":
            self.encoder = NoiseAwareSetEncoder(input_dim, hidden_dim, dropout)
            self.aggregator = NeighborAggregator(hidden_dim)
        elif variant in {"mean", "static", "neighbor_concat"}:
            self.encoder = MeanEncoder(input_dim, hidden_dim)
            self.aggregator = NeighborAggregator(hidden_dim)
        else:
            raise ValueError(f"Unsupported variant: {variant}")
        self.head = AlignmentHead(hidden_dim, text_dim)

    def encode_dists(
        self,
        distributions: Sequence[GeneDistribution],
        max_instances: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        emb, ranks, expr, mask = pad_distributions(distributions, max_instances, device)
        aux: dict[str, torch.Tensor] = {}
        if self.variant == "full":
            out = self.encoder(emb, ranks, expr, mask)
            aux = out
            return out["z"], aux
        return self.encoder(emb, mask), aux

    def forward_batch(
        self,
        examples: Sequence[Example],
        max_instances: int,
        max_neighbors: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target_z, aux = self.encode_dists([e.dist for e in examples], max_instances, device)
        if self.variant in {"full", "neighbor_concat"} and max_neighbors > 0:
            flat_neighbors: list[GeneDistribution] = []
            counts: list[int] = []
            for e in examples:
                selected = e.neighbors[:max_neighbors]
                counts.append(len(selected))
                flat_neighbors.extend(selected)
            if flat_neighbors:
                neighbor_z, _ = self.encode_dists(flat_neighbors, max_instances, device)
                hidden_dim = neighbor_z.shape[-1]
                padded = torch.zeros(len(examples), max_neighbors, hidden_dim, device=device)
                mask = torch.zeros(len(examples), max_neighbors, dtype=torch.bool, device=device)
                offset = 0
                for i, count in enumerate(counts):
                    if count:
                        padded[i, :count] = neighbor_z[offset : offset + count]
                        mask[i, :count] = True
                        offset += count
                z = self.aggregator(target_z, padded, mask)
            else:
                z = target_z
        else:
            z = target_z
        return self.head(z), aux


def build_examples(
    distributions: list[GeneDistribution],
    text_records: list[dict[str, str]],
    text_vectors: np.ndarray,
    neighbor_map: dict[str, list[Neighbor]],
) -> list[Example]:
    dist_by_key = {d.key: d for d in distributions}
    text_by_gene: dict[str, tuple[np.ndarray, str]] = {}
    for record, vec in zip(text_records, text_vectors):
        text_by_gene.setdefault(record["gene"], (vec.astype(np.float32), record["text_id"]))

    examples: list[Example] = []
    for dist in distributions:
        if dist.gene not in text_by_gene:
            continue
        neighbors = [
            dist_by_key[n.neighbor_key]
            for n in neighbor_map.get(dist.key, [])
            if n.neighbor_key in dist_by_key
        ]
        vec, text_id = text_by_gene[dist.gene]
        examples.append(Example(dist=dist, text_vec=vec, text_id=text_id, neighbors=neighbors))
    return examples


def train_and_evaluate(
    examples: list[Example],
    variant: str,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_dim: int,
    max_instances: int,
    max_neighbors: int,
    device: str,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    set_seed(seed)
    if not examples:
        raise ValueError("No examples available after matching distributions to text records")
    torch_device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    input_dim = examples[0].dist.embeddings.shape[1]
    text_dim = examples[0].text_vec.shape[0]
    model = MVPModel(input_dim, text_dim, hidden_dim, variant).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    splits = split_by_gene([e.dist for e in examples], seed)
    train_examples = [e for e in examples if e.dist.gene in splits["train"]]
    eval_examples = [e for e in examples if e.dist.gene in (splits["val"] | splits["test"])]
    if not eval_examples:
        eval_examples = train_examples

    for _ in range(epochs):
        random.shuffle(train_examples)
        model.train()
        for start in range(0, len(train_examples), batch_size):
            batch = train_examples[start : start + batch_size]
            if len(batch) < 2:
                continue
            pred, aux = model.forward_batch(batch, max_instances, max_neighbors, torch_device)
            target = torch.tensor(np.stack([e.text_vec for e in batch]), dtype=torch.float32, device=torch_device)
            target = F.normalize(target, dim=-1)
            logits = pred @ target.T / 0.07
            labels = torch.arange(len(batch), device=torch_device)
            loss = F.cross_entropy(logits, labels)
            if variant == "full" and aux:
                weights = aux["weights"]
                prior = aux["prior"]
                kl = torch.sum(weights * (torch.log(weights + 1e-8) - torch.log(prior + 1e-8)), dim=-1).mean()
                entropy = aux["entropy"].mean()
                loss = loss + 0.01 * kl + 0.001 * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_batches: list[np.ndarray] = []
        for start in range(0, len(eval_examples), batch_size):
            batch = eval_examples[start : start + batch_size]
            pred, _ = model.forward_batch(batch, max_instances, max_neighbors, torch_device)
            pred_batches.append(pred.cpu().numpy())
    pred_arr = np.concatenate(pred_batches, axis=0)
    target_arr = np.stack([e.text_vec for e in eval_examples], axis=0)
    metrics = retrieval_metrics(pred_arr, target_arr)
    metrics.update(
        {
            "num_train_examples": float(len(train_examples)),
            "num_eval_examples": float(len(eval_examples)),
            "num_genes": float(len({e.dist.gene for e in examples})),
        }
    )

    rows: list[dict[str, object]] = []
    scores = pred_arr @ (target_arr / np.maximum(np.linalg.norm(target_arr, axis=-1, keepdims=True), 1e-8)).T
    order = np.argsort(-scores, axis=1)
    for i, e in enumerate(eval_examples):
        rank = int(np.where(order[i] == i)[0][0]) + 1
        rows.append(
            {
                "key": e.dist.key,
                "gene": e.dist.gene,
                "context": e.dist.context,
                "text_id": e.text_id,
                "gold_rank": rank,
                "gold_score": float(scores[i, i]),
                "top1_text_id": eval_examples[int(order[i, 0])].text_id,
                "top1_gene": eval_examples[int(order[i, 0])].dist.gene,
                "variant": variant,
            }
        )
    return metrics, rows
