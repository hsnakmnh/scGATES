from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scgates.io import GeneDistribution, TextEvidenceSet
from scgates.models import (
    DistributionBiasedSetEncoder,
    MeanEncoder,
    NeighborAggregator,
    NoiseAwareSetEncoder,
    TextAwareSetScorer,
    TextSetEncoder,
)
from scgates.retrieval import Neighbor
from scgates.sampling import normalize_1d, stratified_instance_indices


def _temperature_adjust_prior(prior: np.ndarray, prior_temperature: float) -> np.ndarray:
    prior = np.asarray(prior, dtype=np.float32)
    if prior.size == 0:
        return prior
    if float(prior.sum()) <= 0:
        prior = np.ones_like(prior, dtype=np.float32)
    prior = prior / max(float(prior.sum()), 1e-8)
    if prior_temperature > 0 and not np.isclose(prior_temperature, 1.0):
        prior = np.power(np.maximum(prior, 1e-8), 1.0 / prior_temperature).astype(np.float32)
        prior = prior / max(float(prior.sum()), 1e-8)
    return prior


def _prior_diverse_indices(
    embeddings: np.ndarray,
    prior: np.ndarray,
    max_evidence: int,
    diversity_weight: float = 0.7,
    candidate_multiplier: int = 8,
    seed: int | None = None,
) -> np.ndarray:
    """Greedy evidence-level selection using prior score and embedding diversity.

    When ``seed`` is provided, the starting point is chosen semi-randomly among
    top-prior candidates instead of always argmax. Useful for multi-view consistency."""
    total = int(embeddings.shape[0])
    if total <= max_evidence:
        return np.arange(total, dtype=np.int64)

    diversity_weight = float(np.clip(diversity_weight, 0.0, 1.0))
    candidate_count = min(total, max(max_evidence, max_evidence * candidate_multiplier))
    candidates = np.argsort(-prior, kind="mergesort")[:candidate_count].astype(np.int64)
    rng = np.random.RandomState(seed) if seed is not None else None

    candidate_emb = embeddings[candidates].astype(np.float32, copy=False)
    candidate_emb = candidate_emb / np.maximum(np.linalg.norm(candidate_emb, axis=1, keepdims=True), 1e-8)
    candidate_prior = prior[candidates].astype(np.float32, copy=True)
    candidate_prior = normalize_1d(candidate_prior)

    selected_local: list[int] = []
    selected_mask = np.zeros(candidate_count, dtype=bool)
    if rng is not None:
        top_pool = min(8, candidate_count)
        top_indices = np.argsort(-candidate_prior)[:top_pool]
        first = int(top_indices[rng.randint(0, top_pool)])
    else:
        first = int(np.argmax(candidate_prior))
    selected_local.append(first)
    selected_mask[first] = True
    max_similarity = candidate_emb @ candidate_emb[first]

    while len(selected_local) < max_evidence:
        novelty = 1.0 - max_similarity
        score = diversity_weight * novelty + (1.0 - diversity_weight) * candidate_prior
        score[selected_mask] = -np.inf
        next_idx = int(np.argmax(score))
        if not np.isfinite(score[next_idx]):
            break
        selected_local.append(next_idx)
        selected_mask[next_idx] = True
        max_similarity = np.maximum(max_similarity, candidate_emb @ candidate_emb[next_idx])

    return candidates[np.asarray(selected_local, dtype=np.int64)]


def pad_gene_distributions(
    distributions: Sequence[GeneDistribution],
    max_instances: int,
    device: torch.device,
    sampling_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(distributions)
    input_dim = distributions[0].embeddings.shape[1]
    emb = torch.zeros(batch_size, max_instances, input_dim, dtype=torch.float32, device=device)
    ranks = torch.zeros(batch_size, max_instances, dtype=torch.float32, device=device)
    expr = torch.zeros(batch_size, max_instances, dtype=torch.float32, device=device)
    mask = torch.zeros(batch_size, max_instances, dtype=torch.bool, device=device)
    occurrence = torch.zeros(batch_size, dtype=torch.float32, device=device)
    selected_indices = torch.full((batch_size, max_instances), -1, dtype=torch.long, device=device)
    instance_scores = torch.zeros(batch_size, max_instances, dtype=torch.float32, device=device)
    for i, dist in enumerate(distributions):
        seed_i = None if sampling_seed is None else int(sampling_seed + i)
        # view A uses mode="priority" (default), view B uses mode="random"
        use_mode = "random" if sampling_seed is not None and sampling_seed < 0 else "priority"
        actual_seed = abs(sampling_seed) + i if sampling_seed is not None and sampling_seed < 0 else seed_i
        selected = stratified_instance_indices(np.asarray(dist.ranks), np.asarray(dist.expr), max_instances, seed=actual_seed, mode=use_mode)
        n = int(selected.shape[0])
        emb[i, :n] = torch.from_numpy(np.asarray(dist.embeddings[selected], dtype=np.float32).copy()).to(device)
        ranks[i, :n] = torch.from_numpy(normalize_1d(np.asarray(dist.ranks[selected]).copy())).to(device)
        expr[i, :n] = torch.from_numpy(normalize_1d(np.asarray(dist.expr[selected]).copy())).to(device)
        if dist.instance_scores is not None:
            scores = np.asarray(dist.instance_scores[selected], dtype=np.float32).copy()
            instance_scores[i, :n] = torch.from_numpy(scores).to(device)
        selected_indices[i, :n] = torch.from_numpy(selected).to(device)
        mask[i, :n] = True
        occurrence[i] = float(dist.occurrence_rate)
    return emb, ranks, expr, mask, occurrence, selected_indices, instance_scores


def pad_text_evidence_sets(
    evidence_sets: Sequence[TextEvidenceSet],
    max_evidence: int,
    device: torch.device,
    sampling: str = "prior_diverse",
    prior_temperature: float = 1.0,
    diversity_weight: float = 0.7,
    sampling_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(evidence_sets)
    input_dim = evidence_sets[0].embeddings.shape[1]
    if max_evidence <= 0:
        max_evidence = max(int(evidence.embeddings.shape[0]) for evidence in evidence_sets)
    metadata_dim = 0
    for evidence in evidence_sets:
        if evidence.metadata_features is not None:
            metadata_dim = int(evidence.metadata_features.shape[1])
            break
    emb = torch.zeros(batch_size, max_evidence, input_dim, dtype=torch.float32, device=device)
    prior = torch.zeros(batch_size, max_evidence, dtype=torch.float32, device=device)
    metadata = torch.zeros(batch_size, max_evidence, metadata_dim, dtype=torch.float32, device=device)
    indices = torch.full((batch_size, max_evidence), -1, dtype=torch.long, device=device)
    mask = torch.zeros(batch_size, max_evidence, dtype=torch.bool, device=device)
    for i, evidence in enumerate(evidence_sets):
        total = evidence.embeddings.shape[0]
        adjusted_prior = _temperature_adjust_prior(evidence.prior, prior_temperature)
        if total <= max_evidence:
            selected = np.arange(total, dtype=np.int64)
        elif sampling == "first":
            selected = np.arange(max_evidence, dtype=np.int64)
        elif sampling == "random_prior":
            probs = adjusted_prior.astype(np.float64)
            probs = probs / max(float(probs.sum()), 1e-12)
            selected = np.sort(np.random.choice(total, size=max_evidence, replace=False, p=probs)).astype(np.int64)
        elif sampling == "top_prior":
            selected = np.argsort(-adjusted_prior, kind="mergesort")[:max_evidence].astype(np.int64)
        elif sampling == "prior_diverse":
            seed_i = None if sampling_seed is None else int(sampling_seed + i)
            # view B uses random_prior to produce meaningfully different evidence subsets
            if sampling_seed is not None and sampling_seed < 0:
                probs = adjusted_prior.astype(np.float64)
                probs = probs / max(float(probs.sum()), 1e-12)
                actual_seed = abs(sampling_seed) + i
                selected = np.sort(np.random.RandomState(actual_seed).choice(total, size=max_evidence, replace=False, p=probs)).astype(np.int64)
            else:
                selected = _prior_diverse_indices(
                    evidence.embeddings,
                    adjusted_prior,
                    max_evidence,
                    diversity_weight=diversity_weight,
                    seed=seed_i,
                )
        else:
            raise ValueError(f"Unsupported text evidence sampling: {sampling}")
        n = len(selected)
        emb[i, :n] = torch.from_numpy(evidence.embeddings[selected]).to(device)
        prior_i = adjusted_prior[selected].astype(np.float32)
        prior[i, :n] = torch.from_numpy(prior_i / max(float(prior_i.sum()), 1e-8)).to(device)
        if metadata_dim and evidence.metadata_features is not None:
            metadata[i, :n] = torch.from_numpy(evidence.metadata_features[selected]).to(device)
        indices[i, :n] = torch.from_numpy(selected).to(device)
        mask[i, :n] = True
    return emb, prior, mask, metadata, indices


class Stage1AlignmentModel(nn.Module):
    def __init__(
        self,
        cell_dim: int,
        text_dim: int,
        hidden_dim: int = 128,
        align_dim: int = 128,
        dropout: float = 0.1,
        text_metadata_dim: int = 0,
        cell_encoder: str = "distribution_mha",
        cell_num_heads: int = 4,
        cell_num_layers: int = 2,
        cell_denoise_mode: str = "normal",
        text_conf_beta_init: float = 0.0,
        text_conf_mode: str = "text_only",
        text_conf_gate_mode: str = "none",
        text_conf_gate_scale: float = 1.0,
    ):
        super().__init__()
        if cell_encoder == "distribution_mha":
            self.cell_set = DistributionBiasedSetEncoder(
                cell_dim,
                hidden_dim,
                dropout=dropout,
                num_heads=cell_num_heads,
                num_layers=cell_num_layers,
                denoise_mode=cell_denoise_mode,
            )
        elif cell_encoder == "attention_pool":
            self.cell_set = NoiseAwareSetEncoder(cell_dim, hidden_dim, dropout, denoise_mode=cell_denoise_mode)
        else:
            raise ValueError(f"Unsupported Stage1 cell encoder: {cell_encoder}")
        self.cell_single = MeanEncoder(cell_dim, hidden_dim)
        self.text_set = TextSetEncoder(
            text_dim,
            hidden_dim,
            dropout,
            metadata_dim=text_metadata_dim,
            conf_mode=text_conf_mode,
            conf_gate_mode=text_conf_gate_mode,
            conf_gate_scale=text_conf_gate_scale,
        )
        if text_conf_beta_init != 0.0:
            with torch.no_grad():
                self.text_set.beta.fill_(float(text_conf_beta_init))
        self.text_single = MeanEncoder(text_dim, hidden_dim)
        self.neighbor_agg = NeighborAggregator(hidden_dim)
        self.cell_single_proj = nn.Linear(hidden_dim, align_dim)
        self.cell_set_proj = nn.Linear(hidden_dim, align_dim)
        self.text_single_proj = nn.Linear(hidden_dim, align_dim)
        self.text_set_proj = nn.Linear(hidden_dim, align_dim)
        self.text_aware_set_scorer = TextAwareSetScorer(align_dim, align_dim, dropout)

    def forward(
        self,
        distributions: Sequence[GeneDistribution],
        evidence_sets: Sequence[TextEvidenceSet],
        max_instances: int,
        max_text_evidence: int,
        device: torch.device,
        text_evidence_sampling: str = "prior_diverse",
        text_prior_temperature: float = 1.0,
        text_diversity_weight: float = 0.7,
        neighbor_distributions: Sequence[Sequence[GeneDistribution]] | None = None,
        neighbor_evidence_sets: Sequence[Sequence[TextEvidenceSet]] | None = None,
        confusion_evidence_sets: Sequence[Sequence[TextEvidenceSet]] | None = None,
        neighbor_infos: Sequence[Sequence[Neighbor]] | None = None,
        max_neighbors: int = 0,
        neighbor_agg_alpha: float = 1.0,
        instance_score_alpha: float = 0.0,
        instance_score_mode: str = "signed",
        max_confusions: int = 0,
        set_fusion: str = "direct",
        residual_scale: float = 0.25,
        sampling_seed: int | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        cell_emb, ranks, expr, cell_mask, occurrence, cell_selected_indices, cell_instance_scores = pad_gene_distributions(
            distributions, max_instances, device, sampling_seed=sampling_seed
        )
        text_emb, text_prior, text_mask, text_metadata, text_indices = pad_text_evidence_sets(
            evidence_sets,
            max_text_evidence,
            device,
            sampling=text_evidence_sampling,
            prior_temperature=text_prior_temperature,
            diversity_weight=text_diversity_weight,
            sampling_seed=sampling_seed,
        )

        cell_single = self.cell_single(cell_emb, cell_mask)
        text_single = self.text_single(text_emb, text_mask)
        cell_aux = self.cell_set(
            cell_emb,
            ranks,
            expr,
            cell_mask,
            occurrence,
            instance_scores=cell_instance_scores,
            instance_score_alpha=instance_score_alpha,
            instance_score_mode=instance_score_mode,
        )
        text_aux = self.text_set(text_emb, text_prior, text_mask, text_metadata)
        neighbor_aux = self._encode_neighbors(
            neighbor_distributions=neighbor_distributions,
            neighbor_evidence_sets=neighbor_evidence_sets,
            neighbor_infos=neighbor_infos,
            batch_size=len(distributions),
            max_neighbors=max_neighbors,
            max_instances=max_instances,
            max_text_evidence=max_text_evidence,
            device=device,
            sampling=text_evidence_sampling,
            prior_temperature=text_prior_temperature,
            diversity_weight=text_diversity_weight,
            sampling_seed=sampling_seed,
        )
        confusion_text_aux = self._encode_confusion_texts(
            confusion_evidence_sets=confusion_evidence_sets,
            batch_size=len(distributions),
            max_confusions=max_confusions,
            max_text_evidence=max_text_evidence,
            device=device,
            sampling=text_evidence_sampling,
            prior_temperature=text_prior_temperature,
            diversity_weight=text_diversity_weight,
            sampling_seed=sampling_seed,
        )
        if neighbor_aux is not None and neighbor_agg_alpha > 0:
            neighbor_struct_hidden = self.neighbor_agg(
                cell_aux["z"],
                neighbor_aux["neighbor_cell_hidden"],
                neighbor_aux["neighbor_mask"],
                neighbor_aux["neighbor_prior_logits"],
            )
            alpha = float(np.clip(neighbor_agg_alpha, 0.0, 1.0))
            cell_struct_hidden = (1.0 - alpha) * cell_aux["z"] + alpha * neighbor_struct_hidden
        else:
            cell_struct_hidden = cell_aux["z"]
        text_struct_hidden = text_aux["z"]

        if set_fusion == "direct":
            cell_set_hidden = cell_struct_hidden
            text_set_hidden = text_struct_hidden
        elif set_fusion == "residual":
            cell_set_hidden = cell_single + residual_scale * cell_struct_hidden
            text_set_hidden = text_single + residual_scale * text_struct_hidden
        else:
            raise ValueError(f"Unsupported set fusion mode: {set_fusion}")

        out: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "cell_single": F.normalize(self.cell_single_proj(cell_single), dim=-1),
            "text_single": F.normalize(self.text_single_proj(text_single), dim=-1),
            "cell_set": F.normalize(self.cell_set_proj(cell_set_hidden), dim=-1),
            "cell_set_base": F.normalize(self.cell_set_proj(cell_struct_hidden), dim=-1),
            "cell_instance_set": F.normalize(self.cell_set_proj(cell_aux["instance_hidden"]), dim=-1),
            "text_set": F.normalize(self.text_set_proj(text_set_hidden), dim=-1),
            "text_set_base": F.normalize(self.text_set_proj(text_struct_hidden), dim=-1),
            "cell_aux": cell_aux,
            "text_aux": text_aux,
            "cell_mask": cell_mask,
            "text_mask": text_mask,
            "cell_embeddings": cell_emb,
            "cell_ranks": ranks,
            "cell_expr": expr,
            "cell_occurrence": occurrence,
            "cell_instance_scores": cell_instance_scores,
            "cell_selected_indices": cell_selected_indices,
            "text_embeddings": text_emb,
            "text_prior": text_prior,
            "text_metadata": text_metadata,
            "text_indices": text_indices,
        }
        if neighbor_aux is not None:
            out.update(
                {
                    "neighbor_cell_set": F.normalize(self.cell_set_proj(neighbor_aux["neighbor_cell_hidden"]), dim=-1),
                    "neighbor_text_set": F.normalize(self.text_set_proj(neighbor_aux["neighbor_text_hidden"]), dim=-1),
                    "neighbor_mask": neighbor_aux["neighbor_mask"],
                    "neighbor_prior_logits": neighbor_aux["neighbor_prior_logits"],
                }
            )
        if confusion_text_aux is not None:
            out.update(
                {
                    "confusion_text_set": F.normalize(self.text_set_proj(confusion_text_aux["confusion_text_hidden"]), dim=-1),
                    "confusion_mask": confusion_text_aux["confusion_mask"],
                }
            )
        return out

    def _encode_neighbors(
        self,
        neighbor_distributions: Sequence[Sequence[GeneDistribution]] | None,
        neighbor_evidence_sets: Sequence[Sequence[TextEvidenceSet]] | None,
        neighbor_infos: Sequence[Sequence[Neighbor]] | None,
        batch_size: int,
        max_neighbors: int,
        max_instances: int,
        max_text_evidence: int,
        device: torch.device,
        sampling: str,
        prior_temperature: float,
        diversity_weight: float,
        sampling_seed: int | None,
    ) -> dict[str, torch.Tensor] | None:
        if max_neighbors <= 0 or neighbor_distributions is None:
            return None

        flat_dists: list[GeneDistribution] = []
        flat_evidence: list[TextEvidenceSet] = []
        counts: list[int] = []
        prior_rows: list[list[float]] = []
        if neighbor_infos is None:
            neighbor_infos = [[] for _ in range(len(neighbor_distributions))]
        if neighbor_evidence_sets is None:
            neighbor_evidence_sets = [[] for _ in range(len(neighbor_distributions))]
        for dists, evidence_sets, infos in zip(neighbor_distributions, neighbor_evidence_sets, neighbor_infos):
            if not infos:
                infos = [None] * len(dists)  # type: ignore[list-item]
            paired = list(zip(dists, evidence_sets, infos))[:max_neighbors]
            counts.append(len(paired))
            row_prior: list[float] = []
            for dist, evidence, info in paired:
                flat_dists.append(dist)
                flat_evidence.append(evidence)
                if info is None:
                    row_prior.append(0.0)
                else:
                    rank_prior = 1.0 / max(float(info.rank), 1.0)
                    row_prior.append(float(info.cosine_score) - float(info.sinkhorn_cost) + 0.25 * rank_prior)
            if row_prior:
                row = np.asarray(row_prior, dtype=np.float32)
                row = row - float(row.mean())
                row_std = float(row.std())
                if row_std > 1e-6:
                    row = row / row_std
                prior_rows.append(row.astype(np.float32).tolist())
            else:
                prior_rows.append([])

        hidden_dim = self.cell_single_proj.in_features
        text_hidden_dim = self.text_set_proj.in_features
        neighbor_cell_hidden = torch.zeros(batch_size, max_neighbors, hidden_dim, dtype=torch.float32, device=device)
        neighbor_text_hidden = torch.zeros(batch_size, max_neighbors, text_hidden_dim, dtype=torch.float32, device=device)
        neighbor_mask = torch.zeros(batch_size, max_neighbors, dtype=torch.bool, device=device)
        neighbor_prior_logits = torch.zeros(batch_size, max_neighbors, dtype=torch.float32, device=device)
        if not flat_dists:
            return {
                "neighbor_cell_hidden": neighbor_cell_hidden,
                "neighbor_text_hidden": neighbor_text_hidden,
                "neighbor_mask": neighbor_mask,
                "neighbor_prior_logits": neighbor_prior_logits,
            }

        n_cell_emb, n_ranks, n_expr, n_cell_mask, n_occurrence, _, n_instance_scores = pad_gene_distributions(
            flat_dists, max_instances, device
        )
        n_cell_aux = self.cell_set(n_cell_emb, n_ranks, n_expr, n_cell_mask, n_occurrence)
        n_text_emb, n_text_prior, n_text_mask, n_text_metadata, _ = pad_text_evidence_sets(
            flat_evidence,
            max_text_evidence,
            device,
            sampling=sampling,
            prior_temperature=prior_temperature,
            diversity_weight=diversity_weight,
            sampling_seed=sampling_seed,
        )
        n_text_aux = self.text_set(n_text_emb, n_text_prior, n_text_mask, n_text_metadata)

        offset = 0
        for row_idx, count in enumerate(counts):
            if count <= 0:
                continue
            neighbor_cell_hidden[row_idx, :count] = n_cell_aux["z"][offset : offset + count]
            neighbor_text_hidden[row_idx, :count] = n_text_aux["z"][offset : offset + count]
            neighbor_mask[row_idx, :count] = True
            neighbor_prior_logits[row_idx, :count] = torch.tensor(prior_rows[row_idx], dtype=torch.float32, device=device)
            offset += count
        return {
            "neighbor_cell_hidden": neighbor_cell_hidden,
            "neighbor_text_hidden": neighbor_text_hidden,
            "neighbor_mask": neighbor_mask,
            "neighbor_prior_logits": neighbor_prior_logits,
        }

    def _encode_confusion_texts(
        self,
        confusion_evidence_sets: Sequence[Sequence[TextEvidenceSet]] | None,
        batch_size: int,
        max_confusions: int,
        max_text_evidence: int,
        device: torch.device,
        sampling: str,
        prior_temperature: float,
        diversity_weight: float,
        sampling_seed: int | None,
    ) -> dict[str, torch.Tensor] | None:
        if max_confusions <= 0 or confusion_evidence_sets is None:
            return None

        hidden_dim = self.text_set_proj.in_features
        confusion_text_hidden = torch.zeros(batch_size, max_confusions, hidden_dim, dtype=torch.float32, device=device)
        confusion_mask = torch.zeros(batch_size, max_confusions, dtype=torch.bool, device=device)
        flat_evidence: list[TextEvidenceSet] = []
        row_lengths: list[int] = []
        for row in confusion_evidence_sets:
            selected = list(row[:max_confusions])
            row_lengths.append(len(selected))
            flat_evidence.extend(selected)
        if not flat_evidence:
            return {
                "confusion_text_hidden": confusion_text_hidden,
                "confusion_mask": confusion_mask,
            }

        text_emb, text_prior, text_mask, text_metadata, _ = pad_text_evidence_sets(
            flat_evidence,
            max_text_evidence,
            device,
            sampling=sampling,
            prior_temperature=prior_temperature,
            diversity_weight=diversity_weight,
            sampling_seed=sampling_seed,
        )
        text_aux = self.text_set(text_emb, text_prior, text_mask, text_metadata)
        offset = 0
        for row_idx, count in enumerate(row_lengths):
            if count <= 0:
                continue
            confusion_text_hidden[row_idx, :count] = text_aux["z"][offset : offset + count]
            confusion_mask[row_idx, :count] = True
            offset += count
        return {
            "confusion_text_hidden": confusion_text_hidden,
            "confusion_mask": confusion_mask,
        }

