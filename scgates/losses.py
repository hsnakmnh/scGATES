from __future__ import annotations

import torch
import numpy as np
from torch.nn import functional as F


def bidirectional_info_nce(cell_z: torch.Tensor, text_z: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    cell_z = F.normalize(cell_z, dim=-1)
    text_z = F.normalize(text_z, dim=-1)
    logits = cell_z @ text_z.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def matching_ce(cell_z: torch.Tensor, text_z: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    cell_z = F.normalize(cell_z, dim=-1)
    text_z = F.normalize(text_z, dim=-1)
    logits = cell_z @ text_z.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)


def single_align_loss(
    cell_z: torch.Tensor,
    text_z: torch.Tensor,
    temperature_nce: float = 0.07,
    temperature_ce: float = 0.07,
) -> torch.Tensor:
    return bidirectional_info_nce(cell_z, text_z, temperature_nce) + matching_ce(cell_z, text_z, temperature_ce)


def set_align_loss(
    cell_z: torch.Tensor,
    text_z: torch.Tensor,
    temperature_nce: float = 0.07,
    temperature_ce: float = 0.07,
) -> torch.Tensor:
    return bidirectional_info_nce(cell_z, text_z, temperature_nce) + matching_ce(cell_z, text_z, temperature_ce)


def _positive_mask_from_labels(labels: list[str], device: torch.device) -> torch.Tensor:
    label_arr = np.asarray(labels, dtype=object)
    mask = label_arr[:, None] == label_arr[None, :]
    return torch.tensor(mask, dtype=torch.bool, device=device)


def multi_positive_info_nce(
    cell_z: torch.Tensor,
    text_z: torch.Tensor,
    positive_mask: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    cell_z = F.normalize(cell_z, dim=-1)
    text_z = F.normalize(text_z, dim=-1)
    logits = cell_z @ text_z.T / temperature
    positive_mask = positive_mask.to(device=logits.device, dtype=torch.bool)
    if not bool(positive_mask.any(dim=1).all()) or not bool(positive_mask.any(dim=0).all()):
        raise ValueError("Every row and column must have at least one positive")
    neg_inf = torch.finfo(logits.dtype).min
    row_pos = logits.masked_fill(~positive_mask, neg_inf)
    col_pos = logits.T.masked_fill(~positive_mask.T, neg_inf)
    row_loss = torch.logsumexp(logits, dim=1) - torch.logsumexp(row_pos, dim=1)
    col_loss = torch.logsumexp(logits.T, dim=1) - torch.logsumexp(col_pos, dim=1)
    return 0.5 * (row_loss.mean() + col_loss.mean())


def multi_positive_matching_ce(
    cell_z: torch.Tensor,
    text_z: torch.Tensor,
    positive_mask: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    cell_z = F.normalize(cell_z, dim=-1)
    text_z = F.normalize(text_z, dim=-1)
    logits = cell_z @ text_z.T / temperature
    positive_mask = positive_mask.to(device=logits.device, dtype=logits.dtype)
    targets = positive_mask / positive_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    log_probs = F.log_softmax(logits, dim=1)
    return -(targets * log_probs).sum(dim=1).mean()


def multi_positive_align_loss(
    cell_z: torch.Tensor,
    text_z: torch.Tensor,
    positive_mask: torch.Tensor,
    temperature_nce: float = 0.07,
    temperature_ce: float = 0.07,
) -> torch.Tensor:
    return multi_positive_info_nce(cell_z, text_z, positive_mask, temperature_nce) + multi_positive_matching_ce(
        cell_z, text_z, positive_mask, temperature_ce
    )


def _kl_to_prior(weights: torch.Tensor, prior: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is not None:
        weights = weights.masked_fill(~mask, 0.0)
        prior = prior.masked_fill(~mask, 0.0)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    prior = prior / prior.sum(dim=1, keepdim=True).clamp_min(1e-8)
    kl = weights * (torch.log(weights + 1e-8) - torch.log(prior + 1e-8))
    if mask is not None:
        kl = kl.masked_fill(~mask, 0.0)
    return kl.sum(dim=-1).mean()


def denoise_loss(
    cell_aux: dict[str, torch.Tensor],
    text_aux: dict[str, torch.Tensor],
    cell_mask: torch.Tensor,
    text_mask: torch.Tensor,
    lambda_prior: float = 1.0,
    lambda_entropy: float = 0.01,
    lambda_cell: float = 1.0,
    lambda_text: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cell_kl = _kl_to_prior(cell_aux["weights"], cell_aux["prior"], cell_mask)
    text_kl = _kl_to_prior(text_aux["weights"], text_aux["prior"], text_mask)
    cell_entropy = cell_aux["entropy"].mean()
    text_entropy = text_aux["entropy"].mean()
    cell_loss = lambda_prior * cell_kl + lambda_entropy * cell_entropy
    text_loss = lambda_prior * text_kl + lambda_entropy * text_entropy
    loss = float(lambda_cell) * cell_loss + float(lambda_text) * text_loss
    parts = {
        "loss_cell_kl": cell_kl.detach(),
        "loss_text_kl": text_kl.detach(),
        "loss_cell_denoise": cell_loss.detach(),
        "loss_text_denoise": text_loss.detach(),
        "cell_entropy": cell_entropy.detach(),
        "text_entropy": text_entropy.detach(),
    }
    return loss, parts


def soft_positive_ce_loss(
    cell_z: torch.Tensor,
    text_z: torch.Tensor,
    soft_targets: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Soft positive CE loss for neighbor-based smoothing.

    Unlike regular CE where targets are one-hot (only self-gene is positive),
    this accepts a float target matrix where neighbor genes receive partial
    weight proportional to their cosine similarity. Forces the model to learn
    distributional semantics instead of gene-ID memorisation.

    Args:
        cell_z: (B, D) cell-side representations.
        text_z: (B, D) text-side representations.
        soft_targets: (B, B) float matrix, each row sums to 1. Diagonal holds
            self-weight (typically 0.7-0.85), off-diagonals hold neighbour
            weights.
        temperature: softmax temperature.

    Returns:
        Scalar loss.
    """
    cell_z = F.normalize(cell_z, dim=-1)
    text_z = F.normalize(text_z, dim=-1)
    logits = cell_z @ text_z.T / temperature
    log_probs = F.log_softmax(logits, dim=1)
    targets = soft_targets.to(device=logits.device, dtype=logits.dtype)
    return -(targets * log_probs).sum(dim=1).mean()


def multi_view_consistency_loss(
    cell_view_a: torch.Tensor,
    cell_view_b: torch.Tensor,
    text_view_a: torch.Tensor,
    text_view_b: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Cross-view consistency loss.

    Penalises deviation between two independent views of the same gene distribution.
    Each view is produced by sampling different cell instances from Dg and different
    text evidence from Pg. Minimising this loss forces the encoder to extract
    gene-level stable signal rather than memorising instance-level patterns.

    Returns:
        loss: scalar MSE loss.
        parts: dict with ``loss_multi_view``, ``loss_mv_cell``, ``loss_mv_text``.
    """
    cell_diff = F.mse_loss(cell_view_a, cell_view_b)
    text_diff = F.mse_loss(text_view_a, text_view_b)
    loss = cell_diff + text_diff
    parts = {
        "loss_multi_view": loss.detach(),
        "loss_mv_cell": cell_diff.detach(),
        "loss_mv_text": text_diff.detach(),
    }
    return loss, parts

