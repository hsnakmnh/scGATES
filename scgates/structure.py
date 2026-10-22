from __future__ import annotations

import torch
from torch.nn import functional as F


def _valid_set(
    embeddings: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = embeddings[mask]
    w = weights[mask].to(embeddings.dtype)
    w = w / w.sum().clamp_min(1e-8)
    return x, w


def sinkhorn_cost_torch(
    a: torch.Tensor,
    b: torch.Tensor,
    wa: torch.Tensor,
    wb: torch.Tensor,
    epsilon: float = 0.08,
    max_iter: int = 30,
) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return a.new_tensor(0.0)
    a_norm = F.normalize(a, dim=-1)
    b_norm = F.normalize(b, dim=-1)
    cost = 1.0 - a_norm @ b_norm.T
    cost = cost - cost.min().detach()
    kernel = torch.exp(-cost / max(epsilon, 1e-6)).clamp_min(1e-30)
    u = torch.ones_like(wa)
    v = torch.ones_like(wb)
    for _ in range(max_iter):
        u = wa / (kernel @ v).clamp_min(1e-30)
        v = wb / (kernel.T @ u).clamp_min(1e-30)
    plan = u[:, None] * kernel * v[None, :]
    return torch.sum(plan * cost)


def structure_consistency_loss(
    cell_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    cell_weights: torch.Tensor,
    text_weights: torch.Tensor,
    cell_mask: torch.Tensor,
    text_mask: torch.Tensor,
    epsilon: float = 0.08,
    max_iter: int = 30,
    tau_cell: float = 0.1,
    tau_text: float = 0.1,
    max_pairs: int = 64,
    huber_delta: float = 0.1,
    loss_mode: str = "huber",
    rank_margin: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Structure consistency loss with two modes.

    ``"huber"`` (default, original): penalises |S_D - S_P| per pair.
    ``"rank"`` (T4): triplet ranking loss - only requires order consistency,
    not absolute value alignment. Robust to cross-modal scale differences.

    When using ``"rank"`` with negative ``max_pairs``, all (i,j,k) triples
    are sampled stochastically each call to avoid O(n^3) memory.
    """
    batch_size = int(cell_embeddings.shape[0])
    if loss_mode == "rank":
        return _rank_structure_loss(
            cell_embeddings, text_embeddings,
            cell_weights, text_weights,
            cell_mask, text_mask,
            epsilon=epsilon, max_iter=max_iter,
            tau_cell=tau_cell, tau_text=tau_text,
            max_pairs=max_pairs, rank_margin=rank_margin,
        )

    pairs = [(i, j) for i in range(batch_size) for j in range(i + 1, batch_size)]
    if max_pairs > 0 and len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]
    if not pairs:
        zero = cell_embeddings.new_tensor(0.0)
        return zero, {"loss_struct": zero, "struct_mean_abs_gap": zero, "struct_pairs": zero}

    gaps: list[torch.Tensor] = []
    for i, j in pairs:
        cell_i, wc_i = _valid_set(cell_embeddings[i], cell_weights[i], cell_mask[i])
        cell_j, wc_j = _valid_set(cell_embeddings[j], cell_weights[j], cell_mask[j])
        text_i, wt_i = _valid_set(text_embeddings[i], text_weights[i], text_mask[i])
        text_j, wt_j = _valid_set(text_embeddings[j], text_weights[j], text_mask[j])
        d_cell = sinkhorn_cost_torch(cell_i, cell_j, wc_i, wc_j, epsilon=epsilon, max_iter=max_iter)
        d_text = sinkhorn_cost_torch(text_i, text_j, wt_i, wt_j, epsilon=epsilon, max_iter=max_iter)
        s_cell = torch.exp(-d_cell / max(tau_cell, 1e-6))
        s_text = torch.exp(-d_text / max(tau_text, 1e-6))
        gaps.append(s_cell - s_text)

    gap = torch.stack(gaps)
    loss = F.huber_loss(gap, torch.zeros_like(gap), delta=huber_delta, reduction="mean")
    parts = {
        "loss_struct": loss.detach(),
        "struct_mean_abs_gap": gap.abs().mean().detach(),
        "struct_pairs": cell_embeddings.new_tensor(float(len(pairs))),
    }
    return loss, parts


def _rank_structure_loss(
    cell_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    cell_weights: torch.Tensor,
    text_weights: torch.Tensor,
    cell_mask: torch.Tensor,
    text_mask: torch.Tensor,
    epsilon: float = 0.08,
    max_iter: int = 30,
    tau_cell: float = 0.1,
    tau_text: float = 0.1,
    max_pairs: int = 64,
    rank_margin: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """T4: triplet ranking structure loss.

    For each anchor i, sample j where S_D[i,j] > S_D[i,k] by at least margin,
    then penalise if S_P[i,j] < S_P[i,k] + margin.
    """
    batch_size = int(cell_embeddings.shape[0])
    if batch_size < 3:
        zero = cell_embeddings.new_tensor(0.0)
        return zero, {"loss_struct": zero, "struct_mean_abs_gap": zero, "struct_pairs": zero}

    # 1. Compute all pairwise S_D and S_P
    s_cell = cell_embeddings.new_zeros(batch_size, batch_size)
    s_text = cell_embeddings.new_zeros(batch_size, batch_size)
    for i in range(batch_size):
        cell_i, wc_i = _valid_set(cell_embeddings[i], cell_weights[i], cell_mask[i])
        text_i, wt_i = _valid_set(text_embeddings[i], text_weights[i], text_mask[i])
        for j in range(i + 1, batch_size):
            cell_j, wc_j = _valid_set(cell_embeddings[j], cell_weights[j], cell_mask[j])
            text_j, wt_j = _valid_set(text_embeddings[j], text_weights[j], text_mask[j])
            d_cell = sinkhorn_cost_torch(cell_i, cell_j, wc_i, wc_j, epsilon=epsilon, max_iter=max_iter)
            d_text = sinkhorn_cost_torch(text_i, text_j, wt_i, wt_j, epsilon=epsilon, max_iter=max_iter)
            sc = torch.exp(-d_cell / max(tau_cell, 1e-6))
            st = torch.exp(-d_text / max(tau_text, 1e-6))
            s_cell[i, j] = sc
            s_cell[j, i] = sc
            s_text[i, j] = st
            s_text[j, i] = st

    # 2. Sample triples: anchor i, positive j (closer in cell), negative k (farther)
    num_triplets = max(batch_size * 4, min(abs(max_pairs), batch_size * batch_size // 2)) if max_pairs != 0 else batch_size * 8
    num_triplets = min(num_triplets, batch_size * (batch_size - 1) * (batch_size - 2) // 2)

    margin = max(float(rank_margin), 1e-4)
    losses: list[torch.Tensor] = []
    for _ in range(num_triplets):
        i = torch.randint(0, batch_size, (1,)).item()
        perm = torch.randperm(batch_size, device=cell_embeddings.device)
        perm_others = [int(x) for x in perm.tolist() if x != i]
        j, k = perm_others[:2]
        if s_cell[i, j] < s_cell[i, k] + margin:
            j, k = k, j  # ensure j is the cell-side positive
        pos_cell = s_cell[i, j]
        neg_cell = s_cell[i, k]
        pos_text = s_text[i, j]
        neg_text = s_text[i, k]
        if pos_cell - neg_cell < margin * 0.5:
            continue  # skip ambiguous triples
        # triplet loss: text-side order should match cell-side
        loss_trip = F.relu(neg_text - pos_text + margin)
        losses.append(loss_trip)

    if not losses:
        zero = cell_embeddings.new_tensor(0.0)
        return zero, {"loss_struct": zero, "struct_mean_abs_gap": zero, "struct_pairs": zero}

    loss = torch.stack(losses).mean()
    parts = {
        "loss_struct": loss.detach(),
        "struct_mean_abs_gap": cell_embeddings.new_tensor(0.0),  # not meaningful for rank mode
        "struct_pairs": cell_embeddings.new_tensor(float(len(losses))),
    }
    return loss, parts
