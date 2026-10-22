from __future__ import annotations

import numpy as np

from scgates.ot import l2_normalize


def retrieval_metrics(pred: np.ndarray, target: np.ndarray, ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    pred = l2_normalize(pred.astype(np.float32))
    target = l2_normalize(target.astype(np.float32))
    scores = pred @ target.T
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


def retrieval_metrics_by_label(
    pred: np.ndarray,
    target: np.ndarray,
    pred_labels: list[str],
    target_labels: list[str],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    pred = l2_normalize(pred.astype(np.float32))
    target = l2_normalize(target.astype(np.float32))
    scores = pred @ target.T
    order = np.argsort(-scores, axis=1)
    target_labels_arr = np.asarray(target_labels, dtype=object)
    ranks: list[int] = []
    for i, label in enumerate(pred_labels):
        positives = np.flatnonzero(target_labels_arr == label)
        if positives.size == 0:
            continue
        ranked = order[i]
        positive_positions = np.flatnonzero(np.isin(ranked, positives))
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
