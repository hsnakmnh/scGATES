from __future__ import annotations

import numpy as np


def normalize_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    lo = float(np.min(x))
    hi = float(np.max(x))
    if np.isclose(lo, hi):
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo)


def stratified_instance_indices(
    ranks: np.ndarray,
    expr: np.ndarray,
    max_instances: int,
    num_strata: int = 4,
    seed: int | None = None,
    mode: str = "priority",
) -> np.ndarray:
    """Select Dg instances across rank strata instead of keeping only leading rows.

    Parameters
    ----------
    mode : str
        ``"priority"`` (default): select highest-priority instances within each stratum.
        ``"random"``: randomly select instances within each stratum, ignoring priority.
        Used for multi-view consistency: view A uses ``"priority"``,
        view B uses ``"random"`` to produce meaningfully different cell subsets.
    """
    if mode not in ("priority", "random"):
        raise ValueError(f"stratified_instance_indices mode must be 'priority' or 'random', got {mode!r}")

    total = int(np.asarray(ranks).shape[0])
    if max_instances <= 0 or total <= max_instances:
        return np.arange(total, dtype=np.int64)

    ranks = np.asarray(ranks).reshape(-1)
    expr = np.asarray(expr).reshape(-1)
    rank_norm = normalize_1d(ranks.astype(np.float32))
    expr_norm = normalize_1d(expr.astype(np.float32))
    priority = 0.65 * (1.0 - rank_norm) + 0.35 * expr_norm

    sorted_idx = np.argsort(ranks, kind="mergesort")
    strata = [s for s in np.array_split(sorted_idx, min(num_strata, total)) if s.size]
    base_quota = max_instances // len(strata)
    remainder = max_instances % len(strata)

    rng = np.random.RandomState(seed) if seed is not None else None

    selected: list[int] = []
    used: set[int] = set()
    for stratum_id, stratum in enumerate(strata):
        quota = base_quota + (1 if stratum_id < remainder else 0)
        if quota <= 0:
            continue
        if mode == "random" and rng is not None:
            ordered = stratum.copy()
            rng.shuffle(ordered)
        else:
            ordered = stratum[np.argsort(-priority[stratum], kind="mergesort")]
        for idx in ordered[:quota]:
            idx_int = int(idx)
            selected.append(idx_int)
            used.add(idx_int)

    if len(selected) < max_instances:
        remaining = [int(idx) for idx in np.argsort(-priority, kind="mergesort") if idx not in used]
        if mode == "random" and rng is not None and remaining:
            rng.shuffle(remaining)
        for idx_int in remaining:
            selected.append(idx_int)
            if len(selected) >= max_instances:
                break

    return np.asarray(selected[:max_instances], dtype=np.int64)

