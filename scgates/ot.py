from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


def cosine_cost(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = l2_normalize(a.astype(np.float32))
    b_norm = l2_normalize(b.astype(np.float32))
    return 1.0 - np.matmul(a_norm, b_norm.T)


def sinkhorn_cost(
    a: np.ndarray,
    b: np.ndarray,
    epsilon: float = 0.08,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> float:
    """Compute entropic OT cost with uniform marginals and cosine ground cost."""

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("sinkhorn_cost expects 2D arrays")
    if len(a) == 0 or len(b) == 0:
        return float("inf")

    cost = cosine_cost(a, b).astype(np.float64)
    cost = cost - cost.min()
    kernel = np.exp(-cost / max(epsilon, 1e-6))
    kernel = np.maximum(kernel, 1e-300)

    r = np.full(cost.shape[0], 1.0 / cost.shape[0], dtype=np.float64)
    c = np.full(cost.shape[1], 1.0 / cost.shape[1], dtype=np.float64)
    u = np.ones_like(r)
    v = np.ones_like(c)

    for _ in range(max_iter):
        u_prev = u.copy()
        u = r / np.maximum(kernel @ v, 1e-300)
        v = c / np.maximum(kernel.T @ u, 1e-300)
        if np.max(np.abs(u - u_prev)) < tol:
            break

    plan = (u[:, None] * kernel) * v[None, :]
    return float(np.sum(plan * cost))
