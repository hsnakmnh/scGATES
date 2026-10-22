from __future__ import annotations

import hashlib
import re

import numpy as np


TOKEN_RE = re.compile(r"[A-Za-z0-9_+-]+")


class HashingTextEncoder:
    """Deterministic lightweight text encoder for MVP sanity runs.

    This is intentionally dependency-free. For real runs, replace the produced
    text vectors with BioMedBERT/SapBERT embeddings while keeping the same API.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = TOKEN_RE.findall(text.lower())
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def encode_many(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.encode(text) for text in texts], axis=0)

