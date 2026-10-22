from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class GeneDistribution:
    key: str
    gene: str
    context: str
    embeddings: np.ndarray
    ranks: np.ndarray
    expr: np.ndarray
    occurrence_rate: float = 0.0
    instance_scores: np.ndarray | None = None


@dataclass(frozen=True)
class TextEvidenceSet:
    gene: str
    context: str
    text_ids: list[str]
    embeddings: np.ndarray
    prior: np.ndarray
    metadata: list[dict[str, str]]
    metadata_features: np.ndarray | None = None


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_text_records(path: str | Path) -> list[dict[str, str]]:
    records = read_jsonl(path)
    normalized: list[dict[str, str]] = []
    for idx, record in enumerate(records):
        gene = str(record.get("gene", "")).strip()
        text = str(record.get("text", "")).strip()
        context = str(record.get("context", "global")).strip() or "global"
        if not gene or not text:
            raise ValueError(f"Bad text record at line {idx + 1}: expected gene and text")
        normalized.append(
            {
                "gene": gene,
                "context": context,
                "text": text,
                "text_id": str(record.get("text_id", f"{gene}|{context}")),
            }
        )
    return normalized


def load_text_vectors_npz(path: str | Path) -> tuple[list[dict[str, str]], np.ndarray]:
    data = np.load(path, allow_pickle=True)
    vectors = data["vectors"].astype(np.float32)
    genes = data["genes"].astype(str)
    contexts = data["contexts"].astype(str)
    text_ids = data["text_ids"].astype(str)
    records = [
        {
            "gene": str(gene),
            "context": str(context),
            "text_id": str(text_id),
            "text": "",
        }
        for gene, context, text_id in zip(genes, contexts, text_ids)
    ]
    return records, vectors


def load_text_evidence_genes_npz(path: str | Path) -> set[str]:
    data = np.load(path, allow_pickle=True)
    return set(data["genes"].astype(str).tolist())


def load_text_evidence_genes_store(store_dir: str | Path) -> set[str]:
    store = Path(store_dir)
    return set(np.load(store / "genes.npy", allow_pickle=True).astype(str).tolist())


def load_text_evidence_sets_npz(path: str | Path, gene_allowlist: set[str] | None = None) -> list[TextEvidenceSet]:
    data = np.load(path, allow_pickle=True)
    genes = data["genes"].astype(str)
    contexts = data["contexts"].astype(str) if "contexts" in data else np.asarray(["global"] * len(genes))
    selected_gene_indices = [
        i for i, gene in enumerate(genes) if gene_allowlist is None or str(gene) in gene_allowlist
    ]

    if "evidence_vectors" in data:
        flat_vectors = data["evidence_vectors"].astype(np.float32)
        offsets = data["evidence_offsets"].astype(np.int64)
    elif "vectors" in data and "evidence_offsets" in data and int(data["evidence_offsets"][-1]) == data["vectors"].shape[0]:
        flat_vectors = data["vectors"].astype(np.float32)
        offsets = data["evidence_offsets"].astype(np.int64)
    else:
        vectors = data["vectors"].astype(np.float32)
        flat_vectors = vectors
        offsets = np.arange(len(vectors) + 1, dtype=np.int64)

    if "evidence_text_ids" in data:
        flat_text_ids = data["evidence_text_ids"].astype(str)
    elif "text_ids" in data and len(data["text_ids"]) == flat_vectors.shape[0]:
        flat_text_ids = data["text_ids"].astype(str)
    else:
        flat_text_ids = np.asarray(
            [
                f"{genes[i]}|evidence_{j}"
                for i in range(len(genes))
                for j in range(int(offsets[i + 1] - offsets[i]))
            ],
            dtype=object,
        ).astype(str)

    if "evidence_prior" in data:
        flat_prior = data["evidence_prior"].astype(np.float32)
    else:
        flat_prior = np.ones(flat_vectors.shape[0], dtype=np.float32)
    flat_metadata_features = None
    if "evidence_metadata_features" in data:
        flat_metadata_features = data["evidence_metadata_features"].astype(np.float32)

    metadata_by_id: dict[str, dict[str, str]] = {}
    if "evidence_metadata_json" in data:
        flat_metadata_json = data["evidence_metadata_json"].astype(str)
        if len(flat_metadata_json) == flat_vectors.shape[0]:
            metadata_items: list[str] = []
            for i in selected_gene_indices:
                start = int(offsets[i])
                end = int(offsets[i + 1])
                metadata_items.extend(str(item) for item in flat_metadata_json[start:end])
        else:
            selected_evidence_ids: set[str] = set()
            for i in selected_gene_indices:
                start = int(offsets[i])
                end = int(offsets[i + 1])
                selected_evidence_ids.update(str(x) for x in flat_text_ids[start:end])
            metadata_items = []
            for item in flat_metadata_json:
                payload = json.loads(str(item))
                text_id = str(payload.get("text_id", ""))
                if text_id in selected_evidence_ids:
                    metadata_by_id[text_id] = {str(k): str(v) for k, v in payload.items()}
            metadata_items = []
        for item in metadata_items:
            payload = json.loads(str(item))
            text_id = str(payload.get("text_id", ""))
            metadata_by_id[text_id] = {str(k): str(v) for k, v in payload.items()}

    sets: list[TextEvidenceSet] = []
    for i in selected_gene_indices:
        gene = genes[i]
        start = int(offsets[i])
        end = int(offsets[i + 1])
        if end <= start:
            continue
        prior = flat_prior[start:end].astype(np.float32)
        prior = prior / max(float(prior.sum()), 1e-8)
        text_ids = [str(x) for x in flat_text_ids[start:end]]
        metadata = [metadata_by_id.get(text_id, {"text_id": text_id}) for text_id in text_ids]
        metadata_features = None
        if flat_metadata_features is not None:
            metadata_features = flat_metadata_features[start:end].astype(np.float32)
        sets.append(
            TextEvidenceSet(
                gene=str(gene),
                context=str(contexts[i]),
                text_ids=text_ids,
                embeddings=flat_vectors[start:end].astype(np.float32),
                prior=prior,
                metadata=metadata,
                metadata_features=metadata_features,
            )
        )
    return sets


def load_text_evidence_sets_store(
    store_dir: str | Path,
    gene_allowlist: set[str] | None = None,
) -> list[TextEvidenceSet]:
    store = Path(store_dir)
    genes = np.load(store / "genes.npy", allow_pickle=True).astype(str)
    contexts = np.load(store / "contexts.npy", allow_pickle=True).astype(str)
    offsets = np.load(store / "offsets.npy").astype(np.int64)
    flat_vectors = np.load(store / "evidence_vectors.npy", mmap_mode="r")
    flat_text_ids = np.load(store / "evidence_text_ids.npy", mmap_mode="r")
    flat_prior = (
        np.load(store / "evidence_prior.npy", mmap_mode="r")
        if (store / "evidence_prior.npy").exists()
        else np.ones(flat_vectors.shape[0], dtype=np.float32)
    )
    flat_metadata_features = (
        np.load(store / "evidence_metadata_features.npy", mmap_mode="r")
        if (store / "evidence_metadata_features.npy").exists()
        else None
    )
    metadata_offsets = (
        np.load(store / "evidence_metadata_offsets.npy").astype(np.int64)
        if (store / "evidence_metadata_offsets.npy").exists()
        else None
    )
    metadata_jsonl = store / "evidence_metadata.jsonl"

    selected_gene_indices = [
        i for i, gene in enumerate(genes) if gene_allowlist is None or str(gene) in gene_allowlist
    ]

    sets: list[TextEvidenceSet] = []
    metadata_handle = metadata_jsonl.open("rb") if metadata_offsets is not None and metadata_jsonl.exists() else None
    try:
        for i in selected_gene_indices:
            gene = str(genes[i])
            start = int(offsets[i])
            end = int(offsets[i + 1])
            if end <= start:
                continue
            prior = np.asarray(flat_prior[start:end], dtype=np.float32)
            prior = prior / max(float(prior.sum()), 1e-8)
            text_ids = [str(x) for x in np.asarray(flat_text_ids[start:end])]
            metadata: list[dict[str, str]] = []
            if metadata_handle is not None and metadata_offsets is not None:
                for evidence_idx, text_id in zip(range(start, end), text_ids):
                    metadata_handle.seek(int(metadata_offsets[evidence_idx]))
                    line = metadata_handle.readline().decode("utf-8").strip()
                    metadata.append({str(k): str(v) for k, v in json.loads(line).items()} if line else {"text_id": text_id})
            else:
                metadata = [{"text_id": text_id} for text_id in text_ids]
            metadata_features = None
            if flat_metadata_features is not None:
                metadata_features = np.asarray(flat_metadata_features[start:end], dtype=np.float32)
            sets.append(
                TextEvidenceSet(
                    gene=gene,
                    context=str(contexts[i]),
                    text_ids=text_ids,
                    embeddings=np.asarray(flat_vectors[start:end], dtype=np.float32),
                    prior=prior,
                    metadata=metadata,
                    metadata_features=metadata_features,
                )
            )
    finally:
        if metadata_handle is not None:
            metadata_handle.close()
    return sets


def load_distributions_npz(path: str | Path) -> list[GeneDistribution]:
    data = np.load(path, allow_pickle=True)
    keys = data["keys"].astype(str)
    genes = data["genes"].astype(str)
    contexts = data["contexts"].astype(str)
    offsets = data["offsets"].astype(np.int64)
    embeddings = data["embeddings"].astype(np.float32)
    ranks = data["ranks"].astype(np.float32)
    expr = data["expr"].astype(np.float32)
    occurrence_rates = (
        data["occurrence_rates"].astype(np.float32)
        if "occurrence_rates" in data
        else np.zeros(len(keys), dtype=np.float32)
    )

    distributions: list[GeneDistribution] = []
    for i, key in enumerate(keys):
        start = int(offsets[i])
        end = int(offsets[i + 1])
        distributions.append(
            GeneDistribution(
                key=str(key),
                gene=str(genes[i]),
                context=str(contexts[i]),
                embeddings=embeddings[start:end],
                ranks=ranks[start:end],
                expr=expr[start:end],
                occurrence_rate=float(occurrence_rates[i]),
            )
        )
    return distributions


def load_distribution_store_metadata(store_dir: str | Path) -> dict[str, Any]:
    metadata_path = Path(store_dir) / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Distribution store metadata not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_distributions_store(
    store_dir: str | Path,
    gene_allowlist: set[str] | None = None,
    context_filter: str = "",
    max_distributions: int = 0,
) -> list[GeneDistribution]:
    store = Path(store_dir)
    genes = np.load(store / "genes.npy", allow_pickle=True).astype(str)
    contexts = np.load(store / "contexts.npy", allow_pickle=True).astype(str)
    offsets = np.load(store / "offsets.npy").astype(np.int64)
    embeddings = np.load(store / "embeddings.npy", mmap_mode="r")
    ranks = np.load(store / "ranks.npy", mmap_mode="r")
    expr = np.load(store / "expr.npy", mmap_mode="r")
    occurrence_rates = (
        np.load(store / "occurrence_rates.npy").astype(np.float32)
        if (store / "occurrence_rates.npy").exists()
        else np.zeros(len(genes), dtype=np.float32)
    )

    distributions: list[GeneDistribution] = []
    for i, (gene, context) in enumerate(zip(genes, contexts)):
        gene = str(gene)
        context = str(context)
        if gene_allowlist is not None and gene not in gene_allowlist:
            continue
        if context_filter and context != context_filter:
            continue
        start = int(offsets[i])
        end = int(offsets[i + 1])
        distributions.append(
            GeneDistribution(
                key=f"{gene}|{context}",
                gene=gene,
                context=context,
                embeddings=embeddings[start:end],
                ranks=ranks[start:end],
                expr=expr[start:end],
                occurrence_rate=float(occurrence_rates[i]),
            )
        )
        if max_distributions and len(distributions) >= max_distributions:
            break
    return distributions


def load_instance_score_npz(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    keys = data["keys"].astype(str)
    offsets = data["offsets"].astype(np.int64)
    scores = data["scores"].astype(np.float32)
    out: dict[str, np.ndarray] = {}
    for i, key in enumerate(keys):
        start = int(offsets[i])
        end = int(offsets[i + 1])
        out[str(key)] = scores[start:end].astype(np.float32)
    return out


def attach_instance_scores(
    distributions: list[GeneDistribution],
    score_map: dict[str, np.ndarray],
    strict: bool = False,
) -> tuple[list[GeneDistribution], dict[str, Any]]:
    attached: list[GeneDistribution] = []
    matched = 0
    mismatched: list[str] = []
    for dist in distributions:
        scores = score_map.get(dist.key)
        if scores is not None:
            scores = np.asarray(scores, dtype=np.float32).reshape(-1)
            if scores.shape[0] != dist.embeddings.shape[0]:
                mismatched.append(dist.key)
                if strict:
                    raise ValueError(
                        f"Instance score length mismatch for {dist.key}: "
                        f"{scores.shape[0]} != {dist.embeddings.shape[0]}"
                    )
                scores = None
            else:
                matched += 1
        attached.append(
            GeneDistribution(
                key=dist.key,
                gene=dist.gene,
                context=dist.context,
                embeddings=dist.embeddings,
                ranks=dist.ranks,
                expr=dist.expr,
                occurrence_rate=dist.occurrence_rate,
                instance_scores=scores,
            )
        )
    return attached, {
        "num_distributions": len(distributions),
        "num_score_keys": len(score_map),
        "matched_distributions": matched,
        "coverage": matched / max(len(distributions), 1),
        "mismatched_distributions": mismatched[:20],
        "num_mismatched_distributions": len(mismatched),
    }


def save_distributions_npz(path: str | Path, distributions: list[GeneDistribution]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not distributions:
        raise ValueError("No gene distributions to save")

    keys: list[str] = []
    genes: list[str] = []
    contexts: list[str] = []
    offsets = [0]
    emb_parts: list[np.ndarray] = []
    rank_parts: list[np.ndarray] = []
    expr_parts: list[np.ndarray] = []
    occurrence_rates: list[float] = []

    for dist in distributions:
        if dist.embeddings.ndim != 2:
            raise ValueError(f"{dist.key} embeddings must be 2D")
        n = dist.embeddings.shape[0]
        if n == 0:
            continue
        keys.append(dist.key)
        genes.append(dist.gene)
        contexts.append(dist.context)
        emb_parts.append(dist.embeddings.astype(np.float32))
        rank_parts.append(dist.ranks.astype(np.float32).reshape(n))
        expr_parts.append(dist.expr.astype(np.float32).reshape(n))
        occurrence_rates.append(float(dist.occurrence_rate))
        offsets.append(offsets[-1] + n)

    np.savez_compressed(
        path,
        keys=np.asarray(keys, dtype=object),
        genes=np.asarray(genes, dtype=object),
        contexts=np.asarray(contexts, dtype=object),
        offsets=np.asarray(offsets, dtype=np.int64),
        embeddings=np.concatenate(emb_parts, axis=0).astype(np.float32),
        ranks=np.concatenate(rank_parts, axis=0).astype(np.float32),
        expr=np.concatenate(expr_parts, axis=0).astype(np.float32),
        occurrence_rates=np.asarray(occurrence_rates, dtype=np.float32),
    )


def summarize_distributions(distributions: list[GeneDistribution]) -> dict[str, Any]:
    counts = np.asarray([d.embeddings.shape[0] for d in distributions], dtype=np.int64)
    dims = {int(d.embeddings.shape[1]) for d in distributions if d.embeddings.size}
    return {
        "num_distributions": len(distributions),
        "num_genes": len({d.gene for d in distributions}),
        "num_contexts": len({d.context for d in distributions}),
        "embedding_dims": sorted(dims),
        "min_instances": int(counts.min()) if len(counts) else 0,
        "median_instances": float(np.median(counts)) if len(counts) else 0.0,
        "max_instances": int(counts.max()) if len(counts) else 0,
    }

