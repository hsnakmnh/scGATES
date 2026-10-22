from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import itertools
from pathlib import Path
from typing import Any

import numpy as np


def _stable_rng(seed: int, gene: str) -> np.random.RandomState:
    digest = hashlib.sha256(f"{seed}:{gene}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:4], "little", signed=False)
    return np.random.RandomState(value)


def _read_metadata(store: Path, total: int) -> list[dict[str, Any]]:
    offsets_path = store / "evidence_metadata_offsets.npy"
    jsonl_path = store / "evidence_metadata.jsonl"
    if not offsets_path.exists() or not jsonl_path.exists():
        return [{"text_id": ""} for _ in range(total)]
    offsets = np.load(offsets_path).astype(np.int64)
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("rb") as f:
        for i in range(total):
            f.seek(int(offsets[i]))
            line = f.readline().decode("utf-8").strip()
            rows.append(json.loads(line) if line else {"text_id": ""})
    return rows


def _group_indices(
    local_indices: list[int],
    metadata: list[dict[str, Any]],
    group_by: str,
) -> list[list[int]]:
    if group_by == "text_id":
        return [[idx] for idx in local_indices]
    groups: dict[str, list[int]] = {}
    for idx in local_indices:
        meta = metadata[idx]
        if group_by == "pmid":
            key = str(meta.get("pmid") or "").strip()
            if not key or key.lower() in {"nan", "none"}:
                key = str(meta.get("text_id") or idx)
        else:
            key = str(meta.get("text_id") or idx)
        groups.setdefault(key, []).append(idx)
    return list(groups.values())


def _float_meta(meta: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = meta.get(key, default)
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _group_features(group: list[int], metadata: list[dict[str, Any]], prior: np.ndarray) -> np.ndarray:
    feats: list[list[float]] = []
    for idx in group:
        meta = metadata[idx]
        feats.append(
            [
                float(prior[idx]),
                _float_meta(meta, "evidence_strength"),
                _float_meta(meta, "ad_specific"),
                _float_meta(meta, "has_pmid"),
                _float_meta(meta, "recency_score"),
                _float_meta(meta, "sentence_len_norm"),
            ]
        )
    if not feats:
        return np.zeros(6, dtype=np.float64)
    return np.asarray(feats, dtype=np.float64).mean(axis=0)


def _split_objective(
    train_groups: list[list[int]],
    eval_groups: list[list[int]],
    feature_by_idx: dict[int, np.ndarray],
) -> float:
    train_idx = [i for group in train_groups for i in group]
    eval_idx = [i for group in eval_groups for i in group]
    if not train_idx or not eval_idx:
        return float("inf")

    # The first term balances train/eval means; the second keeps eval close to
    # the whole gene's evidence profile, important when eval has only one item.
    train_features = np.asarray([feature_by_idx[idx] for idx in train_idx], dtype=np.float64)
    eval_features = np.asarray([feature_by_idx[idx] for idx in eval_idx], dtype=np.float64)
    all_features = np.asarray([feature_by_idx[idx] for idx in train_idx + eval_idx], dtype=np.float64)
    train_mean = np.mean(train_features, axis=0)
    eval_mean = np.mean(eval_features, axis=0)
    global_mean = np.mean(all_features, axis=0)
    scale = np.std(all_features, axis=0) + 1e-6
    train_eval_gap = np.mean(np.abs((train_mean - eval_mean) / scale))
    eval_global_gap = np.mean(np.abs((eval_mean - global_mean) / scale))
    return float(train_eval_gap + 0.5 * eval_global_gap)


def _select_balanced_groups(
    gene: str,
    groups: list[list[int]],
    metadata: list[dict[str, Any]],
    prior: np.ndarray,
    target_eval: int,
    min_train: int,
    min_eval: int,
    seed: int,
    max_exact_groups: int = 18,
    random_trials: int = 512,
) -> tuple[list[int], list[int]] | None:
    total_n = sum(len(group) for group in groups)
    if target_eval <= 0 or total_n - target_eval < min_train:
        return None

    all_indices = [idx for group in groups for idx in group]
    feature_by_idx = {
        idx: np.asarray(
            [
                float(prior[idx]),
                _float_meta(metadata[idx], "evidence_strength"),
                _float_meta(metadata[idx], "ad_specific"),
                _float_meta(metadata[idx], "has_pmid"),
                _float_meta(metadata[idx], "recency_score"),
                _float_meta(metadata[idx], "sentence_len_norm"),
            ],
            dtype=np.float64,
        )
        for idx in all_indices
    }

    best_eval_groups: list[list[int]] | None = None
    best_score = float("inf")

    def consider(eval_group_ids: list[int]) -> None:
        nonlocal best_eval_groups, best_score
        eval_groups = [groups[i] for i in eval_group_ids]
        eval_count = sum(len(group) for group in eval_groups)
        train_count = total_n - eval_count
        if eval_count < min_eval or train_count < min_train:
            return
        if abs(eval_count - target_eval) > max(1, int(round(0.1 * total_n))):
            return
        eval_set = set(eval_group_ids)
        train_groups = [group for i, group in enumerate(groups) if i not in eval_set]
        score = _split_objective(train_groups, eval_groups, feature_by_idx)
        # A tiny deterministic jitter breaks ties without favoring early groups.
        score += _stable_rng(seed, f"{gene}:{','.join(map(str, sorted(eval_group_ids)))}").rand() * 1e-9
        if score < best_score:
            best_score = score
            best_eval_groups = eval_groups

    if len(groups) <= max_exact_groups:
        group_ids = list(range(len(groups)))
        for r in range(1, len(groups)):
            for combo in itertools.combinations(group_ids, r):
                eval_count = sum(len(groups[i]) for i in combo)
                if eval_count > target_eval + max(1, int(round(0.1 * total_n))):
                    continue
                consider(list(combo))
    else:
        rng = _stable_rng(seed, gene)
        group_ids = list(range(len(groups)))
        for _ in range(random_trials):
            rng.shuffle(group_ids)
            eval_ids: list[int] = []
            eval_count = 0
            for group_id in group_ids:
                if eval_count < target_eval:
                    eval_ids.append(group_id)
                    eval_count += len(groups[group_id])
            consider(eval_ids)

    if best_eval_groups is None:
        return None
    eval_indices = sorted(idx for group in best_eval_groups for idx in group)
    eval_set = set(eval_indices)
    train_indices = sorted(idx for group in groups for idx in group if idx not in eval_set)
    return train_indices, eval_indices


def _split_gene_indices(
    gene: str,
    local_indices: list[int],
    metadata: list[dict[str, Any]],
    eval_fraction: float,
    min_train: int,
    min_eval: int,
    seed: int,
    group_by: str,
    unsplittable_policy: str,
    split_strategy: str,
    prior: np.ndarray,
) -> tuple[list[int], list[int], str]:
    n = len(local_indices)
    if n < min_train + min_eval:
        if unsplittable_policy == "duplicate":
            return list(local_indices), list(local_indices), "duplicated_too_few"
        return list(local_indices), [], "train_only_too_few"

    groups = _group_indices(local_indices, metadata, group_by=group_by)
    if len(groups) < 2:
        if unsplittable_policy == "duplicate":
            return list(local_indices), list(local_indices), "duplicated_one_group"
        return list(local_indices), [], "train_only_one_group"

    rng = _stable_rng(seed, gene)
    order = np.arange(len(groups))
    rng.shuffle(order)
    shuffled = [groups[int(i)] for i in order]

    target_eval = max(min_eval, int(round(n * eval_fraction)))
    target_eval = min(target_eval, n - min_train)
    if split_strategy == "balanced":
        balanced = _select_balanced_groups(
            gene,
            groups,
            metadata,
            prior,
            target_eval=target_eval,
            min_train=min_train,
            min_eval=min_eval,
            seed=seed,
        )
        if balanced is not None:
            return balanced[0], balanced[1], "split_balanced"

    eval_indices: list[int] = []
    train_indices: list[int] = []
    for group in shuffled:
        if len(eval_indices) < target_eval and len(train_indices) + (n - len(eval_indices) - len(group)) >= min_train:
            eval_indices.extend(group)
        else:
            train_indices.extend(group)

    if len(eval_indices) < min_eval or len(train_indices) < min_train:
        if unsplittable_policy == "duplicate":
            return list(local_indices), list(local_indices), "duplicated_constraint"
        return list(local_indices), [], "train_only_constraint"
    return sorted(train_indices), sorted(eval_indices), "split"


def _write_store(
    output: Path,
    source_store: Path,
    selected_by_gene: list[tuple[int, list[int]]],
    genes: np.ndarray,
    contexts: np.ndarray,
    vectors: np.ndarray,
    text_ids: np.ndarray,
    prior: np.ndarray,
    metadata: list[dict[str, Any]],
    metadata_features: np.ndarray | None,
    overwrite: bool,
) -> dict[str, Any]:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output store exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    out_genes: list[str] = []
    out_contexts: list[str] = []
    out_offsets = [0]
    flat_indices: list[int] = []
    for gene_idx, evidence_indices in selected_by_gene:
        if not evidence_indices:
            continue
        out_genes.append(str(genes[gene_idx]))
        out_contexts.append(str(contexts[gene_idx]))
        flat_indices.extend(evidence_indices)
        out_offsets.append(len(flat_indices))

    flat = np.asarray(flat_indices, dtype=np.int64)
    np.save(output / "genes.npy", np.asarray(out_genes, dtype=str))
    np.save(output / "contexts.npy", np.asarray(out_contexts, dtype=str))
    np.save(output / "offsets.npy", np.asarray(out_offsets, dtype=np.int64))
    np.save(output / "evidence_vectors.npy", np.asarray(vectors[flat], dtype=np.float32))
    np.save(output / "evidence_text_ids.npy", np.asarray(text_ids[flat]).astype(str))
    np.save(output / "evidence_prior.npy", np.asarray(prior[flat], dtype=np.float32))
    if metadata_features is not None:
        np.save(output / "evidence_metadata_features.npy", np.asarray(metadata_features[flat], dtype=np.float32))
        names_path = source_store / "evidence_metadata_feature_names.npy"
        if names_path.exists():
            np.save(output / "evidence_metadata_feature_names.npy", np.load(names_path, allow_pickle=True).astype(str))

    byte_offsets: list[int] = []
    with (output / "evidence_metadata.jsonl").open("wb") as f:
        for source_idx in flat:
            byte_offsets.append(f.tell())
            payload = dict(metadata[int(source_idx)])
            if "text_id" not in payload:
                payload["text_id"] = str(text_ids[int(source_idx)])
            f.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
            f.write(b"\n")
    np.save(output / "evidence_metadata_offsets.npy", np.asarray(byte_offsets, dtype=np.int64))

    counts = np.diff(np.asarray(out_offsets, dtype=np.int64))
    summary = {
        "source_store": str(source_store),
        "store": str(output),
        "storage": "text_evidence_store",
        "num_sets": int(len(out_genes)),
        "num_genes": int(len(set(out_genes))),
        "num_contexts": int(len(set(out_contexts))),
        "total_evidence": int(len(flat)),
        "min_evidence": int(counts.min()) if len(counts) else 0,
        "median_evidence": float(np.median(counts)) if len(counts) else 0.0,
        "max_evidence": int(counts.max()) if len(counts) else 0,
        "vector_dim": int(vectors.shape[1]) if vectors.ndim == 2 else 0,
        "vector_dtype": str(vectors.dtype),
    }
    with (output / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministically split a text evidence store into train/eval evidence stores per gene."
    )
    parser.add_argument("--input-store", required=True)
    parser.add_argument("--train-output-store", required=True)
    parser.add_argument("--eval-output-store", required=True)
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--min-train-evidence", type=int, default=1)
    parser.add_argument("--min-eval-evidence", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-by", choices=["text_id", "pmid"], default="text_id")
    parser.add_argument(
        "--split-strategy",
        choices=["balanced", "random"],
        default="balanced",
        help=(
            "balanced selects held-out evidence to match train/eval evidence strength, AD specificity, "
            "recency, PMID availability, sentence length, and evidence prior as closely as possible."
        ),
    )
    parser.add_argument(
        "--unsplittable-policy",
        choices=["duplicate", "train_only"],
        default="duplicate",
        help=(
            "How to handle genes whose evidence cannot be split while satisfying min train/eval constraints. "
            "'duplicate' keeps the gene in both train/eval stores using the same evidence and records this in the summary."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = Path(args.input_store)
    genes = np.load(source / "genes.npy", allow_pickle=True).astype(str)
    contexts = np.load(source / "contexts.npy", allow_pickle=True).astype(str)
    offsets = np.load(source / "offsets.npy").astype(np.int64)
    vectors = np.load(source / "evidence_vectors.npy", mmap_mode="r")
    text_ids = np.load(source / "evidence_text_ids.npy", allow_pickle=True).astype(str)
    prior = (
        np.load(source / "evidence_prior.npy", mmap_mode="r")
        if (source / "evidence_prior.npy").exists()
        else np.ones(vectors.shape[0], dtype=np.float32)
    )
    metadata_features = (
        np.load(source / "evidence_metadata_features.npy", mmap_mode="r")
        if (source / "evidence_metadata_features.npy").exists()
        else None
    )
    metadata = _read_metadata(source, total=int(vectors.shape[0]))

    train_by_gene: list[tuple[int, list[int]]] = []
    eval_by_gene: list[tuple[int, list[int]]] = []
    status_counts: dict[str, int] = {}
    per_gene_rows: list[dict[str, Any]] = []
    for gene_idx, gene in enumerate(genes):
        start = int(offsets[gene_idx])
        end = int(offsets[gene_idx + 1])
        local = list(range(start, end))
        train_idx, eval_idx, status = _split_gene_indices(
            str(gene),
            local,
            metadata,
            eval_fraction=args.eval_fraction,
            min_train=args.min_train_evidence,
            min_eval=args.min_eval_evidence,
            seed=args.seed,
            group_by=args.group_by,
            unsplittable_policy=args.unsplittable_policy,
            split_strategy=args.split_strategy,
            prior=prior,
        )
        train_by_gene.append((gene_idx, train_idx))
        eval_by_gene.append((gene_idx, eval_idx))
        status_counts[status] = status_counts.get(status, 0) + 1
        per_gene_rows.append(
            {
                "gene": str(gene),
                "context": str(contexts[gene_idx]),
                "total_evidence": len(local),
                "train_evidence": len(train_idx),
                "eval_evidence": len(eval_idx),
                "status": status,
            }
        )

    train_summary = _write_store(
        Path(args.train_output_store),
        source,
        train_by_gene,
        genes,
        contexts,
        vectors,
        text_ids,
        prior,
        metadata,
        metadata_features,
        overwrite=args.overwrite,
    )
    eval_summary = _write_store(
        Path(args.eval_output_store),
        source,
        eval_by_gene,
        genes,
        contexts,
        vectors,
        text_ids,
        prior,
        metadata,
        metadata_features,
        overwrite=args.overwrite,
    )

    summary = {
        "source_store": str(source),
        "train_output_store": args.train_output_store,
        "eval_output_store": args.eval_output_store,
        "eval_fraction": args.eval_fraction,
        "min_train_evidence": args.min_train_evidence,
        "min_eval_evidence": args.min_eval_evidence,
        "seed": args.seed,
        "group_by": args.group_by,
        "split_strategy": args.split_strategy,
        "unsplittable_policy": args.unsplittable_policy,
        "leakage_note": (
            "Genes marked duplicated_* use the same evidence in train and eval because they cannot be strictly split. "
            "Report duplicated_* counts separately when interpreting held-out text results."
            if args.unsplittable_policy == "duplicate"
            else "Unsplittable genes are absent from eval text store."
        ),
        "status_counts": status_counts,
        "train_summary": train_summary,
        "eval_summary": eval_summary,
    }
    if args.summary_json:
        path = Path(args.summary_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump({**summary, "per_gene": per_gene_rows}, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

