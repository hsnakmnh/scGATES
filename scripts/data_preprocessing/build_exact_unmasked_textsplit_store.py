from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


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


def _copy_if_exists(source: Path, output: Path, filename: str) -> None:
    src = source / filename
    if src.exists():
        shutil.copy2(src, output / filename)


def build_exact_store(masked_store: Path, unmasked_source_store: Path, output_store: Path, overwrite: bool) -> dict[str, Any]:
    if output_store.exists():
        if not overwrite:
            raise FileExistsError(f"Output store exists: {output_store}")
        shutil.rmtree(output_store)
    output_store.mkdir(parents=True, exist_ok=True)

    genes = np.load(masked_store / "genes.npy", allow_pickle=True).astype(str)
    contexts = np.load(masked_store / "contexts.npy", allow_pickle=True).astype(str)
    offsets = np.load(masked_store / "offsets.npy").astype(np.int64)
    target_text_ids = np.load(masked_store / "evidence_text_ids.npy", allow_pickle=True).astype(str)

    source_text_ids = np.load(unmasked_source_store / "evidence_text_ids.npy", allow_pickle=True).astype(str)
    source_vectors = np.load(unmasked_source_store / "evidence_vectors.npy")
    source_prior = np.load(unmasked_source_store / "evidence_prior.npy")

    id_to_idx: dict[str, int] = {}
    duplicates: list[str] = []
    for i, text_id in enumerate(source_text_ids):
        if text_id in id_to_idx:
            duplicates.append(text_id)
        id_to_idx[text_id] = i
    if duplicates:
        raise ValueError(f"Duplicate evidence_text_ids in source store, examples: {duplicates[:10]}")

    missing = [text_id for text_id in target_text_ids if text_id not in id_to_idx]
    if missing:
        raise ValueError(f"Missing text ids in unmasked source: n={len(missing)}, examples={missing[:10]}")

    source_indices = np.asarray([id_to_idx[text_id] for text_id in target_text_ids], dtype=np.int64)

    np.save(output_store / "genes.npy", genes)
    np.save(output_store / "contexts.npy", contexts)
    np.save(output_store / "offsets.npy", offsets)
    np.save(output_store / "evidence_text_ids.npy", target_text_ids)
    np.save(output_store / "evidence_vectors.npy", np.asarray(source_vectors[source_indices], dtype=np.float32))
    np.save(output_store / "evidence_prior.npy", np.asarray(source_prior[source_indices], dtype=np.float32))

    features_path = unmasked_source_store / "evidence_metadata_features.npy"
    if features_path.exists():
        features = np.load(features_path)
        np.save(output_store / "evidence_metadata_features.npy", np.asarray(features[source_indices], dtype=np.float32))
    _copy_if_exists(unmasked_source_store, output_store, "evidence_metadata_feature_names.npy")

    source_metadata = _read_metadata(unmasked_source_store, len(source_text_ids))
    byte_offsets: list[int] = []
    with (output_store / "evidence_metadata.jsonl").open("wb") as f:
        for source_idx, text_id in zip(source_indices, target_text_ids):
            byte_offsets.append(f.tell())
            payload = dict(source_metadata[int(source_idx)])
            payload["text_id"] = str(text_id)
            f.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
            f.write(b"\n")
    np.save(output_store / "evidence_metadata_offsets.npy", np.asarray(byte_offsets, dtype=np.int64))

    counts = np.diff(offsets)
    summary = {
        "source_store": str(unmasked_source_store),
        "masked_reference_store": str(masked_store),
        "store": str(output_store),
        "storage": "text_evidence_store",
        "mask_target_gene": False,
        "split_matched_to_masked_reference": True,
        "num_sets": int(len(genes)),
        "num_genes": int(len(set(genes))),
        "num_contexts": int(len(set(contexts))),
        "total_evidence": int(len(target_text_ids)),
        "min_evidence": int(counts.min()) if len(counts) else 0,
        "median_evidence": float(np.median(counts)) if len(counts) else 0.0,
        "max_evidence": int(counts.max()) if len(counts) else 0,
        "vector_dim": int(source_vectors.shape[1]) if source_vectors.ndim == 2 else 0,
        "vector_dtype": str(source_vectors.dtype),
    }
    with (output_store / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def _check_same_ids(masked_store: Path, output_store: Path) -> dict[str, Any]:
    masked_ids = np.load(masked_store / "evidence_text_ids.npy", allow_pickle=True).astype(str)
    output_ids = np.load(output_store / "evidence_text_ids.npy", allow_pickle=True).astype(str)
    return {
        "masked_store": str(masked_store),
        "output_store": str(output_store),
        "masked_len": int(len(masked_ids)),
        "output_len": int(len(output_ids)),
        "same_order": bool(len(masked_ids) == len(output_ids) and (masked_ids == output_ids).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build an exact unmasked text evidence store by matching the evidence_text_ids "
            "and order of an existing masked split store."
        )
    )
    parser.add_argument("--unmasked-source-store", required=True)
    parser.add_argument("--masked-train-store", required=True)
    parser.add_argument("--masked-eval-store", required=True)
    parser.add_argument("--train-output-store", required=True)
    parser.add_argument("--eval-output-store", required=True)
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = Path(args.unmasked_source_store)
    masked_train = Path(args.masked_train_store)
    masked_eval = Path(args.masked_eval_store)
    train_output = Path(args.train_output_store)
    eval_output = Path(args.eval_output_store)

    train_summary = build_exact_store(masked_train, source, train_output, overwrite=args.overwrite)
    eval_summary = build_exact_store(masked_eval, source, eval_output, overwrite=args.overwrite)
    checks = [
        _check_same_ids(masked_train, train_output),
        _check_same_ids(masked_eval, eval_output),
    ]

    summary = {
        "unmasked_source_store": str(source),
        "train_summary": train_summary,
        "eval_summary": eval_summary,
        "checks": checks,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    failed = [check for check in checks if not check["same_order"]]
    if failed:
        raise SystemExit(f"Output text ids do not match masked reference: {failed}")


if __name__ == "__main__":
    main()

