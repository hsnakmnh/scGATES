from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scgates.io import write_json


def _log(message: str) -> None:
    print(f"[text-store] {message}", flush=True)


def _flat_arrays(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    if "evidence_vectors" in data:
        return data["evidence_vectors"], data["evidence_offsets"].astype(np.int64)
    if "vectors" in data and "evidence_offsets" in data and int(data["evidence_offsets"][-1]) == data["vectors"].shape[0]:
        return data["vectors"], data["evidence_offsets"].astype(np.int64)
    vectors = data["vectors"]
    return vectors, np.arange(vectors.shape[0] + 1, dtype=np.int64)


def export_store(input_npz: str | Path, output_store: str | Path, overwrite: bool = False) -> dict[str, Any]:
    input_npz = Path(input_npz)
    output_store = Path(output_store)
    if output_store.exists():
        if not overwrite:
            raise FileExistsError(f"Output store already exists: {output_store}")
        shutil.rmtree(output_store)
    output_store.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    _log(f"loading {input_npz}")
    data = np.load(input_npz, allow_pickle=True)
    genes = data["genes"].astype(str)
    contexts = data["contexts"].astype(str) if "contexts" in data else np.asarray(["global"] * len(genes), dtype=str)
    flat_vectors, offsets = _flat_arrays(data)
    total_evidence = int(flat_vectors.shape[0])

    _log("writing metadata arrays")
    np.save(output_store / "genes.npy", genes)
    np.save(output_store / "contexts.npy", contexts)
    np.save(output_store / "offsets.npy", offsets.astype(np.int64))

    _log(f"writing evidence vectors: shape={flat_vectors.shape}, dtype={flat_vectors.dtype}")
    np.save(output_store / "evidence_vectors.npy", np.asarray(flat_vectors))

    if "evidence_text_ids" in data:
        text_ids = data["evidence_text_ids"].astype(str)
    elif "text_ids" in data and len(data["text_ids"]) == total_evidence:
        text_ids = data["text_ids"].astype(str)
    else:
        text_ids = np.asarray(
            [
                f"{genes[i]}|evidence_{j}"
                for i in range(len(genes))
                for j in range(int(offsets[i + 1] - offsets[i]))
            ],
            dtype=str,
        )
    _log("writing evidence text ids")
    np.save(output_store / "evidence_text_ids.npy", text_ids)

    if "evidence_prior" in data:
        prior = data["evidence_prior"].astype(np.float32)
    else:
        prior = np.ones(total_evidence, dtype=np.float32)
    _log("writing evidence prior")
    np.save(output_store / "evidence_prior.npy", prior)

    if "evidence_metadata_features" in data:
        _log("writing evidence metadata features")
        np.save(output_store / "evidence_metadata_features.npy", data["evidence_metadata_features"].astype(np.float32))
    if "evidence_metadata_feature_names" in data:
        np.save(output_store / "evidence_metadata_feature_names.npy", data["evidence_metadata_feature_names"].astype(str))

    if "evidence_metadata_json" in data:
        _log("writing evidence metadata jsonl with byte offsets")
        metadata_json = data["evidence_metadata_json"].astype(str)
        if len(metadata_json) != total_evidence:
            by_id: dict[str, str] = {}
            for item in metadata_json:
                payload = json.loads(str(item))
                by_id[str(payload.get("text_id", ""))] = str(item)
            metadata_json = np.asarray([by_id.get(str(text_id), json.dumps({"text_id": str(text_id)})) for text_id in text_ids], dtype=object)
        byte_offsets: list[int] = []
        with (output_store / "evidence_metadata.jsonl").open("wb") as f:
            for item, text_id in zip(metadata_json, text_ids):
                byte_offsets.append(f.tell())
                payload = str(item).strip()
                if not payload:
                    payload = json.dumps({"text_id": str(text_id)})
                f.write(payload.encode("utf-8"))
                f.write(b"\n")
        np.save(output_store / "evidence_metadata_offsets.npy", np.asarray(byte_offsets, dtype=np.int64))

    counts = np.diff(offsets)
    metadata = {
        "source": str(input_npz),
        "storage": "text_evidence_store",
        "store": str(output_store),
        "num_sets": int(len(genes)),
        "num_genes": int(len(set(genes.tolist()))),
        "num_contexts": int(len(set(contexts.tolist()))),
        "total_evidence": total_evidence,
        "min_evidence": int(counts.min()) if len(counts) else 0,
        "median_evidence": float(np.median(counts)) if len(counts) else 0.0,
        "max_evidence": int(counts.max()) if len(counts) else 0,
        "vector_dim": int(flat_vectors.shape[1]) if flat_vectors.ndim == 2 else 0,
        "vector_dtype": str(flat_vectors.dtype),
        "elapsed_sec": time.time() - start_time,
    }
    write_json(output_store / "metadata.json", metadata)
    _log(f"done: {json.dumps(metadata, ensure_ascii=False)}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Export text evidence NPZ to mmap-friendly store.")
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--output-store", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    export_store(args.input_npz, args.output_store, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

