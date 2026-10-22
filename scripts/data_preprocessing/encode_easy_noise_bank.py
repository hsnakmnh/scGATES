from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.encode_ad_alterome_text_set import (
        encode_sentences_hashing,
        encode_sentences_transformer,
        l2_normalize,
    )
except ModuleNotFoundError:
    from scripts.data_preprocessing.encode_ad_alterome_text_set import (
        encode_sentences_hashing,
        encode_sentences_transformer,
        l2_normalize,
    )


def read_noise_jsonl(path: str | Path) -> tuple[list[str], list[str], list[str]]:
    noise_ids: list[str] = []
    categories: list[str] = []
    texts: list[str] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            noise_id = str(payload.get("noise_id", "")).strip()
            category = str(payload.get("category", "")).strip()
            text = str(payload.get("text", "")).strip()
            if not noise_id or not text:
                raise ValueError(f"Bad noise row at line {line_no}: expected noise_id and text")
            noise_ids.append(noise_id)
            categories.append(category)
            texts.append(text)
    if not texts:
        raise ValueError(f"No noise text records found: {path}")
    if len(set(noise_ids)) != len(noise_ids):
        raise ValueError(f"Duplicate noise_id values found: {path}")
    return noise_ids, categories, texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode a synthetic easy-noise JSONL bank into an NPZ embedding bank.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--encoder", choices=["transformer", "hashing"], default="transformer")
    parser.add_argument("--model-name-or-path", default="")
    parser.add_argument("--hash-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    args = parser.parse_args()

    if args.encoder == "transformer" and not args.model_name_or_path:
        raise SystemExit("--model-name-or-path is required when --encoder transformer")

    noise_ids, categories, texts = read_noise_jsonl(args.input_jsonl)
    if args.encoder == "hashing":
        vectors = encode_sentences_hashing(texts, dim=args.hash_dim)
        encoder_name = "hashing"
    else:
        vectors = encode_sentences_transformer(
            sentences=texts,
            model_name_or_path=args.model_name_or_path,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device_arg=args.device,
            pooling=args.pooling,
        )
        encoder_name = args.model_name_or_path
    vectors = l2_normalize(vectors.astype(np.float32))

    out = Path(args.output_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        noise_ids=np.asarray(noise_ids, dtype=object),
        categories=np.asarray(categories, dtype=object),
        texts=np.asarray(texts, dtype=object),
        embeddings=vectors.astype(np.float32),
        encoder=str(encoder_name),
        pooling=str(args.pooling),
        max_length=np.asarray(args.max_length, dtype=np.int64),
    )
    print(
        json.dumps(
            {
                "input_jsonl": args.input_jsonl,
                "output_npz": str(out),
                "num_noise_texts": len(texts),
                "encoder": str(encoder_name),
                "pooling": str(args.pooling),
                "max_length": int(args.max_length),
                "embedding_dim": int(vectors.shape[1]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

