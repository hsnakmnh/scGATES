from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scgates.text import HashingTextEncoder


AD_TERMS_RE = re.compile(
    r"\b(ad|alzheimer|alzheimer's|amyloid|tau|dementia|neurodegeneration|a.?beta|abeta)\b",
    re.IGNORECASE,
)
METADATA_FEATURE_NAMES = [
    "source_reliability",
    "ad_specific",
    "evidence_strength",
    "recency_score",
    "has_pmid",
    "sentence_len_norm",
]


@dataclass(frozen=True)
class EvidenceSentence:
    gene: str
    pmid: str
    sentence: str
    year: str
    bp_type: str
    bp_name: str
    reg_type: str


def clean_text(text: str) -> str:
    return " ".join((text or "").replace("\t", " ").split())


def mask_target_gene_mentions(text: str, gene: str, mask_token: str = "[GENE]") -> tuple[str, int]:
    """Replace exact target-gene mentions with a neutral placeholder.

    We use alphanumeric boundaries instead of ``\b`` because gene symbols can
    contain punctuation such as ``-`` or ``+``. This intentionally masks only
    the target gene for the current evidence row, not other genes mentioned in
    the same sentence.
    """
    gene = str(gene).strip()
    if not gene:
        return text, 0
    pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(gene)}(?![A-Za-z0-9])", re.IGNORECASE)
    return pattern.subn(mask_token, text)


def parse_year(year: str) -> int | None:
    try:
        value = int(str(year).strip())
    except ValueError:
        return None
    return value if 1900 <= value <= 2100 else None


def evidence_strength(ev: EvidenceSentence) -> float:
    text = f"{ev.bp_type} {ev.bp_name} {ev.reg_type}".lower()
    score = 0.5
    if any(term in text for term in ["experiment", "validated", "curated", "manual"]):
        score += 0.3
    if ev.bp_name:
        score += 0.1
    if ev.reg_type:
        score += 0.1
    return min(score, 1.0)


def metadata_features(ev: EvidenceSentence, latest_year: int) -> np.ndarray:
    year = parse_year(ev.year)
    recency = 0.5 if year is None else max(0.0, min(1.0, (year - 1990) / max(latest_year - 1990, 1)))
    sentence_len = len(ev.sentence.split())
    return np.asarray(
        [
            1.0,  # AD-Alterome is a manually organized AD evidence source.
            1.0 if AD_TERMS_RE.search(f"{ev.sentence} {ev.bp_name}") else 0.0,
            evidence_strength(ev),
            recency,
            1.0 if ev.pmid else 0.0,
            max(0.0, min(1.0, sentence_len / 80.0)),
        ],
        dtype=np.float32,
    )


def evidence_prior_from_features(
    features: np.ndarray,
    gamma_source: float,
    gamma_ad: float,
    gamma_evidence: float,
    gamma_recency: float,
) -> np.ndarray:
    logits = (
        gamma_source * features[:, 0]
        + gamma_ad * features[:, 1]
        + gamma_evidence * features[:, 2]
        + gamma_recency * features[:, 3]
    )
    logits = logits - float(np.max(logits))
    weights = np.exp(logits).astype(np.float32)
    return weights / max(float(weights.sum()), 1e-8)


def read_evidence_sets(
    input_tsv: str | Path,
    dedup_key: str,
    gene_filter: set[str] | None,
    max_rows: int,
    mask_target_gene: bool = False,
    gene_mask_token: str = "[GENE]",
) -> OrderedDict[str, list[EvidenceSentence]]:
    evidence_by_gene: OrderedDict[str, list[EvidenceSentence]] = OrderedDict()
    seen: set[tuple[str, ...]] = set()
    with Path(input_tsv).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row_idx, row in enumerate(reader, start=1):
            if max_rows and row_idx > max_rows:
                break
            gene = clean_text(row.get("Gene", ""))
            sentence = clean_text(row.get("Sentence", ""))
            if not gene or not sentence:
                continue
            if gene_filter is not None and gene not in gene_filter:
                continue
            if mask_target_gene:
                sentence, _num_masked = mask_target_gene_mentions(sentence, gene, mask_token=gene_mask_token)
            pmid = clean_text(row.get("PMID", ""))
            if dedup_key == "gene_sentence":
                key = (gene, sentence)
            else:
                key = (gene, pmid, sentence)
            if key in seen:
                continue
            seen.add(key)
            evidence_by_gene.setdefault(gene, []).append(
                EvidenceSentence(
                    gene=gene,
                    pmid=pmid,
                    sentence=sentence,
                    year=clean_text(row.get("Year", "")),
                    bp_type=clean_text(row.get("BP_type", "")),
                    bp_name=clean_text(row.get("BP_name", "")),
                    reg_type=clean_text(row.get("RegType", "")),
                )
            )
    return evidence_by_gene


def l2_normalize(arr: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(denom, 1e-8)


def mean_pool(last_hidden, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-8)
    return summed / denom


def encode_sentences_hashing(sentences: list[str], dim: int) -> np.ndarray:
    encoder = HashingTextEncoder(dim=dim)
    return encoder.encode_many(sentences).astype(np.float32)


def encode_sentences_transformer(
    sentences: list[str],
    model_name_or_path: str,
    batch_size: int,
    max_length: int,
    device_arg: str,
    pooling: str,
) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = torch.device(device_arg if device_arg != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path).to(device)
    model.eval()

    vectors: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(sentences), batch_size):
            batch = sentences[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            out = model(**encoded)
            if pooling == "cls":
                pooled = out.last_hidden_state[:, 0]
            else:
                pooled = mean_pool(out.last_hidden_state, encoded["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            vectors.append(pooled.cpu().numpy().astype(np.float32))
    return np.concatenate(vectors, axis=0)


def write_gene_jsonl(path: str | Path, genes: list[str], counts: list[int]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for gene, count in zip(genes, counts):
            f.write(
                json.dumps(
                    {
                        "gene": gene,
                        "context": "global",
                        "text_id": f"{gene}|ad_alterome_raw_text_set",
                        "text": f"{gene} AD-Alterome raw text set with {count} deduplicated evidence sentences.",
                        "source": "AD-Alterome",
                        "num_evidence": count,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_evidence_csv(path: str | Path, evidence_by_gene: OrderedDict[str, list[EvidenceSentence]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "gene",
            "evidence_index",
            "pmid",
            "year",
            "bp_type",
            "bp_name",
            "reg_type",
            *METADATA_FEATURE_NAMES,
            "sentence",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        years = [
            parse_year(ev.year)
            for evidences in evidence_by_gene.values()
            for ev in evidences
            if parse_year(ev.year) is not None
        ]
        latest_year = max(years) if years else 2026
        for gene, evidences in evidence_by_gene.items():
            for idx, ev in enumerate(evidences):
                features = metadata_features(ev, latest_year)
                writer.writerow(
                    {
                        "gene": gene,
                        "evidence_index": idx,
                        "pmid": ev.pmid,
                        "year": ev.year,
                        "bp_type": ev.bp_type,
                        "bp_name": ev.bp_name,
                        "reg_type": ev.reg_type,
                        **{
                            name: float(value)
                            for name, value in zip(METADATA_FEATURE_NAMES, features, strict=True)
                        },
                        "sentence": ev.sentence,
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build raw gene-level text-set embeddings Pg from all deduplicated AD-Alterome evidence sentences. "
            "This intentionally does not apply AD keyword, length, gene-mention, or disease filters."
        )
    )
    parser.add_argument("--input-tsv", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--evidence-output-csv", default="")
    parser.add_argument("--encoder", choices=["transformer", "hashing"], default="transformer")
    parser.add_argument("--model-name-or-path", default="")
    parser.add_argument("--hash-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--dedup-key", choices=["gene_pmid_sentence", "gene_sentence"], default="gene_pmid_sentence")
    parser.add_argument("--gene-filter", default="", help="Optional comma-separated genes for local inspection.")
    parser.add_argument("--max-rows", type=int, default=0, help="Debug only. 0 means all rows.")
    parser.add_argument("--gamma-source", type=float, default=0.5)
    parser.add_argument("--gamma-ad", type=float, default=1.0)
    parser.add_argument("--gamma-evidence", type=float, default=0.5)
    parser.add_argument("--gamma-recency", type=float, default=0.2)
    parser.add_argument(
        "--mask-target-gene",
        action="store_true",
        help="Replace the target gene symbol in each evidence sentence with --gene-mask-token before encoding.",
    )
    parser.add_argument("--gene-mask-token", default="[GENE]")
    args = parser.parse_args()

    if args.encoder == "transformer" and not args.model_name_or_path:
        raise SystemExit("--model-name-or-path is required when --encoder transformer")

    gene_filter = {g.strip() for g in args.gene_filter.split(",") if g.strip()} if args.gene_filter else None
    evidence_by_gene = read_evidence_sets(
        input_tsv=args.input_tsv,
        dedup_key=args.dedup_key,
        gene_filter=gene_filter,
        max_rows=args.max_rows,
        mask_target_gene=args.mask_target_gene,
        gene_mask_token=args.gene_mask_token,
    )
    genes = list(evidence_by_gene.keys())
    counts = [len(evidence_by_gene[gene]) for gene in genes]
    if not genes:
        raise SystemExit("No evidence sentences found after raw deduplication")
    years = [
        parse_year(ev.year)
        for evidences in evidence_by_gene.values()
        for ev in evidences
        if parse_year(ev.year) is not None
    ]
    latest_year = max(years) if years else 2026

    flat_sentences: list[str] = []
    flat_text_ids: list[str] = []
    flat_metadata_json: list[str] = []
    flat_metadata_features: list[np.ndarray] = []
    flat_prior: list[float] = []
    offsets = [0]
    for gene in genes:
        gene_features = np.stack([metadata_features(ev, latest_year) for ev in evidence_by_gene[gene]], axis=0)
        gene_prior = evidence_prior_from_features(
            gene_features,
            gamma_source=args.gamma_source,
            gamma_ad=args.gamma_ad,
            gamma_evidence=args.gamma_evidence,
            gamma_recency=args.gamma_recency,
        )
        for evidence_index, ev in enumerate(evidence_by_gene[gene]):
            text_id = f"{gene}|{ev.pmid or 'no_pmid'}|{evidence_index}"
            features = gene_features[evidence_index]
            flat_sentences.append(ev.sentence)
            flat_text_ids.append(text_id)
            flat_metadata_features.append(features.astype(np.float32))
            flat_prior.append(float(gene_prior[evidence_index]))
            flat_metadata_json.append(
                json.dumps(
                    {
                        "text_id": text_id,
                        "gene": gene,
                        "text": ev.sentence,
                        "sentence": ev.sentence,
                        "pmid": ev.pmid,
                        "year": ev.year,
                        "bp_type": ev.bp_type,
                        "bp_name": ev.bp_name,
                        "reg_type": ev.reg_type,
                        "source": "AD-Alterome",
                        **{
                            name: float(value)
                            for name, value in zip(METADATA_FEATURE_NAMES, features, strict=True)
                        },
                    },
                    ensure_ascii=False,
                )
            )
        offsets.append(len(flat_sentences))

    if args.encoder == "hashing":
        sentence_vectors = encode_sentences_hashing(flat_sentences, dim=args.hash_dim)
        encoder_name = "hashing"
    else:
        sentence_vectors = encode_sentences_transformer(
            sentences=flat_sentences,
            model_name_or_path=args.model_name_or_path,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device_arg=args.device,
            pooling=args.pooling,
        )
        encoder_name = args.model_name_or_path

    sentence_vectors = l2_normalize(sentence_vectors.astype(np.float32))
    gene_vectors = []
    for start, end in zip(offsets[:-1], offsets[1:]):
        gene_vectors.append(sentence_vectors[start:end].mean(axis=0))
    gene_vectors_arr = l2_normalize(np.stack(gene_vectors, axis=0).astype(np.float32))

    write_gene_jsonl(args.output_jsonl, genes, counts)
    if args.evidence_output_csv:
        write_evidence_csv(args.evidence_output_csv, evidence_by_gene)

    out_npz = Path(args.output_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        vectors=gene_vectors_arr,
        evidence_vectors=sentence_vectors.astype(np.float32),
        genes=np.asarray(genes, dtype=object),
        contexts=np.asarray(["global"] * len(genes), dtype=object),
        text_ids=np.asarray([f"{gene}|ad_alterome_raw_text_set" for gene in genes], dtype=object),
        evidence_text_ids=np.asarray(flat_text_ids, dtype=object),
        evidence_metadata_json=np.asarray(flat_metadata_json, dtype=object),
        evidence_metadata_features=np.stack(flat_metadata_features, axis=0).astype(np.float32),
        evidence_metadata_feature_names=np.asarray(METADATA_FEATURE_NAMES, dtype=object),
        evidence_prior=np.asarray(flat_prior, dtype=np.float32),
        num_evidence=np.asarray(counts, dtype=np.int64),
        evidence_offsets=np.asarray(offsets, dtype=np.int64),
        encoder=encoder_name,
        pooling=args.pooling,
        max_length=args.max_length,
        dedup_key=args.dedup_key,
        mask_target_gene=bool(args.mask_target_gene),
        gene_mask_token=str(args.gene_mask_token),
        prior_gamma=np.asarray(
            [args.gamma_source, args.gamma_ad, args.gamma_evidence, args.gamma_recency],
            dtype=np.float32,
        ),
    )

    print(
        json.dumps(
            {
                "input_tsv": args.input_tsv,
                "output_jsonl": args.output_jsonl,
                "output_npz": str(out_npz),
                "evidence_output_csv": args.evidence_output_csv,
                "encoder": encoder_name,
                "genes_written": len(genes),
                "deduplicated_sentences": len(flat_sentences),
                "min_evidence_per_gene": int(min(counts)),
                "median_evidence_per_gene": float(np.median(np.asarray(counts))),
                "max_evidence_per_gene": int(max(counts)),
                "vector_dim": int(gene_vectors_arr.shape[1]),
                "mask_target_gene": bool(args.mask_target_gene),
                "gene_mask_token": str(args.gene_mask_token),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

