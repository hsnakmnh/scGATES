from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import GeneformerNeighborCollator, GeneformerNeighborDataset, aggregate_gene_priors


def _context_fields(value: str) -> list[str]:
    return [field.strip() for field in str(value).split(",") if field.strip()]


def _load_model(checkpoint_path: str | Path, device: torch.device):
    from .model import GeneformerNeighborAligner

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    hparams = dict(checkpoint.get("hyper_parameters", {}))
    if not hparams:
        raise ValueError(
            "Checkpoint does not contain Lightning hyper_parameters. "
            "Expected a GeneformerNeighborAligner checkpoint from stage1_neighbor_prior.train."
        )
    model = GeneformerNeighborAligner(**hparams)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().to(device)
    return model


@torch.no_grad()
def export_prior_from_dataset(
    model,
    dataset: GeneformerNeighborDataset,
    output_npz: str | Path,
    min_observations: int,
    source: str = "neighbor",
):
    if source not in {"neighbor", "target", "blend"}:
        raise ValueError(f"Unsupported prior export source: {source}")
    model.eval()
    rows: list[tuple[str, np.ndarray]] = []
    collator = GeneformerNeighborCollator()
    for sample in dataset:
        batch = collator([sample])
        target_z, neighbor_z = model.encode_batch(batch)
        if source == "target":
            selected_z = target_z
        elif source == "blend":
            selected_z = torch.nn.functional.normalize(0.5 * (target_z + neighbor_z), dim=-1)
        else:
            selected_z = neighbor_z
        vectors = np.asarray(selected_z.detach().cpu().float().tolist(), dtype=np.float32)
        for gene, vector in zip(sample["target_genes"], vectors):
            rows.append((str(gene), vector))
    genes, vectors, counts = aggregate_gene_priors(rows, min_observations=min_observations)
    metadata = {
        "source": "geneformer_neighbor_align",
        "prior_export_source": source,
        "stage1_training_objective": {
            "lambda_hard_negative": float(getattr(model.hparams, "lambda_hard_negative", 0.0)),
            "hard_negative_margin": float(getattr(model.hparams, "hard_negative_margin", 0.05)),
            "hard_negative_top_k": int(getattr(model.hparams, "hard_negative_top_k", 8)),
        },
        "cell_embeddings_jsonl": dataset.cell_embeddings_jsonl,
        "gene_neighbor_path": dataset.gene_neighbor_path,
        "num_genes": int(len(genes)),
        "vector_dim": int(vectors.shape[1]),
        "min_observations": int(min_observations),
        "min_gene_observations": int(counts.min()),
        "median_gene_observations": float(np.median(counts)),
        "max_gene_observations": int(counts.max()),
        "dataset_summary": dataset.summary,
        "device": str(model.device),
    }
    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        genes=genes,
        vectors=vectors.astype(np.float32),
        counts=counts,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False), dtype=object),
    )
    with output_npz.with_suffix(output_npz.suffix + ".metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Stage1-ready frozen gene-neighbor prior vectors from a Geneformer-native checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cell-embeddings-jsonl", required=True)
    parser.add_argument("--gene-neighbor-path", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--gene-allowlist-jsonl", default="")
    parser.add_argument("--context-fields", default="condition,context")
    parser.add_argument("--context-template", default="")
    parser.add_argument("--context-filter", default="")
    parser.add_argument("--sample-ratio", type=float, default=1.0)
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--max-genes-per-cell", type=int, default=512)
    parser.add_argument("--min-pairs-per-cell", type=int, default=4)
    parser.add_argument("--min-neighbors-per-gene", type=int, default=2)
    parser.add_argument("--max-neighbors-per-gene", type=int, default=10)
    parser.add_argument(
        "--prior-export-source",
        choices=["neighbor", "target", "blend"],
        default="neighbor",
        help=(
            "Which Stage1 representation to aggregate: neighbor keeps the original "
            "neighbor-set prior, target uses target-gene projector features, blend "
            "averages both."
        ),
    )
    parser.add_argument("--prior-min-observations", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _load_model(args.checkpoint, device)
    dataset = GeneformerNeighborDataset(
        cell_embeddings_jsonl=args.cell_embeddings_jsonl,
        gene_neighbor_path=args.gene_neighbor_path,
        context_fields=_context_fields(args.context_fields),
        context_template=args.context_template,
        context_filter=args.context_filter,
        gene_allowlist_jsonl=args.gene_allowlist_jsonl,
        max_cells=args.max_cells if args.max_cells and args.max_cells > 0 else None,
        sample_ratio=args.sample_ratio,
        seed=args.seed,
        max_genes_per_cell=args.max_genes_per_cell,
        min_pairs_per_cell=args.min_pairs_per_cell,
        min_neighbors_per_gene=args.min_neighbors_per_gene,
        max_neighbors_per_gene=args.max_neighbors_per_gene,
    )
    metadata = export_prior_from_dataset(
        model=model,
        dataset=dataset,
        output_npz=args.output_npz,
        min_observations=args.prior_min_observations,
        source=args.prior_export_source,
    )
    metadata.update(
        {
            "checkpoint": str(Path(args.checkpoint)),
            "output_npz": str(Path(args.output_npz)),
        }
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

