from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, random_split

from .data import GeneformerNeighborCollator, GeneformerNeighborDataset, aggregate_gene_priors
from .model import GeneformerNeighborAligner


def _load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _merge_config(args, config: dict):
    merged = deepcopy(vars(args))
    for key, value in config.items():
        merged[key] = value
    return argparse.Namespace(**merged)


def _context_fields(value: str) -> list[str]:
    return [field.strip() for field in str(value).split(",") if field.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Geneformer-native gene-neighbor prior model.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--filename", type=str, default="geneformer-neighbor-align")
    parser.add_argument("--output_dir", type=str, default="all_checkpoints")

    parser.add_argument("--cell_embeddings_jsonl", type=str, default="")
    parser.add_argument("--gene_neighbor_path", type=str, default="")
    parser.add_argument("--gene_allowlist_jsonl", type=str, default="")
    parser.add_argument("--context_fields", type=str, default="condition,context")
    parser.add_argument("--context_template", type=str, default="")
    parser.add_argument("--context_filter", type=str, default="")
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    parser.add_argument("--max_cells", type=int, default=0)

    parser.add_argument("--max_genes_per_cell", type=int, default=512)
    parser.add_argument("--min_pairs_per_cell", type=int, default=4)
    parser.add_argument("--min_neighbors_per_gene", type=int, default=2)
    parser.add_argument("--max_neighbors_per_gene", type=int, default=10)

    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--init_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument(
        "--lambda_hard_negative",
        type=float,
        default=0.0,
        help=(
            "Weight for the in-cell hard-negative margin loss. Non-matching "
            "target/neighbor pairs from the same cell are treated as hard negatives."
        ),
    )
    parser.add_argument("--hard_negative_margin", type=float, default=0.05)
    parser.add_argument("--hard_negative_top_k", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)

    parser.add_argument(
        "--output_prior_npz",
        type=str,
        default="",
        help="Optional Stage1-ready prior NPZ exported from the trained final checkpoint.",
    )
    parser.add_argument(
        "--prior_export_source",
        choices=["neighbor", "target", "blend"],
        default="neighbor",
        help=(
            "Which Stage1 representation to aggregate into the downstream prior. "
            "'neighbor' keeps the original neighbor-set prior; 'target' aggregates "
            "the target-gene projector representation; 'blend' averages target and "
            "neighbor features before aggregation."
        ),
    )
    parser.add_argument("--prior_min_observations", type=int, default=1)
    return parser


@torch.no_grad()
def export_prior_from_dataset(
    model: GeneformerNeighborAligner,
    dataset: GeneformerNeighborDataset,
    output_npz: str | Path,
    min_observations: int = 1,
    source: str = "neighbor",
) -> dict:
    if source not in {"neighbor", "target", "blend"}:
        raise ValueError(f"Unsupported prior export source: {source}")
    model.eval()
    device = model.device
    rows: list[tuple[str, np.ndarray]] = []
    for sample in dataset:
        batch = GeneformerNeighborCollator()([sample])
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
        "device": str(device),
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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.config:
        args = _merge_config(args, _load_config(args.config))

    if not args.cell_embeddings_jsonl or not args.gene_neighbor_path:
        raise ValueError("Both --cell_embeddings_jsonl and --gene_neighbor_path are required.")
    if args.batch_size != 1:
        raise ValueError("Geneformer neighbor training currently expects --batch_size 1.")

    pl.seed_everything(args.seed, workers=True)
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
    if len(dataset) < 2:
        raise RuntimeError("Need at least 2 valid cell samples after filtering.")

    n_total = len(dataset)
    n_train = max(1, int(n_total * args.train_ratio))
    if n_train >= n_total:
        n_train = n_total - 1
    n_val = n_total - n_train
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=generator)
    collator = GeneformerNeighborCollator()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    model = GeneformerNeighborAligner(
        input_dim=dataset.embedding_dim,
        hidden_dim=args.hidden_dim,
        proj_dim=args.proj_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        init_lr=args.init_lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        temperature=args.temperature,
        lambda_hard_negative=args.lambda_hard_negative,
        hard_negative_margin=args.hard_negative_margin,
        hard_negative_top_k=args.hard_negative_top_k,
    )

    run_dir = os.path.join(args.output_dir, args.filename)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "resolved_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    manifest = {
        **dataset.summary,
        "num_train_cell_samples": int(n_train),
        "num_val_cell_samples": int(n_val),
        "run_dir": run_dir,
    }
    with open(os.path.join(run_dir, "training_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger = CSVLogger(save_dir=os.path.join(run_dir, "logs"), name="train")
    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(run_dir, "checkpoints"),
        filename="geneformer-neighbor-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=args.patience,
        verbose=True,
    )
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint, early_stop],
        logger=logger,
        log_every_n_steps=5,
        gradient_clip_val=args.gradient_clip_val,
    )
    trainer.fit(model, train_loader, val_loader)

    final_path = os.path.join(run_dir, "final.ckpt")
    trainer.save_checkpoint(final_path)
    print(f"[INFO] Saved final checkpoint: {final_path}")

    if args.output_prior_npz:
        metadata = export_prior_from_dataset(
            model,
            dataset,
            args.output_prior_npz,
            args.prior_min_observations,
            source=args.prior_export_source,
        )
        print(json.dumps({"event": "exported_geneformer_neighbor_prior", **metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

