from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "run_id",
    "num_train_examples",
    "num_eval_examples",
    "paired_eval_all_genes",
    "example_split_mode",
    "eval_set_recall_at_1",
    "eval_set_recall_at_5",
    "eval_set_recall_at_10",
    "eval_set_mrr",
    "eval_set_mean_rank",
    "eval_set_median_rank",
    "eval_set_conf_filtered_recall_at_1",
    "eval_set_conf_filtered_recall_at_5",
    "eval_set_conf_filtered_recall_at_10",
    "eval_set_conf_filtered_mrr",
    "eval_set_conf_filtered_mean_rank",
    "eval_set_conf_filtered_median_rank",
    "eval_fused_set_recall_at_1",
    "eval_fused_set_recall_at_5",
    "eval_fused_set_recall_at_10",
    "eval_fused_set_mrr",
    "eval_fused_set_mean_rank",
    "eval_fused_set_median_rank",
    "eval_text_aware_set_recall_at_1",
    "eval_text_aware_set_recall_at_5",
    "eval_text_aware_set_recall_at_10",
    "eval_text_aware_set_mrr",
    "eval_set_diag_recall_at_1",
    "eval_set_diag_recall_at_5",
    "eval_set_diag_recall_at_10",
    "eval_set_diag_mrr",
    "eval_single_recall_at_1",
    "eval_single_recall_at_5",
    "eval_single_recall_at_10",
    "eval_single_mrr",
    "train_set_recall_at_1",
    "train_set_recall_at_5",
    "train_set_recall_at_10",
    "train_set_mrr",
    "train_set_conf_filtered_recall_at_1",
    "train_set_conf_filtered_recall_at_5",
    "train_set_conf_filtered_recall_at_10",
    "train_set_conf_filtered_mrr",
    "train_fused_set_recall_at_1",
    "train_fused_set_recall_at_5",
    "train_fused_set_recall_at_10",
    "train_fused_set_mrr",
    "train_text_aware_set_recall_at_1",
    "train_text_aware_set_recall_at_5",
    "train_text_aware_set_recall_at_10",
    "train_text_aware_set_mrr",
    "train_eval_set_recall_at_1_gap",
    "train_eval_set_mrr_gap",
    "train_eval_fused_set_recall_at_1_gap",
    "train_eval_fused_set_mrr_gap",
    "instance_score_eval_coverage",
    "instance_score_alpha",
    "instance_score_mode",
    "cell_denoise_mode",
    "lambda_denoise",
    "lambda_cell_denoise",
    "lambda_text_denoise",
    "text_conf_mode",
    "text_conf_gate_mode",
    "text_conf_gate_scale",
    "text_conf_beta_init",
    "text_conf_beta_final",
    "text_noise_kind",
    "text_noise_splits",
    "text_noise_train_noise_fraction",
    "text_noise_eval_noise_fraction",
    "text_noise_train_bank_size",
    "text_noise_eval_bank_size",
    "text_noise_train_replacement",
    "text_noise_eval_replacement",
    "cell_noise_ratio",
    "cell_noise_max",
    "cell_noise_splits",
    "cell_noise_source",
    "cell_noise_instance_score_mode",
    "cell_noise_train_noise_fraction",
    "cell_noise_eval_noise_fraction",
    "cell_noise_train_noise_instances",
    "cell_noise_eval_noise_instances",
    "eval_text_conf_filter_mode",
    "eval_text_conf_filter_frac",
    "eval_text_conf_filter_min_keep",
    "eval_conf_filter_mean_keep",
    "eval_conf_filter_mean_drop",
    "eval_conf_filter_mean_keep_fraction",
    "train_conf_filter_mean_keep",
    "train_conf_filter_mean_drop",
    "train_conf_filter_mean_keep_fraction",
    "fused_set_beta",
    "lambda_fused_set",
    "lambda_text_aware_set",
    "lambda_text_noise_conf",
    "text_noise_conf_margin",
    "loss_text_noise_conf",
    "text_noise_conf_margin_acc",
    "text_noise_conf_groups",
    "text_noise_conf_pairs",
    "best_epoch",
    "best_eval_set_recall_at_1",
    "best_eval_set_mrr",
    "best_fused_set_epoch",
    "best_eval_fused_set_recall_at_1",
    "best_eval_fused_set_mrr",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_epoch_summary(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row: dict[str, float] = {}
            for key, value in raw.items():
                if value is None or value == "":
                    continue
                try:
                    row[key] = float(value)
                except ValueError:
                    continue
            rows.append(row)
    return rows


def _summarize_run(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found: {metrics_path}")
    metrics = _load_json(metrics_path)
    row: dict[str, Any] = {"run_id": metrics.get("run_id", run_dir.name)}
    for field in FIELDS:
        if field == "run_id":
            continue
        row[field] = metrics.get(field, "")

    if "train_set_recall_at_1" in metrics and "eval_set_recall_at_1" in metrics:
        row["train_eval_set_recall_at_1_gap"] = float(metrics["train_set_recall_at_1"]) - float(metrics["eval_set_recall_at_1"])
    if "train_set_mrr" in metrics and "eval_set_mrr" in metrics:
        row["train_eval_set_mrr_gap"] = float(metrics["train_set_mrr"]) - float(metrics["eval_set_mrr"])
    if "train_fused_set_recall_at_1" in metrics and "eval_fused_set_recall_at_1" in metrics:
        row["train_eval_fused_set_recall_at_1_gap"] = (
            float(metrics["train_fused_set_recall_at_1"]) - float(metrics["eval_fused_set_recall_at_1"])
        )
    if "train_fused_set_mrr" in metrics and "eval_fused_set_mrr" in metrics:
        row["train_eval_fused_set_mrr_gap"] = float(metrics["train_fused_set_mrr"]) - float(metrics["eval_fused_set_mrr"])

    epoch_rows = _read_epoch_summary(run_dir / "epoch_eval_summary.csv")
    if epoch_rows:
        best = max(epoch_rows, key=lambda item: item.get("eval_set_recall_at_1", float("-inf")))
        row["best_epoch"] = int(best.get("epoch", 0))
        row["best_eval_set_recall_at_1"] = best.get("eval_set_recall_at_1", "")
        row["best_eval_set_mrr"] = best.get("eval_set_mrr", "")
        fused_candidates = [item for item in epoch_rows if "eval_fused_set_recall_at_1" in item]
        if fused_candidates:
            best_fused = max(fused_candidates, key=lambda item: item.get("eval_fused_set_recall_at_1", float("-inf")))
            row["best_fused_set_epoch"] = int(best_fused.get("epoch", 0))
            row["best_eval_fused_set_recall_at_1"] = best_fused.get("eval_fused_set_recall_at_1", "")
            row["best_eval_fused_set_mrr"] = best_fused.get("eval_fused_set_mrr", "")
    return row


def _fmt(value: Any) -> str:
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_markdown(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fields)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(_fmt(row.get(field, "")) for field in fields) + " |\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize scGATES metrics.json files into paper-ready tables.")
    parser.add_argument("run_dirs", nargs="+", help="Run directories under outputs/ or absolute paths.")
    parser.add_argument("--output-csv", default="outputs/selected_run_metrics.csv")
    parser.add_argument("--output-md", default="outputs/selected_run_metrics.md")
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Write a shorter table with the most commonly reported metrics.",
    )
    args = parser.parse_args()

    fields = FIELDS
    if args.compact:
        fields = [
            "run_id",
            "num_train_examples",
            "num_eval_examples",
            "eval_set_recall_at_1",
            "eval_set_recall_at_5",
            "eval_set_recall_at_10",
            "eval_set_mrr",
            "eval_set_conf_filtered_recall_at_1",
            "eval_set_conf_filtered_recall_at_5",
            "eval_set_conf_filtered_recall_at_10",
            "eval_set_conf_filtered_mrr",
            "eval_fused_set_recall_at_1",
            "eval_fused_set_mrr",
            "eval_text_aware_set_recall_at_1",
            "eval_text_aware_set_mrr",
            "eval_single_recall_at_1",
            "eval_single_recall_at_5",
            "eval_single_recall_at_10",
            "eval_single_mrr",
            "train_eval_set_recall_at_1_gap",
            "train_eval_fused_set_recall_at_1_gap",
            "instance_score_eval_coverage",
            "text_noise_kind",
            "text_noise_splits",
            "text_noise_train_noise_fraction",
            "text_noise_eval_noise_fraction",
            "cell_noise_ratio",
            "cell_noise_eval_noise_fraction",
            "eval_text_conf_filter_mode",
            "eval_conf_filter_mean_keep_fraction",
            "fused_set_beta",
            "lambda_fused_set",
            "lambda_text_noise_conf",
            "loss_text_noise_conf",
            "text_noise_conf_margin_acc",
            "best_epoch",
            "best_eval_set_recall_at_1",
            "best_fused_set_epoch",
            "best_eval_fused_set_recall_at_1",
        ]

    rows = [_summarize_run(Path(run_dir)) for run_dir in args.run_dirs]
    write_csv(Path(args.output_csv), rows, fields)
    write_markdown(Path(args.output_md), rows, fields)
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_md}")
    print("| " + " | ".join(fields) + " |")
    print("| " + " | ".join(["---"] * len(fields)) + " |")
    for row in rows:
        print("| " + " | ".join(_fmt(row.get(field, "")) for field in fields) + " |")


if __name__ == "__main__":
    main()

