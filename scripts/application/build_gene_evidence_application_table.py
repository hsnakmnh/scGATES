from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _is_noise(row: dict[str, str]) -> bool:
    return str(row.get("is_noise", "")).strip().lower() in {"1", "true", "yes"}


def _escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _compact_text(text: str, max_chars: int) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _read_md_summary(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    summary: dict[str, str] = {}
    in_summary = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "## Summary":
            in_summary = True
            continue
        if in_summary and stripped.startswith("## "):
            break
        if not in_summary or not stripped.startswith("- "):
            continue
        item = stripped[2:]
        if ":" not in item:
            continue
        key, value = item.split(":", 1)
        summary[key.strip()] = value.strip()
    return summary


def _ranked_by_weight(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: _as_float(row.get("weight")), reverse=True)


def _parse_selected_ranks(value: str) -> dict[str, list[int]]:
    selected: dict[str, list[int]] = {}
    if not value.strip():
        return selected
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid --selected-ranks item: {item!r}")
        gene, ranks_raw = item.split(":", 1)
        ranks: list[int] = []
        for rank_raw in ranks_raw.split(","):
            rank_raw = rank_raw.strip()
            if not rank_raw:
                continue
            rank = int(rank_raw)
            if rank <= 0:
                raise ValueError(f"Rank must be positive in --selected-ranks: {item!r}")
            ranks.append(rank)
        selected[gene.strip().upper()] = ranks
    return selected


def _summary_row(gene: str, rows: list[dict[str, str]], md_summary: dict[str, str], top_noise_k: int) -> dict[str, str]:
    ranked = sorted(rows, key=lambda row: _as_float(row.get("weight")), reverse=True)
    top = ranked[:top_noise_k]
    num_evidence = md_summary.get("num_evidence") or str(len(rows))
    top_mass = sum(_as_float(row.get("weight")) for row in top)
    noise_count = sum(1 for row in top if _is_noise(row))
    return {
        "gene": gene,
        "context": md_summary.get("context", rows[0].get("context", "") if rows else ""),
        "num_evidence": str(_as_int(num_evidence, len(rows))),
        f"top_{top_noise_k}_attention_mass": f"{top_mass:.6f}",
        f"top_{top_noise_k}_noise_count": str(noise_count),
        f"top_{top_noise_k}_noise_fraction": f"{noise_count / max(len(top), 1):.6f}",
    }


def _build_rows(
    explanation_dir: Path,
    genes: list[str],
    top_k: int,
    top_noise_k: int,
    max_text_chars: int,
    selected_ranks: dict[str, list[int]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str]]:
    detail_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []
    skipped: list[str] = []

    for gene in genes:
        gene_upper = gene.upper()
        csv_path = explanation_dir / f"{gene_upper}.csv"
        md_path = explanation_dir / f"{gene_upper}.md"
        if not csv_path.exists():
            skipped.append(gene_upper)
            continue
        rows = _read_csv(csv_path)
        if not rows:
            skipped.append(gene_upper)
            continue
        summary = _read_md_summary(md_path)
        summary_rows.append(_summary_row(gene_upper, rows, summary, top_noise_k))
        ranked_rows = _ranked_by_weight(rows)
        requested_ranks = selected_ranks.get(gene_upper)
        if requested_ranks:
            selected_rows = []
            for original_rank in requested_ranks:
                if original_rank > len(ranked_rows):
                    continue
                selected_rows.append((original_rank, ranked_rows[original_rank - 1]))
        else:
            selected_rows = list(enumerate(ranked_rows[:top_k], start=1))
        for display_rank, (original_rank, row) in enumerate(selected_rows, start=1):
            detail_rows.append(
                {
                    "gene": gene_upper,
                    "rank": str(display_rank),
                    "original_weight_rank": str(original_rank),
                    "context": row.get("context", ""),
                    "text_id": row.get("text_id", ""),
                    "evidence": _compact_text(row.get("text", ""), max_text_chars),
                    "weight": f"{_as_float(row.get('weight')):.6f}",
                    "conf": f"{_as_float(row.get('conf')):.6f}",
                    "prior": f"{_as_float(row.get('prior')):.6f}",
                    "gated_prior": f"{_as_float(row.get('gated_prior')):.6f}" if row.get("gated_prior", "") != "" else "",
                    "is_noise": row.get("is_noise", ""),
                    "noise_kind": row.get("noise_kind", ""),
                    "source_gene": row.get("source_gene", ""),
                    "source_text_id": row.get("source_text_id", ""),
                }
            )
    return detail_rows, summary_rows, skipped


def _write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_md(
    path: Path,
    detail_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
    detail_fields: list[str],
    summary_fields: list[str],
    skipped: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# scGATES Gene Evidence Explanation Application Table")
    lines.append("")
    lines.append(
        "Given a query gene's cell-side distribution representation, scGATES returns the top-weighted text evidence selected during set-level alignment."
    )
    lines.append("")
    lines.append("These rows are model-selected supporting evidence, not newly validated biological mechanisms.")
    lines.append("")
    if skipped:
        lines.append("Skipped genes with no exported explanation CSV: " + ", ".join(skipped))
        lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| " + " | ".join(summary_fields) + " |")
    lines.append("| " + " | ".join(["---"] * len(summary_fields)) + " |")
    for row in summary_rows:
        lines.append("| " + " | ".join(_escape_md(row.get(field, "")) for field in summary_fields) + " |")
    lines.append("")
    lines.append("## Top Evidence")
    lines.append("")
    lines.append("| " + " | ".join(detail_fields) + " |")
    lines.append("| " + " | ".join(["---"] * len(detail_fields)) + " |")
    for row in detail_rows:
        lines.append("| " + " | ".join(_escape_md(row.get(field, "")) for field in detail_fields) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a multi-gene application table from exported scGATES evidence explanations.")
    parser.add_argument("--explanation-dir", required=True)
    parser.add_argument("--genes", required=True, help="Comma-separated genes to include.")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--top-noise-k", type=int, default=10)
    parser.add_argument("--max-text-chars", type=int, default=240)
    parser.add_argument(
        "--selected-ranks",
        default="",
        help="Optional semicolon-separated original weight ranks, e.g. 'CD33:1,2,5;PICALM:1,3,4'.",
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    genes = [gene.strip().upper() for gene in args.genes.split(",") if gene.strip()]
    if not genes:
        raise SystemExit("No genes provided.")
    if args.top_k <= 0 or args.top_noise_k <= 0:
        raise SystemExit("--top-k and --top-noise-k must be positive.")

    detail_rows, summary_rows, skipped = _build_rows(
        Path(args.explanation_dir),
        genes,
        args.top_k,
        args.top_noise_k,
        args.max_text_chars,
        _parse_selected_ranks(args.selected_ranks),
    )
    if not detail_rows:
        raise SystemExit("No application table rows were generated.")

    detail_fields = [
        "gene",
        "rank",
        "original_weight_rank",
        "evidence",
        "weight",
        "gated_prior",
        "is_noise",
        "source_gene",
    ]
    csv_fields = [
        "gene",
        "rank",
        "original_weight_rank",
        "context",
        "text_id",
        "evidence",
        "weight",
        "conf",
        "prior",
        "gated_prior",
        "is_noise",
        "noise_kind",
        "source_gene",
        "source_text_id",
    ]
    summary_fields = [
        "gene",
        "context",
        "num_evidence",
        f"top_{args.top_noise_k}_attention_mass",
        f"top_{args.top_noise_k}_noise_count",
        f"top_{args.top_noise_k}_noise_fraction",
    ]
    _write_csv(Path(args.output_csv), detail_rows, csv_fields)
    _write_md(Path(args.output_md), detail_rows, summary_rows, detail_fields, summary_fields, skipped)
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_md}")
    if skipped:
        print("warning: skipped " + ", ".join(skipped))


if __name__ == "__main__":
    main()

