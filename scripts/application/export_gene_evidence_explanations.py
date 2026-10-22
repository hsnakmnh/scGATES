from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


DEFAULT_FIELDS = [
    "gene",
    "context",
    "split",
    "rank_by_weight",
    "rank_by_conf",
    "selected_by",
    "text_id",
    "evidence_index",
    "local_index",
    "text",
    "weight",
    "conf",
    "prior",
    "gated_prior",
    "content_logit",
    "prior_logit",
    "conf_logit",
    "attn_logit",
    "is_noise",
    "noise_kind",
    "source_gene",
    "source_context",
    "source_text_id",
    "target_gene",
    "target_context",
]

SUMMARY_NUMERIC_FIELDS = [
    "weight",
    "conf",
    "prior",
    "gated_prior",
    "content_logit",
    "prior_logit",
    "conf_logit",
    "attn_logit",
]


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed):
        return default
    return parsed


def _maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    parsed = _maybe_float(value)
    if parsed is not None:
        return f"{parsed:.{digits}f}"
    return str(value)


def _gene_matches(row: dict[str, str], gene: str) -> bool:
    gene_upper = gene.upper()
    row_gene = row.get("gene", "").upper()
    target_gene = row.get("target_gene", "").upper()
    return row_gene == gene_upper or target_gene == gene_upper


def _split_matches(row: dict[str, str], split: str) -> bool:
    if split.lower() in {"", "all", "any"}:
        return True
    return row.get("split", "") == split


def _read_gene_rows(path: Path, gene: str, split: str) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows = [dict(row) for row in reader if _gene_matches(row, gene) and _split_matches(row, split)]
        return rows, list(reader.fieldnames)


def _require_text(rows: list[dict[str, str]], path: Path) -> None:
    if not rows:
        return
    if "text" not in rows[0]:
        raise ValueError(f"`text` column not found in {path}")
    non_empty = sum(1 for row in rows if row.get("text", "").strip())
    if non_empty == 0:
        raise ValueError(f"`text` column is present but empty for selected gene rows in {path}")


def _rank_rows(rows: list[dict[str, str]], field: str) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: _as_float(row.get(field, ""), default=float("-inf")), reverse=True)


def _add_ranks(rows: list[dict[str, str]], ranked_weight: list[dict[str, str]], ranked_conf: list[dict[str, str]]) -> None:
    rank_by_weight = {id(row): idx for idx, row in enumerate(ranked_weight, start=1)}
    rank_by_conf = {id(row): idx for idx, row in enumerate(ranked_conf, start=1)}
    for row in rows:
        row["rank_by_weight"] = str(rank_by_weight.get(id(row), ""))
        row["rank_by_conf"] = str(rank_by_conf.get(id(row), ""))


def _top_union(
    ranked_weight: list[dict[str, str]],
    ranked_conf: list[dict[str, str]],
    top_k: int,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    selected_ids: set[int] = set()
    by_id: dict[int, set[str]] = {}

    for row in ranked_weight[:top_k]:
        row_id = id(row)
        by_id.setdefault(row_id, set()).add("weight")
        if row_id not in selected_ids:
            selected.append(row)
            selected_ids.add(row_id)
    for row in ranked_conf[:top_k]:
        row_id = id(row)
        by_id.setdefault(row_id, set()).add("conf")
        if row_id not in selected_ids:
            selected.append(row)
            selected_ids.add(row_id)

    for row in selected:
        row["selected_by"] = "+".join(sorted(by_id.get(id(row), set())))
    return selected


def _numeric_values(rows: list[dict[str, str]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        parsed = _maybe_float(row.get(field, ""))
        if parsed is not None:
            values.append(parsed)
    return values


def _noise_fraction(rows: list[dict[str, str]]) -> float | None:
    if not rows:
        return None
    noisy = 0
    has_noise_col = False
    for row in rows:
        if "is_noise" in row:
            has_noise_col = True
            noisy += 1 if str(row.get("is_noise", "")).strip() in {"1", "true", "True", "yes"} else 0
    if not has_noise_col:
        return None
    return noisy / len(rows)


def _context_summary(rows: list[dict[str, str]]) -> str:
    contexts = sorted({row.get("context", "") for row in rows if row.get("context", "")})
    if not contexts:
        return ""
    if len(contexts) == 1:
        return contexts[0]
    return "; ".join(contexts[:3]) + ("; ..." if len(contexts) > 3 else "")


def _cell_summary(path: Path | None, gene: str, split: str) -> dict[str, Any]:
    if path is None:
        return {}
    rows, _ = _read_gene_rows(path, gene, split)
    out: dict[str, Any] = {
        "cell_num_instances": len(rows),
        "cell_context": _context_summary(rows),
    }
    for field in ["weight", "conf", "prior", "instance_score", "attn_logit"]:
        out[f"cell_{field}_mean"] = _mean(_numeric_values(rows, field))
    return out


def _summarize(rows: list[dict[str, str]], top_weight: list[dict[str, str]], cell_stats: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_evidence": len(rows),
        "context": _context_summary(rows),
        "top_k": len(top_weight),
        "total_attention_mass_top_k": sum(_numeric_values(top_weight, "weight")),
        "top_k_noise_fraction": _noise_fraction(top_weight),
    }
    for field in SUMMARY_NUMERIC_FIELDS:
        summary[f"mean_{field}"] = _mean(_numeric_values(rows, field))
        summary[f"mean_{field}_top_k"] = _mean(_numeric_values(top_weight, field))
    summary.update(cell_stats)
    return summary


def _escape_md(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _truncate(text: str, max_chars: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _md_table(rows: list[dict[str, str]], max_text_chars: int) -> list[str]:
    fields = ["rank", "weight", "conf", "prior", "gated_prior", "is_noise", "source_gene", "text"]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for idx, row in enumerate(rows, start=1):
        values = {
            "rank": str(idx),
            "weight": _fmt(row.get("weight", "")),
            "conf": _fmt(row.get("conf", "")),
            "prior": _fmt(row.get("prior", "")),
            "gated_prior": _fmt(row.get("gated_prior", "")),
            "is_noise": row.get("is_noise", ""),
            "source_gene": row.get("source_gene", ""),
            "text": _truncate(row.get("text", ""), max_text_chars),
        }
        lines.append("| " + " | ".join(_escape_md(values[field]) for field in fields) + " |")
    return lines


def _write_csv(path: Path, rows: list[dict[str, str]], input_fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(DEFAULT_FIELDS)
    for field in input_fields:
        if field not in fields:
            fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_md(
    path: Path,
    gene: str,
    split: str,
    summary: dict[str, Any],
    top_weight: list[dict[str, str]],
    top_conf: list[dict[str, str]],
    max_text_chars: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"# Gene Evidence Explanation: {gene}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    summary_keys = [
        "context",
        "num_evidence",
        "top_k",
        "mean_weight",
        "mean_conf",
        "mean_prior",
        "mean_gated_prior",
        "total_attention_mass_top_k",
        "top_k_noise_fraction",
        "mean_conf_top_k",
        "mean_gated_prior_top_k",
        "cell_num_instances",
        "cell_weight_mean",
        "cell_conf_mean",
        "cell_instance_score_mean",
    ]
    lines.append(f"- gene: {gene}")
    lines.append(f"- split: {split}")
    for key in summary_keys:
        if key in summary and summary[key] not in {None, ""}:
            lines.append(f"- {key}: {_fmt(summary[key])}")
    lines.append("")
    lines.append("## Top Evidence By Attention Weight")
    lines.append("")
    lines.extend(_md_table(top_weight, max_text_chars))
    lines.append("")
    lines.append("## Top Evidence By Confidence")
    lines.append("")
    lines.extend(_md_table(top_conf, max_text_chars))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_genes(args: argparse.Namespace) -> list[str]:
    genes: list[str] = []
    if args.gene:
        genes.append(args.gene)
    if args.genes:
        genes.extend(g.strip() for g in args.genes.split(",") if g.strip())
    seen: set[str] = set()
    unique: list[str] = []
    for gene in genes:
        key = gene.upper()
        if key not in seen:
            unique.append(gene)
            seen.add(key)
    if not unique:
        raise SystemExit("Provide --gene or --genes.")
    return unique


def export_gene(args: argparse.Namespace, gene: str) -> tuple[Path, Path]:
    evidence_path = Path(args.evidence_weights)
    cell_path = Path(args.cell_weights) if args.cell_weights else None
    output_dir = Path(args.output_dir)

    rows, input_fields = _read_gene_rows(evidence_path, gene, args.split)
    if not rows:
        raise ValueError(f"No evidence rows found for gene={gene!r}, split={args.split!r} in {evidence_path}")
    _require_text(rows, evidence_path)

    ranked_weight = _rank_rows(rows, "weight")
    ranked_conf = _rank_rows(rows, "conf")
    _add_ranks(rows, ranked_weight, ranked_conf)

    top_weight = ranked_weight[: args.top_k]
    top_conf = ranked_conf[: args.top_k]
    selected_rows = _top_union(ranked_weight, ranked_conf, args.top_k)
    selected_rows.sort(key=lambda row: int(row.get("rank_by_weight") or 10**9))

    cell_stats = _cell_summary(cell_path, gene, args.split)
    summary = _summarize(rows, top_weight, cell_stats)

    gene_name = gene.upper()
    csv_path = output_dir / f"{gene_name}.csv"
    md_path = output_dir / f"{gene_name}.md"
    _write_csv(csv_path, selected_rows, input_fields)
    _write_md(md_path, gene_name, args.split, summary, top_weight, top_conf, args.max_text_chars)
    return csv_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export per-gene evidence-level explanations from scGATES evidence_weights.csv."
    )
    parser.add_argument("--evidence-weights", required=True, help="Path to evidence_weights.csv.")
    parser.add_argument("--cell-weights", default="", help="Optional path to cell_weights.csv.")
    parser.add_argument("--gene", default="", help="Single gene symbol, e.g. APOE.")
    parser.add_argument("--genes", default="", help="Comma-separated gene symbols.")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--split", default="eval", help="Split to export: eval, train, or all.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-text-chars", type=int, default=450)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on the first missing gene. By default, batch mode skips missing genes with a warning.",
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive.")
    if args.max_text_chars < 40:
        raise SystemExit("--max-text-chars must be >= 40.")

    genes = _parse_genes(args)
    outputs: list[tuple[Path, Path]] = []
    failures: list[str] = []
    for gene in genes:
        try:
            outputs.append(export_gene(args, gene))
        except ValueError as exc:
            if args.strict or len(genes) == 1:
                raise
            failures.append(f"{gene}: {exc}")

    for csv_path, md_path in outputs:
        print(f"wrote {csv_path}")
        print(f"wrote {md_path}")
    for failure in failures:
        print(f"warning: skipped {failure}")
    if not outputs:
        raise SystemExit("No gene explanations were exported.")


if __name__ == "__main__":
    main()

