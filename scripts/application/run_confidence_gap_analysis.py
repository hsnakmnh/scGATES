from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


QUADRANT_LABELS = {
    "high_cell_high_text": "High-confidence aligned genes",
    "high_cell_low_text": "Confidence-gap candidates",
    "low_cell_high_text": "Text-confidence-heavy genes",
    "low_cell_low_text": "Low-confidence genes",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _is_noise(row: dict[str, str]) -> bool:
    return str(row.get("is_noise", "")).strip().lower() in {"1", "true", "yes"}


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    n = len(sorted_values)
    mid = n // 2
    if n % 2:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def _group_by_gene(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        gene = str(row.get("gene", "")).strip()
        if gene:
            grouped[gene].append(row)
    return grouped


def _most_common_context(rows: list[dict[str, str]]) -> str:
    contexts = [str(row.get("context", "")).strip() for row in rows if str(row.get("context", "")).strip()]
    if not contexts:
        return ""
    return Counter(contexts).most_common(1)[0][0]


def _read_gene_allowlist(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Gene allowlist not found: {path}")
    genes: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if not clean or clean.startswith("#") or clean.lower() == "gene":
            continue
        genes.add(clean.split(",")[0].split()[0])
    return genes


def _weighted_confidence(
    rows: list[dict[str, str]],
    weight_column: str,
    conf_column: str,
) -> tuple[float, float, float]:
    total_weight = sum(max(_as_float(row.get(weight_column)), 0.0) for row in rows)
    if total_weight <= 0:
        raw_values = [_as_float(row.get(conf_column)) for row in rows]
        raw_mean = sum(raw_values) / len(raw_values) if raw_values else 0.0
        return _sigmoid(raw_mean), raw_mean, 0.0

    score = 0.0
    raw_score = 0.0
    for row in rows:
        weight = max(_as_float(row.get(weight_column)), 0.0)
        raw_conf = _as_float(row.get(conf_column))
        score += weight * _sigmoid(raw_conf)
        raw_score += weight * raw_conf
    return score / total_weight, raw_score / total_weight, total_weight


def _compute_scores(
    cell_rows: list[dict[str, str]],
    evidence_rows: list[dict[str, str]],
    split: str,
    top_k: int,
    min_clean_evidence: int,
    cell_conf_column: str,
    text_conf_column: str,
    allowlist: set[str] | None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    cell_eval = [row for row in cell_rows if str(row.get("split", "")).strip() == split]
    evidence_eval = [row for row in evidence_rows if str(row.get("split", "")).strip() == split]
    if not cell_eval:
        raise ValueError(f"No cell rows found for split={split!r}")
    if not evidence_eval:
        raise ValueError(f"No evidence rows found for split={split!r}")
    if cell_conf_column not in cell_eval[0]:
        raise ValueError(f"cell rows missing confidence column: {cell_conf_column}")
    if text_conf_column not in evidence_eval[0]:
        raise ValueError(f"evidence rows missing confidence column: {text_conf_column}")

    cell_by_gene = _group_by_gene(cell_eval)
    evidence_by_gene = _group_by_gene(evidence_eval)
    common_genes = sorted(set(cell_by_gene) & set(evidence_by_gene))
    if allowlist is not None:
        common_genes = [gene for gene in common_genes if gene in allowlist]
    if not common_genes:
        raise ValueError("No common genes found after split and allowlist filtering.")

    rows: list[dict[str, Any]] = []
    for gene in common_genes:
        c_rows = cell_by_gene[gene]
        e_rows = evidence_by_gene[gene]
        clean_rows = [row for row in e_rows if not _is_noise(row)]
        if len(clean_rows) < min_clean_evidence:
            continue

        c_top = sorted(c_rows, key=lambda row: _as_float(row.get("weight")), reverse=True)[:top_k]
        e_top = sorted(e_rows, key=lambda row: _as_float(row.get("weight")), reverse=True)[:top_k]

        cell_score, cell_raw_conf, cell_topk_mass = _weighted_confidence(c_top, "weight", cell_conf_column)
        text_score, text_raw_conf, text_topk_mass = _weighted_confidence(e_top, "weight", text_conf_column)
        text_noise_count = sum(1 for row in e_rows if _is_noise(row))
        text_topk_noise_fraction = sum(1 for row in e_top if _is_noise(row)) / len(e_top) if e_top else 0.0

        rows.append(
            {
                "gene": gene,
                "context": _most_common_context(c_rows) or _most_common_context(e_rows),
                "cell_instance_count": len(c_rows),
                "cell_topk_mass": cell_topk_mass,
                "cell_conf_raw_weighted_mean": cell_raw_conf,
                "text_evidence_count": len(e_rows),
                "text_clean_count": len(clean_rows),
                "text_noise_count": text_noise_count,
                "text_topk_noise_fraction": text_topk_noise_fraction,
                "text_topk_attention_mass": text_topk_mass,
                "text_conf_raw_weighted_mean": text_raw_conf,
                "C_score": cell_score,
                "T_score": text_score,
            }
        )

    if not rows:
        raise ValueError("No genes remained after min-clean-evidence filtering.")

    c_threshold = _median([row["C_score"] for row in rows])
    t_threshold = _median([row["T_score"] for row in rows])
    for row in rows:
        high_c = row["C_score"] >= c_threshold
        high_t = row["T_score"] >= t_threshold
        if high_c and high_t:
            quadrant = "high_cell_high_text"
        elif high_c and not high_t:
            quadrant = "high_cell_low_text"
        elif not high_c and high_t:
            quadrant = "low_cell_high_text"
        else:
            quadrant = "low_cell_low_text"
        row["quadrant"] = QUADRANT_LABELS[quadrant]
        row["_quadrant_key"] = quadrant

    thresholds = {"C_score_median": c_threshold, "T_score_median": t_threshold}
    return rows, thresholds


def _write_scores_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "gene",
        "context",
        "cell_instance_count",
        "cell_topk_mass",
        "cell_conf_raw_weighted_mean",
        "text_evidence_count",
        "text_clean_count",
        "text_noise_count",
        "text_topk_noise_fraction",
        "text_topk_attention_mass",
        "text_conf_raw_weighted_mean",
        "C_score",
        "T_score",
        "quadrant",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["quadrant"], item["gene"])):
            writer.writerow(
                {
                    key: _format_float(row[key]) if isinstance(row.get(key), float) else row.get(key, "")
                    for key in fieldnames
                }
            )


def _rank_rows(rows: list[dict[str, Any]], quadrant: str) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["_quadrant_key"] == quadrant]
    if quadrant == "high_cell_high_text":
        return sorted(selected, key=lambda row: row["C_score"] + row["T_score"], reverse=True)
    if quadrant == "high_cell_low_text":
        return sorted(selected, key=lambda row: (row["C_score"] - row["T_score"], row["C_score"]), reverse=True)
    if quadrant == "low_cell_high_text":
        return sorted(selected, key=lambda row: (row["T_score"] - row["C_score"], row["T_score"]), reverse=True)
    return sorted(selected, key=lambda row: row["C_score"] + row["T_score"])


def _write_md(
    rows: list[dict[str, Any]],
    thresholds: dict[str, float],
    output_path: Path,
    top_k: int,
    cell_conf_column: str,
    text_conf_column: str,
) -> None:
    counts = Counter(row["_quadrant_key"] for row in rows)
    lines: list[str] = []
    lines.append("# Confidence-Gap Analysis")
    lines.append("")
    lines.append("## Score Definitions")
    lines.append("")
    lines.append(
        f"- Cell-side confidence support (C_g): attention-weighted mean sigmoid(`{cell_conf_column}`) over top-{top_k} cell-side instances."
    )
    lines.append(
        f"- Text-side confidence support (T_g): attention-weighted mean sigmoid(`{text_conf_column}`) over top-{top_k} text evidence rows."
    )
    lines.append("- Quadrants are assigned with median thresholds over analyzed genes.")
    lines.append("- Conservative interpretation: these are model-side confidence support scores, not biological validation scores.")
    lines.append("")
    lines.append("## Thresholds")
    lines.append("")
    lines.append("| Threshold | Value |")
    lines.append("|---|---:|")
    for key in ("C_score_median", "T_score_median"):
        lines.append(f"| {_escape_md(key)} | {_format_float(thresholds[key])} |")
    lines.append("")
    lines.append("## Quadrant Counts")
    lines.append("")
    lines.append("| Quadrant | Count |")
    lines.append("|---|---:|")
    for key, label in QUADRANT_LABELS.items():
        lines.append(f"| {_escape_md(label)} | {counts.get(key, 0)} |")
    lines.append("")
    lines.append("## Top Genes By Quadrant")
    lines.append("")
    for key, label in QUADRANT_LABELS.items():
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| Gene | C_score | T_score | Cell raw conf | Text raw conf | Clean evidence | Top-k noise fraction |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in _rank_rows(rows, key)[:20]:
            lines.append(
                "| {gene} | {C} | {T} | {Craw} | {Traw} | {clean} | {noise} |".format(
                    gene=_escape_md(row["gene"]),
                    C=_format_float(row["C_score"]),
                    T=_format_float(row["T_score"]),
                    Craw=_format_float(row["cell_conf_raw_weighted_mean"]),
                    Traw=_format_float(row["text_conf_raw_weighted_mean"]),
                    clean=row["text_clean_count"],
                    noise=_format_float(row["text_topk_noise_fraction"]),
                )
            )
        lines.append("")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _label_offsets() -> dict[str, tuple[float, float]]:
    return {
        "APP": (-0.055, -0.020),
        "APOE": (-0.060, 0.030),
        "PSEN1": (0.020, -0.025),
        "TREM2": (0.020, 0.025),
        "MAPT": (-0.060, -0.025),
        "ABCA7": (0.020, 0.028),
        "PICALM": (0.020, -0.040),
        "CD33": (0.020, 0.020),
        "BIN1": (-0.055, 0.020),
        "INPP5D": (-0.065, -0.025),
        "MS4A6A": (0.020, 0.035),
        "CLU": (-0.050, -0.020),
        "SOD1": (0.020, -0.025),
        "HFE": (0.025, 0.030),
        "TSPO": (0.030, -0.025),
        "HSPA5": (0.020, 0.020),
        "RELN": (0.025, -0.030),
        "B2M": (0.020, -0.020),
        "TGFBI": (-0.060, -0.025),
        "GPNMB": (0.020, 0.025),
        "CXCR4": (0.020, -0.030),
    }


def _plot(rows: list[dict[str, Any]], thresholds: dict[str, float], output_dir: Path, label_genes: set[str]) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
            "figure.dpi": 150,
        }
    )
    colors = {
        "high_cell_high_text": "#4C7A5B",
        "high_cell_low_text": "#C76E3A",
        "low_cell_high_text": "#4C78A8",
        "low_cell_low_text": "#8A8A8A",
    }
    fig, ax = plt.subplots(figsize=(4.9, 3.8))
    for key, label in QUADRANT_LABELS.items():
        selected = [row for row in rows if row["_quadrant_key"] == key]
        ax.scatter(
            [row["C_score"] for row in selected],
            [row["T_score"] for row in selected],
            s=23 if key == "high_cell_low_text" else 18,
            c=colors[key],
            alpha=0.82 if key == "high_cell_low_text" else 0.68,
            edgecolors="white",
            linewidths=0.35,
            label=label,
        )
    ax.axvline(thresholds["C_score_median"], color="#555555", linestyle="--", linewidth=0.9)
    ax.axhline(thresholds["T_score_median"], color="#555555", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Cell-side confidence support")
    ax.set_ylabel("Text-side confidence support")
    max_y = max(row["T_score"] for row in rows)
    y_upper = min(1.03, max(0.42, max_y + 0.06, thresholds["T_score_median"] + 0.08))
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.02, y_upper)
    ax.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2)

    offsets = _label_offsets()
    fallback_offsets = [(0.012, 0.012), (0.012, -0.030), (-0.060, 0.016), (-0.060, -0.030)]
    labeled = [row for row in rows if row["gene"] in label_genes]
    labeled = sorted(labeled, key=lambda row: (row["_quadrant_key"], row["gene"]))
    for idx, row in enumerate(labeled):
        dx, dy = offsets.get(row["gene"], fallback_offsets[idx % len(fallback_offsets)])
        x = row["C_score"]
        y = row["T_score"]
        label_x = min(max(x + dx, 0.0), 0.98)
        label_y = min(max(y + dy, 0.0), 0.98)
        ax.annotate(
            row["gene"],
            xy=(x, y),
            xytext=(label_x, label_y),
            textcoords="data",
            fontsize=7,
            color="#222222",
            arrowprops={
                "arrowstyle": "-",
                "color": "#777777",
                "lw": 0.35,
                "shrinkA": 1.5,
                "shrinkB": 2.0,
            },
        )
    ax.text(
        0.985,
        0.055,
        "Confidence-gap\ncandidates",
        ha="right",
        va="bottom",
        fontsize=8,
        color=colors["high_cell_low_text"],
    )
    fig.tight_layout(pad=0.6)
    fig.savefig(output_dir / "confidence_gap_plot.pdf")
    fig.savefig(output_dir / "confidence_gap_plot.png")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scGATES confidence-gap analysis.")
    parser.add_argument("--cell-weights", type=Path, required=True)
    parser.add_argument("--evidence-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="eval")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--cell-conf-column", default="conf")
    parser.add_argument("--text-conf-column", default="conf")
    parser.add_argument("--label-genes", default="")
    parser.add_argument("--gene-allowlist", type=Path)
    parser.add_argument("--min-clean-evidence", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    allowlist = _read_gene_allowlist(args.gene_allowlist) if args.gene_allowlist else None
    label_genes = {gene.strip() for gene in args.label_genes.split(",") if gene.strip()}

    rows, thresholds = _compute_scores(
        cell_rows=_read_csv(args.cell_weights),
        evidence_rows=_read_csv(args.evidence_weights),
        split=args.split,
        top_k=args.top_k,
        min_clean_evidence=args.min_clean_evidence,
        cell_conf_column=args.cell_conf_column,
        text_conf_column=args.text_conf_column,
        allowlist=allowlist,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_scores_csv(rows, args.output_dir / "confidence_gap_scores.csv")
    _write_md(
        rows,
        thresholds,
        args.output_dir / "confidence_gap_quadrants.md",
        args.top_k,
        args.cell_conf_column,
        args.text_conf_column,
    )
    _plot(rows, thresholds, args.output_dir, label_genes)
    print(f"Wrote {len(rows)} genes to {args.output_dir}")
    print("Confidence-gap candidates:", ", ".join(row["gene"] for row in _rank_rows(rows, "high_cell_low_text")[:10]))


if __name__ == "__main__":
    main()

