from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


QUADRANT_LABELS = {
    "high_cell_high_text": "Well-supported aligned genes",
    "high_cell_low_text": "Evidence-gap candidates",
    "low_cell_high_text": "Text-heavy genes",
    "low_cell_low_text": "Low-support genes",
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


def _minmax(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if math.isclose(low, high):
        return {key: 0.5 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def _entropy(weights: list[float]) -> float:
    total = sum(max(weight, 0.0) for weight in weights)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for weight in weights:
        prob = max(weight, 0.0) / total
        if prob > 0:
            entropy -= prob * math.log(prob)
    return entropy


def _read_gene_allowlist(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Gene allowlist not found: {path}")
    genes: set[str] = set()
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "gene" in reader.fieldnames:
                for row in reader:
                    gene = str(row.get("gene", "")).strip()
                    if gene:
                        genes.add(gene)
                return genes
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.reader(f):
                if row and row[0].strip() and row[0].strip().lower() != "gene":
                    genes.add(row[0].strip())
        return genes
    for line in path.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if not clean or clean.startswith("#"):
            continue
        genes.add(clean.split(",")[0].split()[0])
    return genes


def _choose_text_support_column(rows: list[dict[str, str]]) -> str:
    for column in ("gated_prior", "prior", "conf"):
        if column not in rows[0]:
            continue
        values = [_as_float(row.get(column), default=float("nan")) for row in rows[:1000]]
        if any(not math.isnan(value) for value in values):
            return column
    raise ValueError("No usable text support column found. Expected gated_prior, prior, or conf.")


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


def _compute_scores(
    cell_rows: list[dict[str, str]],
    evidence_rows: list[dict[str, str]],
    split: str,
    top_k: int,
    min_clean_evidence: int,
    allowlist: set[str] | None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    cell_eval = [row for row in cell_rows if str(row.get("split", "")).strip() == split]
    evidence_eval = [row for row in evidence_rows if str(row.get("split", "")).strip() == split]
    if not cell_eval:
        raise ValueError(f"No cell rows found for split={split!r}")
    if not evidence_eval:
        raise ValueError(f"No evidence rows found for split={split!r}")

    support_column = _choose_text_support_column(evidence_eval)
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

        c_ranked = sorted(c_rows, key=lambda row: _as_float(row.get("weight")), reverse=True)
        c_top = c_ranked[:top_k]
        c_weights = [_as_float(row.get("weight")) for row in c_rows]
        cell_topk_mass = sum(_as_float(row.get("weight")) for row in c_top)
        cell_entropy = _entropy(c_weights)

        e_ranked = sorted(e_rows, key=lambda row: _as_float(row.get("weight")), reverse=True)
        e_top = e_ranked[:top_k]
        top_support_values = [_as_float(row.get(support_column)) for row in e_top]
        top_support_mean = sum(top_support_values) / len(top_support_values) if top_support_values else 0.0
        text_topk_attention_mass = sum(_as_float(row.get("weight")) for row in e_top)
        text_noise_count = sum(1 for row in e_rows if _is_noise(row))
        text_topk_noise_fraction = sum(1 for row in e_top if _is_noise(row)) / len(e_top) if e_top else 0.0

        context = _most_common_context(c_rows) or _most_common_context(e_rows)
        i_raw = cell_topk_mass
        e_raw = top_support_mean * math.log1p(len(clean_rows))
        rows.append(
            {
                "gene": gene,
                "context": context,
                "cell_instance_count": len(c_rows),
                "cell_topk_mass": cell_topk_mass,
                "cell_weight_entropy": cell_entropy,
                "text_evidence_count": len(e_rows),
                "text_clean_count": len(clean_rows),
                "text_noise_count": text_noise_count,
                "text_topk_noise_fraction": text_topk_noise_fraction,
                "text_topk_gated_prior_mean": top_support_mean,
                "text_topk_attention_mass": text_topk_attention_mass,
                "I_raw": i_raw,
                "E_raw": e_raw,
                "_support_column": support_column,
            }
        )

    if not rows:
        raise ValueError("No genes remained after min-clean-evidence filtering.")

    i_norm = _minmax({row["gene"]: row["I_raw"] for row in rows})
    e_norm = _minmax({row["gene"]: row["E_raw"] for row in rows})
    for row in rows:
        row["I_norm"] = i_norm[row["gene"]]
        row["E_norm"] = e_norm[row["gene"]]

    i_threshold = _median([row["I_norm"] for row in rows])
    e_threshold = _median([row["E_norm"] for row in rows])
    for row in rows:
        high_i = row["I_norm"] >= i_threshold
        high_e = row["E_norm"] >= e_threshold
        if high_i and high_e:
            quadrant = "high_cell_high_text"
        elif high_i and not high_e:
            quadrant = "high_cell_low_text"
        elif not high_i and high_e:
            quadrant = "low_cell_high_text"
        else:
            quadrant = "low_cell_low_text"
        row["quadrant"] = QUADRANT_LABELS[quadrant]
        row["_quadrant_key"] = quadrant

    thresholds = {
        "I_norm_median": i_threshold,
        "E_norm_median": e_threshold,
        "I_raw_median": _median([row["I_raw"] for row in rows]),
        "E_raw_median": _median([row["E_raw"] for row in rows]),
    }
    return rows, thresholds


def _write_scores_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "gene",
        "context",
        "cell_instance_count",
        "cell_topk_mass",
        "cell_weight_entropy",
        "text_evidence_count",
        "text_clean_count",
        "text_noise_count",
        "text_topk_noise_fraction",
        "text_topk_gated_prior_mean",
        "text_topk_attention_mass",
        "I_raw",
        "E_raw",
        "I_norm",
        "E_norm",
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


def _rank_rows(rows: list[dict[str, Any]], quadrant: str | None = None) -> list[dict[str, Any]]:
    selected = rows if quadrant is None else [row for row in rows if row["_quadrant_key"] == quadrant]
    return sorted(selected, key=lambda row: (row["I_norm"] - row["E_norm"], row["I_norm"]), reverse=True)


def _write_md(rows: list[dict[str, Any]], thresholds: dict[str, float], output_path: Path, top_k: int) -> None:
    support_column = rows[0]["_support_column"]
    counts = Counter(row["_quadrant_key"] for row in rows)
    lines: list[str] = []
    lines.append("# Evidence-Gap Analysis")
    lines.append("")
    lines.append("## Score Definitions")
    lines.append("")
    lines.append(
        f"- Cell-side distribution consistency (I_g): top-{top_k} attention mass over cell-side instances for each gene."
    )
    lines.append(
        f"- Text-side evidence support (E_g): mean top-{top_k} `{support_column}` among text evidence, multiplied by log(1 + clean evidence count)."
    )
    lines.append("- Both scores are min-max normalized across analyzed genes before quadrant assignment.")
    lines.append(
        "- Conservative interpretation: these are scGATES model-side support statistics for evidence prioritization, not biological truth scores."
    )
    lines.append("")
    lines.append("## Thresholds")
    lines.append("")
    lines.append("| Threshold | Value |")
    lines.append("|---|---:|")
    for key in ("I_raw_median", "E_raw_median", "I_norm_median", "E_norm_median"):
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
        lines.append("| Gene | I_norm | E_norm | I_raw | E_raw | Clean evidence | Top-k noise fraction |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        if key == "high_cell_low_text":
            ranked = _rank_rows(rows, key)
        elif key == "low_cell_high_text":
            ranked = sorted(
                [row for row in rows if row["_quadrant_key"] == key],
                key=lambda row: (row["E_norm"] - row["I_norm"], row["E_norm"]),
                reverse=True,
            )
        else:
            ranked = sorted(
                [row for row in rows if row["_quadrant_key"] == key],
                key=lambda row: row["I_norm"] + row["E_norm"],
                reverse=True,
            )
        for row in ranked[:15]:
            lines.append(
                "| {gene} | {I_norm} | {E_norm} | {I_raw} | {E_raw} | {clean} | {noise} |".format(
                    gene=_escape_md(row["gene"]),
                    I_norm=_format_float(row["I_norm"]),
                    E_norm=_format_float(row["E_norm"]),
                    I_raw=_format_float(row["I_raw"]),
                    E_raw=_format_float(row["E_raw"]),
                    clean=row["text_clean_count"],
                    noise=_format_float(row["text_topk_noise_fraction"]),
                )
            )
        lines.append("")
    lines.append("## Evidence-Gap Candidates")
    lines.append("")
    lines.append(
        "Genes in this quadrant have stronger cell-side distribution consistency than text-side evidence support under the current scGATES run. They should be interpreted as candidates for follow-up evidence review."
    )
    lines.append("")
    lines.append("| Gene | Context | I_norm | E_norm | Gap | Cell top-k mass | Clean evidence | Text top-k support |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in _rank_rows(rows, "high_cell_low_text")[:25]:
        gap = row["I_norm"] - row["E_norm"]
        lines.append(
            "| {gene} | {context} | {I_norm} | {E_norm} | {gap} | {cell} | {clean} | {support} |".format(
                gene=_escape_md(row["gene"]),
                context=_escape_md(row["context"]),
                I_norm=_format_float(row["I_norm"]),
                E_norm=_format_float(row["E_norm"]),
                gap=_format_float(gap),
                cell=_format_float(row["cell_topk_mass"]),
                clean=row["text_clean_count"],
                support=_format_float(row["text_topk_gated_prior_mean"]),
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
            [row["I_norm"] for row in selected],
            [row["E_norm"] for row in selected],
            s=24 if key == "high_cell_low_text" else 18,
            c=colors[key],
            alpha=0.82 if key == "high_cell_low_text" else 0.68,
            edgecolors="white",
            linewidths=0.35,
            label=label,
        )
    ax.axvline(thresholds["I_norm_median"], color="#555555", linestyle="--", linewidth=0.9)
    ax.axhline(thresholds["E_norm_median"], color="#555555", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Cell-side distribution consistency")
    ax.set_ylabel("Text-side evidence support")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2)

    focus_offsets = {
        "ABCA7": (-0.065, -0.065),
        "BIN1": (-0.070, 0.035),
        "PICALM": (0.050, -0.070),
        "INPP5D": (-0.085, 0.075),
        "MS4A6A": (0.018, 0.050),
        "CD33": (0.025, 0.025),
        "PSEN1": (0.014, -0.022),
        "TREM2": (-0.055, -0.035),
        "SOD1": (0.015, -0.030),
        "HFE": (0.050, 0.055),
        "TSPO": (0.050, -0.010),
        "RELN": (0.045, 0.025),
        "HSPA5": (0.015, 0.025),
        "IGF1": (0.045, -0.060),
        "TFAM": (0.035, 0.030),
        "GPX4": (-0.085, -0.030),
        "HMOX1": (-0.085, 0.030),
        "CASP1": (0.035, -0.035),
        "GRN": (0.025, -0.030),
        "MAPT": (-0.060, 0.035),
        "TGFBI": (-0.085, -0.020),
        "GPNMB": (0.020, 0.025),
        "SCO2": (0.025, -0.035),
        "CXCR4": (0.025, 0.020),
        "BCL11B": (0.025, -0.050),
        "B2M": (0.045, -0.015),
        "ERBB4": (0.038, -0.060),
        "PKD1": (0.080, -0.065),
        "PTPN2": (0.045, -0.100),
        "ACVR1": (0.050, -0.140),
        "APP": (-0.065, -0.020),
        "APOE": (-0.065, 0.030),
        "TARDBP": (0.020, -0.030),
        "SNCA": (0.030, 0.030),
    }
    fallback_offsets = [
        (0.014, 0.014),
        (0.014, -0.032),
        (-0.060, 0.016),
        (-0.060, -0.032),
        (0.020, 0.040),
        (-0.080, 0.040),
    ]
    labeled = sorted(
        [row for row in rows if row["gene"] in label_genes],
        key=lambda row: (row["_quadrant_key"] != "high_cell_low_text", row["gene"]),
    )
    for idx, row in enumerate(labeled):
        dx, dy = focus_offsets.get(row["gene"], fallback_offsets[idx % len(fallback_offsets)])
        x = row["I_norm"]
        y = row["E_norm"]
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
        "Evidence-gap\ncandidates",
        ha="right",
        va="bottom",
        fontsize=8,
        color=colors["high_cell_low_text"],
    )
    fig.tight_layout(pad=0.6)
    fig.savefig(output_dir / "evidence_gap_plot.pdf")
    fig.savefig(output_dir / "evidence_gap_plot.png")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scGATES evidence-gap analysis.")
    parser.add_argument("--cell-weights", type=Path, required=True)
    parser.add_argument("--evidence-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="eval")
    parser.add_argument("--top-k", type=int, default=10)
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

    cell_rows = _read_csv(args.cell_weights)
    evidence_rows = _read_csv(args.evidence_weights)
    rows, thresholds = _compute_scores(
        cell_rows=cell_rows,
        evidence_rows=evidence_rows,
        split=args.split,
        top_k=args.top_k,
        min_clean_evidence=args.min_clean_evidence,
        allowlist=allowlist,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_scores_csv(rows, args.output_dir / "evidence_gap_scores.csv")
    _write_md(rows, thresholds, args.output_dir / "evidence_gap_quadrants.md", args.top_k)
    _plot(rows, thresholds, args.output_dir, label_genes)
    print(f"Wrote {len(rows)} genes to {args.output_dir}")
    print(
        "Evidence-gap candidates:",
        ", ".join(row["gene"] for row in _rank_rows(rows, "high_cell_low_text")[:10]),
    )


if __name__ == "__main__":
    main()

