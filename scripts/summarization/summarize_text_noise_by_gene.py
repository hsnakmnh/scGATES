from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


GROUP_KEYS = ["gene", "context", "split"]

NUMERIC_FIELDS = [
    "weight",
    "conf",
    "conf_logit",
    "attn_logit",
    "prior",
    "content_logit",
]

OUTPUT_FIELDS = [
    "gene",
    "context",
    "split",
    "total_count",
    "clean_count",
    "noise_count",
    "noise_fraction",
    "clean_weight_mass",
    "noise_weight_mass",
    "clean_mean_weight",
    "noise_mean_weight",
    "clean_median_weight",
    "noise_median_weight",
    "clean_mean_conf",
    "noise_mean_conf",
    "delta_conf",
    "clean_mean_conf_logit",
    "noise_mean_conf_logit",
    "delta_conf_logit",
    "clean_mean_attn_logit",
    "noise_mean_attn_logit",
    "delta_attn_logit",
    "clean_mean_prior",
    "noise_mean_prior",
    "delta_prior",
    "clean_mean_content_logit",
    "noise_mean_content_logit",
    "delta_content_logit",
    "noise_weight_expected",
    "noise_weight_excess",
]

WORST_FIELDS = [
    "gene",
    "context",
    "split",
    "clean_count",
    "noise_count",
    "noise_fraction",
    "clean_weight_mass",
    "noise_weight_mass",
    "noise_weight_expected",
    "noise_weight_excess",
    "delta_conf",
    "delta_conf_logit",
    "delta_attn_logit",
]

STRATA_FIELDS = [
    "clean_count_bin",
    "num_gene_groups",
    "mean_clean_count",
    "mean_noise_count",
    "mean_noise_fraction",
    "mean_noise_weight_mass",
    "mean_noise_weight_expected",
    "mean_noise_weight_excess",
    "median_noise_weight_excess",
    "mean_delta_conf",
    "median_delta_conf",
    "mean_delta_attn_logit",
    "median_delta_attn_logit",
    "fraction_delta_conf_positive",
    "fraction_noise_weight_excess_negative",
    "fraction_good",
]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _values(rows: list[dict[str, str]], field: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        value = _to_float(row.get(field))
        if value is not None:
            out.append(value)
    return out


def _sum_field(rows: list[dict[str, str]], field: str) -> float:
    return sum(_values(rows, field))


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return "" if value is None else str(value)


def _md_cell(value: Any) -> str:
    text = _fmt(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def summarize_by_gene(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in GROUP_KEYS)
        grouped[key].append(row)

    out: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        gene, context, split = key
        clean_rows = [row for row in group if str(row.get("is_noise", "0")) != "1"]
        noise_rows = [row for row in group if str(row.get("is_noise", "0")) == "1"]
        total_count = len(group)
        clean_count = len(clean_rows)
        noise_count = len(noise_rows)
        noise_fraction = (noise_count / total_count) if total_count else None
        clean_weight_mass = _sum_field(clean_rows, "weight")
        noise_weight_mass = _sum_field(noise_rows, "weight")
        noise_weight_expected = noise_fraction
        noise_weight_excess = (
            noise_weight_mass - noise_weight_expected
            if noise_weight_expected is not None and noise_count > 0
            else None
        )

        record: dict[str, Any] = {
            "gene": gene,
            "context": context,
            "split": split,
            "total_count": total_count,
            "clean_count": clean_count,
            "noise_count": noise_count,
            "noise_fraction": noise_fraction,
            "clean_weight_mass": clean_weight_mass,
            "noise_weight_mass": noise_weight_mass,
            "noise_weight_expected": noise_weight_expected,
            "noise_weight_excess": noise_weight_excess,
        }

        for field in NUMERIC_FIELDS:
            clean_values = _values(clean_rows, field)
            noise_values = _values(noise_rows, field)
            record[f"clean_mean_{field}"] = _mean(clean_values)
            record[f"noise_mean_{field}"] = _mean(noise_values)
            if field == "weight":
                record["clean_median_weight"] = _median(clean_values)
                record["noise_median_weight"] = _median(noise_values)
            if clean_values and noise_values:
                record[f"delta_{field}"] = _mean(clean_values) - _mean(noise_values)  # type: ignore[operator]
            else:
                record[f"delta_{field}"] = None

        out.append(record)
    return out


def global_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    comparable = [
        row
        for row in records
        if int(row.get("clean_count", 0) or 0) > 0 and int(row.get("noise_count", 0) or 0) > 0
    ]

    def collect(field: str) -> list[float]:
        values: list[float] = []
        for row in comparable:
            value = _to_float(row.get(field))
            if value is not None:
                values.append(value)
        return values

    delta_conf = collect("delta_conf")
    noise_weight_excess = collect("noise_weight_excess")
    good = [
        row
        for row in comparable
        if (_to_float(row.get("delta_conf")) is not None and _to_float(row.get("delta_conf")) > 0)
        and (
            _to_float(row.get("noise_weight_excess")) is not None
            and _to_float(row.get("noise_weight_excess")) < 0
        )
    ]

    n = len(comparable)
    return {
        "num_gene_groups": len(records),
        "num_comparable_gene_groups": n,
        "mean_noise_fraction": _mean(collect("noise_fraction")),
        "mean_clean_weight_mass": _mean(collect("clean_weight_mass")),
        "mean_noise_weight_mass": _mean(collect("noise_weight_mass")),
        "mean_noise_weight_expected": _mean(collect("noise_weight_expected")),
        "mean_noise_weight_excess": _mean(noise_weight_excess),
        "median_noise_weight_excess": _median(noise_weight_excess),
        "mean_delta_conf": _mean(delta_conf),
        "median_delta_conf": _median(delta_conf),
        "mean_delta_conf_logit": _mean(collect("delta_conf_logit")),
        "median_delta_conf_logit": _median(collect("delta_conf_logit")),
        "mean_delta_attn_logit": _mean(collect("delta_attn_logit")),
        "median_delta_attn_logit": _median(collect("delta_attn_logit")),
        "fraction_delta_conf_positive": (
            sum(1 for value in delta_conf if value > 0) / n if n else None
        ),
        "fraction_noise_weight_excess_negative": (
            sum(1 for value in noise_weight_excess if value < 0) / n if n else None
        ),
        "fraction_good": len(good) / n if n else None,
    }


def clean_count_bin(clean_count: int) -> str:
    if clean_count <= 1:
        return "1"
    if clean_count <= 3:
        return "2-3"
    if clean_count <= 7:
        return "4-7"
    if clean_count <= 15:
        return "8-15"
    return ">=16"


def stratified_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        clean_count = int(row.get("clean_count", 0) or 0)
        noise_count = int(row.get("noise_count", 0) or 0)
        if clean_count <= 0 or noise_count <= 0:
            continue
        grouped[clean_count_bin(clean_count)].append(row)

    def collect(group: list[dict[str, Any]], field: str) -> list[float]:
        values: list[float] = []
        for row in group:
            value = _to_float(row.get(field))
            if value is not None:
                values.append(value)
        return values

    rows: list[dict[str, Any]] = []
    for bin_name in ["1", "2-3", "4-7", "8-15", ">=16"]:
        group = grouped.get(bin_name, [])
        if not group:
            continue
        delta_conf = collect(group, "delta_conf")
        noise_weight_excess = collect(group, "noise_weight_excess")
        good = [
            row
            for row in group
            if (_to_float(row.get("delta_conf")) is not None and _to_float(row.get("delta_conf")) > 0)
            and (
                _to_float(row.get("noise_weight_excess")) is not None
                and _to_float(row.get("noise_weight_excess")) < 0
            )
        ]
        n = len(group)
        rows.append(
            {
                "clean_count_bin": bin_name,
                "num_gene_groups": n,
                "mean_clean_count": _mean(collect(group, "clean_count")),
                "mean_noise_count": _mean(collect(group, "noise_count")),
                "mean_noise_fraction": _mean(collect(group, "noise_fraction")),
                "mean_noise_weight_mass": _mean(collect(group, "noise_weight_mass")),
                "mean_noise_weight_expected": _mean(collect(group, "noise_weight_expected")),
                "mean_noise_weight_excess": _mean(noise_weight_excess),
                "median_noise_weight_excess": _median(noise_weight_excess),
                "mean_delta_conf": _mean(delta_conf),
                "median_delta_conf": _median(delta_conf),
                "mean_delta_attn_logit": _mean(collect(group, "delta_attn_logit")),
                "median_delta_attn_logit": _median(collect(group, "delta_attn_logit")),
                "fraction_delta_conf_positive": (
                    sum(1 for value in delta_conf if value > 0) / n if n else None
                ),
                "fraction_noise_weight_excess_negative": (
                    sum(1 for value in noise_weight_excess if value < 0) / n if n else None
                ),
                "fraction_good": len(good) / n if n else None,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field)) for field in fields})


def _write_table(f: Any, rows: list[dict[str, Any]], fields: list[str]) -> None:
    f.write("| " + " | ".join(fields) + " |\n")
    f.write("| " + " | ".join(["---"] * len(fields)) + " |\n")
    for row in rows:
        f.write("| " + " | ".join(_md_cell(row.get(field)) for field in fields) + " |\n")


def write_summary_md(
    path: Path,
    summary: dict[str, Any],
    strata_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Text Noise By Gene Summary\n\n")
        f.write("## Global Summary\n\n")
        _write_table(f, [summary], list(summary.keys()))
        f.write("\n## Clean Count Strata Summary\n\n")
        _write_table(f, strata_rows, STRATA_FIELDS)
        f.write("\n## Per-Gene Summary\n\n")
        _write_table(f, rows, OUTPUT_FIELDS)


def write_worst_md(path: Path, rows: list[dict[str, Any]], top_k: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    comparable = [
        row
        for row in rows
        if int(row.get("clean_count", 0) or 0) > 0 and int(row.get("noise_count", 0) or 0) > 0
    ]
    by_excess = sorted(
        comparable,
        key=lambda row: _to_float(row.get("noise_weight_excess")) if _to_float(row.get("noise_weight_excess")) is not None else -1e18,
        reverse=True,
    )[:top_k]
    by_delta_conf = sorted(
        comparable,
        key=lambda row: _to_float(row.get("delta_conf")) if _to_float(row.get("delta_conf")) is not None else 1e18,
    )[:top_k]
    both_bad = [
        row
        for row in comparable
        if (_to_float(row.get("noise_weight_excess")) is not None and _to_float(row.get("noise_weight_excess")) > 0)
        and (_to_float(row.get("delta_conf")) is not None and _to_float(row.get("delta_conf")) <= 0)
    ][:top_k]

    with path.open("w", encoding="utf-8") as f:
        f.write("# Text Noise Worst Gene Groups\n\n")
        f.write(
            "Use these genes as indices into `evidence_weights.by_gene.md` to inspect the actual clean/noise text evidence.\n\n"
        )
        f.write("## Largest Noise Weight Excess\n\n")
        _write_table(f, by_excess, WORST_FIELDS)
        f.write("\n## Lowest Delta Conf\n\n")
        _write_table(f, by_delta_conf, WORST_FIELDS)
        f.write("\n## Positive Noise Excess And Nonpositive Delta Conf\n\n")
        _write_table(f, both_bad, WORST_FIELDS)


def print_summary(summary: dict[str, Any], strata_rows: list[dict[str, Any]]) -> None:
    fields = list(summary.keys())
    print("| " + " | ".join(fields) + " |")
    print("| " + " | ".join(["---"] * len(fields)) + " |")
    print("| " + " | ".join(_md_cell(summary.get(field)) for field in fields) + " |")
    print()
    print("| " + " | ".join(STRATA_FIELDS) + " |")
    print("| " + " | ".join(["---"] * len(STRATA_FIELDS)) + " |")
    for row in strata_rows:
        print("| " + " | ".join(_md_cell(row.get(field)) for field in STRATA_FIELDS) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize clean/noise text evidence weights per gene from evidence_weights.csv."
    )
    parser.add_argument("evidence_weights_csv", help="Path to evidence_weights.csv")
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--output-md", default="")
    parser.add_argument("--output-worst-md", default="")
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args()

    rows = _read_rows(Path(args.evidence_weights_csv))
    per_gene = summarize_by_gene(rows)
    summary = global_summary(per_gene)
    strata_rows = stratified_summary(per_gene)

    if args.output_csv:
        write_csv(Path(args.output_csv), per_gene, OUTPUT_FIELDS)
    if args.output_md:
        write_summary_md(Path(args.output_md), summary, strata_rows, per_gene)
    if args.output_worst_md:
        write_worst_md(Path(args.output_worst_md), per_gene, top_k=args.top_k)

    print_summary(summary, strata_rows)


if __name__ == "__main__":
    main()

