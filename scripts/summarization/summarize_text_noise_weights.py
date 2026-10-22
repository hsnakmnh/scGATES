from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


NUMERIC_FIELDS = [
    "weight",
    "prior",
    "conf",
    "content_logit",
    "prior_logit",
    "conf_logit",
    "attn_logit",
]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _numeric_values(group: list[dict[str, str]], field: str) -> list[float]:
    values: list[float] = []
    for row in group:
        value = row.get(field, "")
        if value == "":
            continue
        try:
            values.append(float(value))
        except ValueError:
            continue
    return values


def summarize(path: Path) -> list[dict[str, Any]]:
    rows = _read_rows(path)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("is_noise", "0"))].append(row)

    out: list[dict[str, Any]] = []
    for is_noise, group in sorted(groups.items(), key=lambda item: item[0]):
        record: dict[str, Any] = {
            "file": str(path),
            "is_noise": is_noise,
            "n": len(group),
            "unique_target_genes": len({row.get("gene", "") for row in group}),
            "unique_source_genes": len({row.get("source_gene", "") for row in group}),
        }
        for field in NUMERIC_FIELDS:
            values = _numeric_values(group, field)
            record[f"mean_{field}"] = _mean(values) if values else ""
            record[f"median_{field}"] = _median(values) if values else ""
        out.append(record)
    clean = groups.get("0", [])
    noise = groups.get("1", [])
    if clean and noise:
        record = {
            "file": str(path),
            "is_noise": "clean_minus_noise",
            "n": "",
            "unique_target_genes": "",
            "unique_source_genes": "",
        }
        for field in NUMERIC_FIELDS:
            clean_values = _numeric_values(clean, field)
            noise_values = _numeric_values(noise, field)
            if clean_values and noise_values:
                record[f"delta_{field}"] = _mean(clean_values) - _mean(noise_values)
        out.append(record)
    return out


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_md(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fields)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(_fmt(row.get(field, "")).replace("|", "\\|") for field in fields) + " |\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize clean/noise text evidence weights and conf values.")
    parser.add_argument("csv_files", nargs="+", help="evidence_weights.csv or train_evidence_weights.csv files")
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for csv_file in args.csv_files:
        rows.extend(summarize(Path(csv_file)))

    fields = [
        "file",
        "is_noise",
        "n",
        "unique_target_genes",
        "unique_source_genes",
        *[f"mean_{field}" for field in NUMERIC_FIELDS],
        *[f"median_{field}" for field in NUMERIC_FIELDS],
        "delta_weight",
        "delta_conf",
        "delta_conf_logit",
        "delta_attn_logit",
    ]

    if args.output_csv:
        write_csv(Path(args.output_csv), rows, fields)
    if args.output_md:
        write_md(Path(args.output_md), rows, fields)

    print("| " + " | ".join(fields) + " |")
    print("| " + " | ".join(["---"] * len(fields)) + " |")
    for row in rows:
        print("| " + " | ".join(_fmt(row.get(field, "")) for field in fields) + " |")


if __name__ == "__main__":
    main()

