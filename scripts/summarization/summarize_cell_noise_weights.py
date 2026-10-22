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
    "instance_score_logit",
    "attn_logit",
    "instance_score",
    "rank",
    "expr",
]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


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


def _group_name(value: str) -> str:
    return "noise" if str(value).strip() == "1" else "clean"


def summarize(path: Path) -> list[dict[str, Any]]:
    rows = _read_rows(path)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[_group_name(row.get("is_cell_noise", "0"))].append(row)

    out: list[dict[str, Any]] = []
    group_records: dict[str, dict[str, Any]] = {}
    for group_name in ("clean", "noise"):
        group_rows = groups.get(group_name, [])
        record: dict[str, Any] = {
            "file": str(path),
            "group": group_name,
            "n": len(group_rows),
            "unique_target_genes": len({row.get("target_gene") or row.get("gene", "") for row in group_rows}),
            "unique_source_genes": len({row.get("source_gene", "") for row in group_rows}),
        }
        for field in NUMERIC_FIELDS:
            values = [value for row in group_rows if (value := _float(row, field)) is not None]
            record[f"mean_{field}"] = _mean(values) if values else ""
            record[f"median_{field}"] = _median(values) if values else ""
        out.append(record)
        group_records[group_name] = record

    clean = group_records.get("clean", {})
    noise = group_records.get("noise", {})
    delta: dict[str, Any] = {
        "file": str(path),
        "group": "clean_minus_noise",
        "n": "",
        "unique_target_genes": "",
        "unique_source_genes": "",
    }
    for field in NUMERIC_FIELDS:
        c = clean.get(f"mean_{field}", "")
        n = noise.get(f"mean_{field}", "")
        delta[f"mean_{field}"] = float(c) - float(n) if c != "" and n != "" else ""
        delta[f"median_{field}"] = ""
    out.append(delta)
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
            f.write("| " + " | ".join(_fmt(row.get(field, "")) for field in fields) + " |\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize clean vs synthetic cell-noise weights.")
    parser.add_argument("csv_files", nargs="+", help="cell_weights.csv or train_cell_weights.csv files")
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for csv_file in args.csv_files:
        rows.extend(summarize(Path(csv_file)))

    fields = [
        "file",
        "group",
        "n",
        "unique_target_genes",
        "unique_source_genes",
        *[f"mean_{field}" for field in NUMERIC_FIELDS],
        *[f"median_{field}" for field in NUMERIC_FIELDS],
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

