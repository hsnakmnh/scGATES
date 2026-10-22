from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping

import torch
from torch.nn import functional as F


def load_confusion_json(path: str | Path) -> dict[str, list[str]]:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[str, list[str]] = {}
    for gene, negatives in raw.items():
        cleaned: list[str] = []
        seen: set[str] = set()
        for neg in negatives:
            neg = str(neg).strip()
            if not neg:
                continue
            key = neg.upper()
            if key == str(gene).upper() or key in seen:
                continue
            seen.add(key)
            cleaned.append(neg)
        if cleaned:
            out[str(gene)] = cleaned
    return out


def build_confusion_map_from_predictions(
    predictions_csv: str | Path,
    max_negatives: int = 4,
    min_count: int = 1,
    use_diag_errors: bool = False,
) -> dict[str, list[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    with Path(predictions_csv).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gene = str(row.get("gene", "")).strip()
            top1 = str(row.get("top1_gene", "")).strip()
            if not gene or not top1 or top1 == gene:
                continue
            try:
                gold_rank = int(float(row.get("gold_rank", "0")))
            except Exception:
                gold_rank = 0
            try:
                diag_rank = int(float(row.get("diag_rank", "0")))
            except Exception:
                diag_rank = 0
            if gold_rank <= 1 and not (use_diag_errors and diag_rank > 1):
                continue
            counts[gene][top1] += 1
    out: dict[str, list[str]] = {}
    for gene, counter in counts.items():
        selected = [neg for neg, count in counter.most_common() if count >= min_count and neg != gene]
        if max_negatives > 0:
            selected = selected[:max_negatives]
        if selected:
            out[gene] = selected
    return out


def _zero_confusion_metrics(
    zero: torch.Tensor,
    prefixes: tuple[str, ...] = ("prior", "final"),
) -> dict[str, torch.Tensor]:
    metrics = {
        "loss_stage3_confusion": zero.detach(),
        "stage3_confusion_rows": zero.detach(),
        "stage3_confusion_acc": zero.detach(),
        "stage3_confusion_gap": zero.detach(),
    }
    for prefix in prefixes:
        metrics.update(
            {
                f"loss_stage3_confusion_{prefix}": zero.detach(),
                f"stage3_confusion_{prefix}_rows": zero.detach(),
                f"stage3_confusion_{prefix}_acc": zero.detach(),
                f"stage3_confusion_{prefix}_gap": zero.detach(),
            }
        )
    return metrics


def _single_anchor_confusion_loss(
    anchor: torch.Tensor,
    text: torch.Tensor,
    neg_text: torch.Tensor,
    mask: torch.Tensor,
    margin: float,
    mode: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = anchor.new_tensor(0.0)
    valid = mask.any(dim=1)
    if not bool(valid.any()):
        return zero, {
            "loss": zero.detach(),
            "rows": zero.detach(),
            "acc": zero.detach(),
            "gap": zero.detach(),
        }

    anchor = F.normalize(anchor, dim=-1)
    text = F.normalize(text, dim=-1)
    neg_text = F.normalize(neg_text, dim=-1)
    pos = torch.sum(anchor * text, dim=-1, keepdim=True)
    neg = torch.sum(anchor.unsqueeze(1) * neg_text, dim=-1).masked_fill(~mask, -1e9)
    hardest = neg.amax(dim=1, keepdim=True)

    if mode == "softplus":
        raw_loss = F.softplus(hardest[valid] - pos[valid] + float(margin))
    else:
        raw_loss = F.relu(float(margin) + hardest[valid] - pos[valid])
    loss = raw_loss.mean()
    gap = (pos[valid] - hardest[valid]).mean()
    acc = (pos[valid] > hardest[valid]).to(anchor.dtype).mean()
    return loss, {
        "loss": loss.detach(),
        "rows": valid.to(anchor.dtype).sum().detach(),
        "acc": acc.detach(),
        "gap": gap.detach(),
    }


def stage3_confusion_loss(
    out: Mapping[str, torch.Tensor | dict[str, torch.Tensor]],
    lambda_confusion: float,
    margin: float,
    mode: str = "margin",
    anchor_mode: str = "prior",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cell = out["cell_set"]
    assert isinstance(cell, torch.Tensor)
    zero = cell.new_tensor(0.0)
    if lambda_confusion <= 0 or "confusion_text_set" not in out or "confusion_mask" not in out:
        return zero, _zero_confusion_metrics(zero)

    text = out["text_set"]
    neg_text = out["confusion_text_set"]
    mask = out["confusion_mask"]
    assert isinstance(text, torch.Tensor)
    assert isinstance(neg_text, torch.Tensor)
    assert isinstance(mask, torch.Tensor)

    anchors: list[tuple[str, torch.Tensor]] = []
    prior = out.get("prior_cell_set")
    if anchor_mode in {"prior", "both"}:
        if isinstance(prior, torch.Tensor):
            anchors.append(("prior", prior))
        elif anchor_mode == "prior":
            anchors.append(("prior", cell))
    if anchor_mode in {"final", "both"}:
        anchors.append(("final", cell))
    if not anchors:
        raise ValueError(f"Unsupported stage3 confusion anchor mode: {anchor_mode}")

    losses: list[torch.Tensor] = []
    metrics: dict[str, torch.Tensor] = {}
    for name, anchor in anchors:
        raw_loss, part = _single_anchor_confusion_loss(
            anchor=anchor,
            text=text,
            neg_text=neg_text,
            mask=mask,
            margin=margin,
            mode=mode,
        )
        losses.append(raw_loss)
        metrics[f"loss_stage3_confusion_{name}"] = part["loss"]
        metrics[f"stage3_confusion_{name}_rows"] = part["rows"]
        metrics[f"stage3_confusion_{name}_acc"] = part["acc"]
        metrics[f"stage3_confusion_{name}_gap"] = part["gap"]

    loss = torch.stack(losses).mean()
    metrics["loss_stage3_confusion"] = loss.detach()
    metrics["stage3_confusion_rows"] = torch.stack(
        [metrics[f"stage3_confusion_{name}_rows"] for name, _ in anchors]
    ).mean()
    metrics["stage3_confusion_acc"] = torch.stack(
        [metrics[f"stage3_confusion_{name}_acc"] for name, _ in anchors]
    ).mean()
    metrics["stage3_confusion_gap"] = torch.stack(
        [metrics[f"stage3_confusion_{name}_gap"] for name, _ in anchors]
    ).mean()
    return float(lambda_confusion) * loss, metrics

