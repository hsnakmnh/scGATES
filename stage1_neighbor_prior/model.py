from __future__ import annotations

import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, proj_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeighborSetEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        proj_dim: int = 256,
        nhead: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )
        if num_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=max(hidden_dim * 2, 128),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            self.encoder = nn.Identity()
        self.attn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, neighbor_embeddings: torch.Tensor, neighbor_mask: torch.Tensor) -> torch.Tensor:
        if neighbor_embeddings.ndim != 3:
            raise ValueError(f"Expected neighbor embeddings (B, K, D), got {tuple(neighbor_embeddings.shape)}")
        if neighbor_mask.ndim != 2:
            raise ValueError(f"Expected neighbor mask (B, K), got {tuple(neighbor_mask.shape)}")
        hidden = self.input_proj(neighbor_embeddings)
        if not isinstance(self.encoder, nn.Identity):
            hidden = self.encoder(hidden, src_key_padding_mask=~neighbor_mask)
        else:
            hidden = self.encoder(hidden)
        logits = self.attn(hidden).squeeze(-1).masked_fill(~neighbor_mask, -1e9)
        weights = torch.softmax(logits, dim=-1).masked_fill(~neighbor_mask, 0.0)
        pooled = torch.sum(hidden * weights.unsqueeze(-1), dim=1)
        return self.output_proj(pooled)


class SymmetricInfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, target_feat: torch.Tensor, neighbor_feat: torch.Tensor):
        n = int(target_feat.shape[0])
        if n <= 1:
            zero = target_feat.new_tensor(0.0, requires_grad=True)
            return zero, zero.detach(), zero.detach(), zero.detach(), zero.detach()

        target_feat = F.normalize(target_feat, dim=-1)
        neighbor_feat = F.normalize(neighbor_feat, dim=-1)
        logits = target_feat @ neighbor_feat.t() / max(self.temperature, 1e-8)
        labels = torch.arange(n, device=target_feat.device)
        loss_t2n = F.cross_entropy(logits, labels)
        loss_n2t = F.cross_entropy(logits.t(), labels)
        loss = 0.5 * (loss_t2n + loss_n2t)
        top1 = (logits.argmax(dim=1) == labels).float().mean()
        top5 = (logits.topk(min(5, n), dim=1).indices == labels.unsqueeze(1)).any(dim=1).float().mean()
        ranks = (logits.argsort(dim=1, descending=True) == labels.unsqueeze(1)).nonzero(as_tuple=True)[1]
        mrr = (1.0 / (ranks.float() + 1.0)).mean()
        diag_cos = torch.sum(target_feat * neighbor_feat, dim=-1).mean()
        return loss, top1, top5, mrr, diag_cos


class GeneformerNeighborAligner(pl.LightningModule):
    """Geneformer-native Step-1 model.

    Aligns a target gene's contextual Geneformer embedding h_g with a set
    representation of its neighbor genes' contextual Geneformer embeddings in
    the same cell.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        proj_dim: int = 256,
        nhead: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        init_lr: float = 3e-4,
        weight_decay: float = 0.05,
        warmup_steps: int = 200,
        temperature: float = 0.07,
        lambda_hard_negative: float = 0.0,
        hard_negative_margin: float = 0.05,
        hard_negative_top_k: int = 8,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.target_projector = MLPProjector(input_dim=input_dim, hidden_dim=hidden_dim, proj_dim=proj_dim, dropout=dropout)
        self.neighbor_encoder = NeighborSetEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            proj_dim=proj_dim,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.criterion = SymmetricInfoNCELoss(temperature=temperature)
        self.init_lr = float(init_lr)
        self.weight_decay = float(weight_decay)
        self.warmup_steps = int(warmup_steps)
        self.lambda_hard_negative = float(lambda_hard_negative)
        self.hard_negative_margin = float(hard_negative_margin)
        self.hard_negative_top_k = int(hard_negative_top_k)

    def encode_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.device
        target_embeddings = batch["target_embeddings"].to(device=device, dtype=torch.float32)
        neighbor_embeddings = batch["neighbor_embeddings"].to(device=device, dtype=torch.float32)
        neighbor_mask = batch["neighbor_mask"].to(device=device, dtype=torch.bool)
        target_feat = F.normalize(self.target_projector(target_embeddings), dim=-1)
        neighbor_feat = F.normalize(self.neighbor_encoder(neighbor_embeddings, neighbor_mask), dim=-1)
        return target_feat, neighbor_feat

    def _shared_step(self, batch: dict, prefix: str) -> torch.Tensor:
        target_feat, neighbor_feat = self.encode_batch(batch)
        loss, top1, top5, mrr, diag_cos = self.criterion(target_feat, neighbor_feat)
        hard_loss, hard_acc, hard_gap = self._hard_negative_loss(target_feat, neighbor_feat)
        total_loss = loss + self.lambda_hard_negative * hard_loss
        n_pairs = max(1, int(target_feat.shape[0]))
        self.log(f"{prefix}_loss", total_loss, prog_bar=True, on_step=prefix == "train", on_epoch=True, batch_size=n_pairs)
        self.log(f"{prefix}_nce_loss", loss, on_step=prefix == "train", on_epoch=True, batch_size=n_pairs)
        self.log(f"{prefix}_hard_negative_loss", hard_loss, on_step=prefix == "train", on_epoch=True, batch_size=n_pairs)
        self.log(f"{prefix}_hard_negative_acc", hard_acc, on_step=prefix == "train", on_epoch=True, batch_size=n_pairs)
        self.log(f"{prefix}_hard_negative_gap", hard_gap, on_step=prefix == "train", on_epoch=True, batch_size=n_pairs)
        self.log(f"{prefix}_top1", top1, prog_bar=True, on_step=prefix == "train", on_epoch=True, batch_size=n_pairs)
        self.log(f"{prefix}_top5", top5, on_step=prefix == "train", on_epoch=True, batch_size=n_pairs)
        self.log(f"{prefix}_mrr", mrr, on_step=prefix == "train", on_epoch=True, batch_size=n_pairs)
        self.log(f"{prefix}_diag_cos", diag_cos, on_step=prefix == "train", on_epoch=True, batch_size=n_pairs)
        return total_loss

    def _hard_negative_loss(self, target_feat: torch.Tensor, neighbor_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = int(target_feat.shape[0])
        if self.lambda_hard_negative <= 0 or n <= 2:
            zero = target_feat.new_tensor(0.0)
            return zero, zero, zero

        target_feat = F.normalize(target_feat, dim=-1)
        neighbor_feat = F.normalize(neighbor_feat, dim=-1)
        sim = target_feat @ neighbor_feat.t()
        labels = torch.arange(n, device=sim.device)
        pos = sim[labels, labels]
        neg = sim.masked_fill(torch.eye(n, device=sim.device, dtype=torch.bool), -1e9)
        k = min(max(1, self.hard_negative_top_k), n - 1)

        hardest_t2n = neg.topk(k, dim=1).values
        loss_t2n = F.relu(self.hard_negative_margin + hardest_t2n - pos.unsqueeze(1)).mean()
        gap_t2n = (pos - hardest_t2n[:, 0]).mean()
        acc_t2n = (pos > hardest_t2n[:, 0]).to(torch.float32).mean()

        hardest_n2t = neg.t().topk(k, dim=1).values
        loss_n2t = F.relu(self.hard_negative_margin + hardest_n2t - pos.unsqueeze(1)).mean()
        gap_n2t = (pos - hardest_n2t[:, 0]).mean()
        acc_n2t = (pos > hardest_n2t[:, 0]).to(torch.float32).mean()

        loss = 0.5 * (loss_t2n + loss_n2t)
        acc = 0.5 * (acc_t2n + acc_n2t)
        gap = 0.5 * (gap_t2n + gap_n2t)
        return loss, acc.detach(), gap.detach()

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.init_lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-6,
        )
        total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
        if not total_steps or total_steps <= 0:
            total_steps = 1000
        warmup_steps = min(self.warmup_steps, max(1, total_steps // 10))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

