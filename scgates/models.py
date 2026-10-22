from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class NoiseAwareSetEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1, denoise_mode: str = "normal"):
        super().__init__()
        if denoise_mode not in {"normal", "none"}:
            raise ValueError(f"Unsupported NoiseAwareSetEncoder denoise_mode: {denoise_mode}")
        self.denoise_mode = denoise_mode
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.query = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.conf = nn.Sequential(
            nn.Linear(input_dim + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        embeddings: torch.Tensor,
        ranks: torch.Tensor,
        expr: torch.Tensor,
        mask: torch.Tensor,
        occurrence_rate: torch.Tensor | None = None,
        instance_scores: torch.Tensor | None = None,
        instance_score_alpha: float = 0.0,
        instance_score_mode: str = "signed",
    ) -> dict[str, torch.Tensor]:
        x = self.input_proj(embeddings)
        content_logits = torch.einsum("bnh,h->bn", x, self.query)
        if self.denoise_mode == "none":
            prior = self._uniform_prior(mask, embeddings.dtype)
            conf = torch.zeros_like(content_logits)
            prior_logit = torch.zeros_like(content_logits)
            conf_logit = torch.zeros_like(content_logits)
        else:
            prior = self._representativeness_prior(embeddings, mask)
            if occurrence_rate is None:
                occurrence_rate = torch.zeros(embeddings.shape[0], dtype=embeddings.dtype, device=embeddings.device)
            log_occurrence = torch.log(occurrence_rate.clamp_min(1e-8)).view(-1, 1).expand_as(ranks)
            rank_expr = torch.stack([ranks, expr, log_occurrence], dim=-1)
            conf_in = torch.cat([embeddings, rank_expr], dim=-1)
            conf = self.conf(conf_in).squeeze(-1)
            prior_logit = self.alpha * torch.log(prior + 1e-8)
            conf_logit = self.beta * conf
        instance_score_logit = torch.zeros_like(content_logits)
        logits = content_logits + prior_logit + conf_logit
        if instance_scores is not None and instance_score_alpha > 0:
            if instance_score_mode == "positive":
                score_bias = torch.relu(instance_scores)
            elif instance_score_mode == "signed":
                score_bias = instance_scores
            else:
                raise ValueError(f"Unsupported instance_score_mode: {instance_score_mode}")
            instance_score_logit = float(instance_score_alpha) * score_bias.masked_fill(~mask, 0.0)
            logits = logits + instance_score_logit
        logits = logits.masked_fill(~mask, -1e9)
        weights = torch.softmax(logits, dim=-1)
        instance_hidden = self.value(x).masked_fill(~mask.unsqueeze(-1), 0.0)
        pooled = torch.sum(weights.unsqueeze(-1) * instance_hidden, dim=1)
        z = self.ffn(pooled) + pooled
        entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1)
        return {
            "z": z,
            "weights": weights,
            "prior": prior,
            "entropy": entropy,
            "instance_hidden": instance_hidden,
            "conf": conf,
            "content_logit": content_logits,
            "prior_logit": prior_logit,
            "conf_logit": conf_logit,
            "instance_score_logit": instance_score_logit,
            "attn_logit": logits,
        }

    @staticmethod
    def _representativeness_prior(embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = F.normalize(embeddings, dim=-1)
        sim = torch.matmul(x, x.transpose(1, 2))
        pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        sim = sim.masked_fill(~pair_mask, 0.0)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(embeddings.dtype)
        score = sim.sum(dim=-1) / denom
        score = score.masked_fill(~mask, -1e9)
        return torch.softmax(score, dim=-1)

    @staticmethod
    def _uniform_prior(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        prior = mask.to(dtype)
        return prior / prior.sum(dim=1, keepdim=True).clamp_min(1.0)


class DistributionBiasedSelfAttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim must be divisible by num_heads: {hidden_dim} % {num_heads}")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_bias: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim**0.5)
        logits = logits + key_bias[:, None, None, :]
        logits = logits.masked_fill(~mask[:, None, None, :], -1e9)
        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        x = self.norm_attn(x + self.out_proj(context))
        x = self.norm_ffn(x + self.ffn(x))
        x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        return x, attn


class DistributionBiasedSetEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        num_heads: int = 4,
        num_layers: int = 2,
        denoise_mode: str = "normal",
    ):
        super().__init__()
        if denoise_mode not in {"normal", "none"}:
            raise ValueError(f"Unsupported DistributionBiasedSetEncoder denoise_mode: {denoise_mode}")
        self.denoise_mode = denoise_mode
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.conf = nn.Sequential(
            nn.Linear(input_dim + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.layers = nn.ModuleList(
            [DistributionBiasedSelfAttentionLayer(hidden_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        ranks: torch.Tensor,
        expr: torch.Tensor,
        mask: torch.Tensor,
        occurrence_rate: torch.Tensor | None = None,
        instance_scores: torch.Tensor | None = None,
        instance_score_alpha: float = 0.0,
        instance_score_mode: str = "signed",
    ) -> dict[str, torch.Tensor]:
        x = self.input_proj(embeddings)
        if self.denoise_mode == "none":
            prior = NoiseAwareSetEncoder._uniform_prior(mask, embeddings.dtype)
            conf = torch.zeros_like(ranks)
            content_logit = torch.zeros_like(conf)
            prior_logit = torch.zeros_like(conf)
            conf_logit = torch.zeros_like(conf)
        else:
            prior = NoiseAwareSetEncoder._representativeness_prior(embeddings, mask)
            if occurrence_rate is None:
                occurrence_rate = torch.zeros(embeddings.shape[0], dtype=embeddings.dtype, device=embeddings.device)
            log_occurrence = torch.log(occurrence_rate.clamp_min(1e-8)).view(-1, 1).expand_as(ranks)
            conf_in = torch.cat([embeddings, torch.stack([ranks, expr, log_occurrence], dim=-1)], dim=-1)
            conf = self.conf(conf_in).squeeze(-1)
            content_logit = torch.zeros_like(conf)
            prior_logit = self.alpha * torch.log(prior + 1e-8)
            conf_logit = self.beta * conf
        instance_score_logit = torch.zeros_like(conf)
        key_bias = prior_logit + conf_logit
        if instance_scores is not None and instance_score_alpha > 0:
            if instance_score_mode == "positive":
                score_bias = torch.relu(instance_scores)
            elif instance_score_mode == "signed":
                score_bias = instance_scores
            else:
                raise ValueError(f"Unsupported instance_score_mode: {instance_score_mode}")
            instance_score_logit = float(instance_score_alpha) * score_bias.masked_fill(~mask, 0.0)
            key_bias = key_bias + instance_score_logit
        key_bias = key_bias.masked_fill(~mask, -1e9)

        last_attn: torch.Tensor | None = None
        for layer in self.layers:
            x, last_attn = layer(x, key_bias, mask)

        if last_attn is None:
            weights = prior
        else:
            query_mask = mask[:, None, :, None].to(last_attn.dtype)
            weights = (last_attn * query_mask).sum(dim=(1, 2))
            weights = weights.masked_fill(~mask, 0.0)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        pooled = torch.sum(weights.unsqueeze(-1) * x, dim=1)
        z = self.ffn(pooled) + pooled
        entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1)
        return {
            "z": z,
            "weights": weights,
            "prior": prior,
            "entropy": entropy,
            "instance_hidden": x,
            "conf": conf,
            "content_logit": content_logit,
            "prior_logit": prior_logit,
            "conf_logit": conf_logit,
            "instance_score_logit": instance_score_logit,
            "attn_logit": key_bias,
        }


class MeanEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(embeddings.dtype)
        mean = (embeddings * mask.unsqueeze(-1)).sum(dim=1) / denom
        return self.proj(mean)


class TextSetEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        metadata_dim: int = 0,
        conf_mode: str = "text_only",
        conf_gate_mode: str = "none",
        conf_gate_scale: float = 1.0,
    ):
        super().__init__()
        if conf_mode not in {"text_only", "text_prototype", "text_prototype_loo"}:
            raise ValueError(f"Unsupported TextSetEncoder conf_mode: {conf_mode}")
        if conf_gate_mode not in {"none", "prior_sigmoid"}:
            raise ValueError(f"Unsupported TextSetEncoder conf_gate_mode: {conf_gate_mode}")
        self.metadata_dim = metadata_dim
        self.conf_mode = conf_mode
        self.conf_gate_mode = conf_gate_mode
        self.conf_gate_scale = float(conf_gate_scale)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.query = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        conf_input_dim = input_dim + 1 + metadata_dim
        if conf_mode in {"text_prototype", "text_prototype_loo"}:
            conf_input_dim = input_dim * 3 + 1 + metadata_dim
        self.conf = nn.Sequential(
            nn.Linear(conf_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        embeddings: torch.Tensor,
        prior: torch.Tensor,
        mask: torch.Tensor,
        metadata_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        x = self.input_proj(embeddings)
        prior = prior.masked_fill(~mask, 0.0)
        prior = prior / prior.sum(dim=1, keepdim=True).clamp_min(1e-8)
        log_count = torch.log(mask.sum(dim=1, keepdim=True).clamp_min(1).to(embeddings.dtype))
        log_count = log_count.unsqueeze(-1).expand(-1, embeddings.shape[1], -1)
        if self.conf_mode == "text_only":
            conf_base = [embeddings]
        else:
            mask_f = mask.unsqueeze(-1).to(embeddings.dtype)
            if self.conf_mode == "text_prototype_loo":
                denom = (mask_f.sum(dim=1, keepdim=True) - mask_f).clamp_min(1.0)
                proto = ((embeddings * mask_f).sum(dim=1, keepdim=True) - embeddings * mask_f) / denom
                singleton_proto = (embeddings * mask_f).sum(dim=1, keepdim=True) / mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
                proto = torch.where((mask_f.sum(dim=1, keepdim=True) <= 1.0), singleton_proto, proto)
                proto = proto.expand_as(embeddings)
            else:
                proto = (embeddings * mask_f).sum(dim=1, keepdim=True) / mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
                proto = proto.expand_as(embeddings)
            conf_base = [embeddings, proto, embeddings * proto]
        if self.metadata_dim:
            if metadata_features is None:
                metadata_features = torch.zeros(
                    embeddings.shape[0],
                    embeddings.shape[1],
                    self.metadata_dim,
                    dtype=embeddings.dtype,
                    device=embeddings.device,
                )
            conf_in = torch.cat([*conf_base, log_count, metadata_features], dim=-1)
        else:
            conf_in = torch.cat([*conf_base, log_count], dim=-1)
        conf = self.conf(conf_in).squeeze(-1)
        content_logits = torch.einsum("bnh,h->bn", x, self.query)
        if self.conf_gate_mode == "prior_sigmoid":
            conf_gate = torch.sigmoid(float(self.conf_gate_scale) * conf)
            conf_gate = conf_gate.masked_fill(~mask, 0.0)
            prior_for_logit = prior * conf_gate
            prior_for_logit = prior_for_logit.masked_fill(~mask, 0.0)
            prior_for_logit = prior_for_logit / prior_for_logit.sum(dim=1, keepdim=True).clamp_min(1e-8)
        else:
            conf_gate = torch.ones_like(prior).masked_fill(~mask, 0.0)
            prior_for_logit = prior
        prior_logit = self.alpha * torch.log(prior_for_logit + 1e-8)
        conf_logit = self.beta * conf
        logits = content_logits + prior_logit + conf_logit
        logits = logits.masked_fill(~mask, -1e9)
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.sum(weights.unsqueeze(-1) * self.value(x), dim=1)
        z = self.ffn(pooled) + pooled
        entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1)
        return {
            "z": z,
            "weights": weights,
            "prior": prior,
            "conf_gate": conf_gate,
            "gated_prior": prior_for_logit,
            "entropy": entropy,
            "conf": conf,
            "content_logit": content_logits,
            "prior_logit": prior_logit,
            "conf_logit": conf_logit,
            "attn_logit": logits,
        }


class TextAwareSetScorer(nn.Module):
    def __init__(self, cell_dim: int, align_dim: int, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Linear(align_dim, align_dim)
        self.key = nn.Linear(cell_dim, align_dim)
        self.value = nn.Linear(cell_dim, align_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Sequential(
            nn.LayerNorm(align_dim),
            nn.Linear(align_dim, align_dim),
        )

    def forward(
        self,
        cell_embeddings: torch.Tensor,
        cell_mask: torch.Tensor,
        text_set: torch.Tensor,
        instance_scores: torch.Tensor | None = None,
        alpha: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.query(text_set)
        k = self.key(cell_embeddings)
        v = self.value(cell_embeddings)
        scale = float(q.shape[-1]) ** 0.5

        logits = torch.einsum("jd,ind->ijn", q, k) / scale
        if instance_scores is not None and alpha > 0:
            logits = logits + float(alpha) * instance_scores[:, None, :].masked_fill(~cell_mask[:, None, :], 0.0)
        logits = logits.masked_fill(~cell_mask[:, None, :], -1e9)

        weights = torch.softmax(logits, dim=-1)
        weights = self.dropout(weights)
        z = torch.einsum("ijn,ind->ijd", weights, v)
        z = F.normalize(self.out(z), dim=-1)
        text = F.normalize(text_set, dim=-1)
        scores = torch.einsum("ijd,jd->ij", z, text)
        return scores, weights


class NeighborAggregator(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.prior_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        target_z: torch.Tensor,
        neighbor_z: torch.Tensor,
        neighbor_mask: torch.Tensor,
        neighbor_prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if neighbor_z.numel() == 0:
            return self.out(target_z)
        target = target_z.unsqueeze(1).expand_as(neighbor_z)
        pair_feat = torch.cat([target, neighbor_z, target * neighbor_z, torch.abs(target - neighbor_z)], dim=-1)
        logits = self.gate(pair_feat).squeeze(-1)
        logits = logits + F.cosine_similarity(target, neighbor_z, dim=-1)
        if neighbor_prior is not None:
            logits = logits + self.prior_scale * neighbor_prior
        valid_rows = neighbor_mask.any(dim=-1, keepdim=True)
        logits = logits.masked_fill(~neighbor_mask, -1e9)
        logits = torch.where(valid_rows, logits, torch.zeros_like(logits))
        weights = torch.softmax(logits, dim=-1)
        weights = weights * neighbor_mask.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        pooled = torch.sum(weights.unsqueeze(-1) * neighbor_z, dim=1)
        return self.out(target_z + pooled)


class AlignmentHead(nn.Module):

    def __init__(self, hidden_dim: int, text_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, text_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(z), dim=-1)

