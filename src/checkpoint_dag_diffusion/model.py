from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .tensorize import NUM_OPERATORS


class MLP(nn.Module):
    def __init__(self, dims: list[int], dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                layers.append(nn.SiLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RelationEmbedding(nn.Module):
    def __init__(self, hidden_dim: int, max_level: int = 16, max_delta: int = 16, num_operators: int = NUM_OPERATORS) -> None:
        super().__init__()
        self.max_level = max_level
        self.max_delta = max_delta
        self.guide_level_emb = nn.Embedding(max_level + 2, hidden_dim)
        self.target_level_emb = nn.Embedding(max_level + 2, hidden_dim)
        self.delta_emb = nn.Embedding(max_delta + 2, hidden_dim)
        self.operator_emb = nn.Embedding(num_operators, hidden_dim)
        self.proj = nn.Sequential(nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, relation: torch.Tensor) -> torch.Tensor:
        guide_level = relation[:, 0].clamp(min=0, max=self.max_level + 1)
        target_level = relation[:, 1].clamp(min=0, max=self.max_level + 1)
        delta = relation[:, 2].clamp(min=0, max=self.max_delta + 1)
        operator = relation[:, 3].clamp(min=0, max=self.operator_emb.num_embeddings - 1)
        emb = (
            self.guide_level_emb(guide_level)
            + self.target_level_emb(target_level)
            + self.delta_emb(delta)
            + self.operator_emb(operator)
        )
        return self.proj(emb)


class GraphStateEncoder(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        edge_dim: int,
        hidden_dim: int = 192,
        num_layers: int = 4,
        num_heads: int = 6,
        dropout: float = 0.1,
        max_timesteps: int = 500,
        max_level: int = 16,
        max_delta: int = 16,
    ) -> None:
        super().__init__()
        self.node_proj = nn.Linear(node_feature_dim, hidden_dim)
        self.edge_emb = nn.Embedding(edge_dim, hidden_dim)
        self.time_emb = nn.Embedding(max_timesteps + 1, hidden_dim)
        self.relation_emb = RelationEmbedding(hidden_dim=hidden_dim, max_level=max_level, max_delta=max_delta)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge: torch.Tensor, mask: torch.Tensor, t: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        # Equation h_u^(0) aggregates A_{Q,uv} only over quotient nodes v.
        # Padded slots and the diagonal are representation artifacts rather than
        # incident quotient pairs, so neither may contribute to the aggregate.
        # A masked mean keeps the scale stable across checkpoint sizes while
        # preserving permutation equivariance.
        edge_features = self.edge_emb(edge)
        pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
        diagonal = torch.eye(edge.size(1), dtype=torch.bool, device=edge.device).unsqueeze(0)
        pair_mask = pair_mask & ~diagonal
        edge_weights = pair_mask.unsqueeze(-1).to(dtype=edge_features.dtype)
        edge_context = (edge_features * edge_weights).sum(dim=2)
        edge_context = edge_context / edge_weights.sum(dim=2).clamp_min(1.0)
        h = self.node_proj(x) + edge_context
        h = h + self.time_emb(t).unsqueeze(1) + self.relation_emb(relation).unsqueeze(1)
        h = h * mask.unsqueeze(-1).to(dtype=h.dtype)
        safe_mask = mask.clone()
        empty = ~safe_mask.any(dim=1)
        if empty.any():
            safe_mask[empty, 0] = True
        h = self.encoder(h, src_key_padding_mask=~safe_mask)
        return self.norm(h) * mask.unsqueeze(-1).to(dtype=h.dtype)


@dataclass
class DenoiserOutput:
    node_features: torch.Tensor
    edge_logits: torch.Tensor
    active_logits: torch.Tensor
    split_logits: torch.Tensor
    attachment_logits: torch.Tensor


class CheckpointDAGDenoiser(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        edge_dim: int,
        hidden_dim: int = 192,
        num_layers: int = 4,
        guide_num_layers: int | None = None,
        num_heads: int = 6,
        dropout: float = 0.1,
        max_timesteps: int = 500,
        max_level: int = 16,
        max_delta: int = 16,
        num_global_codes: int = 0,
        max_global_edges: int = 0,
    ) -> None:
        super().__init__()
        self.num_global_codes = int(num_global_codes)
        self.max_global_edges = int(max_global_edges)
        self.encoder = GraphStateEncoder(
            node_feature_dim=node_feature_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_timesteps=max_timesteps,
            max_level=max_level,
            max_delta=max_delta,
        )
        self.guide_encoder = GraphStateEncoder(
            node_feature_dim=node_feature_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=max(1, int(guide_num_layers if guide_num_layers is not None else num_layers // 2)),
            num_heads=num_heads,
            dropout=dropout,
            max_timesteps=max_timesteps,
            max_level=max_level,
            max_delta=max_delta,
        )
        if self.num_global_codes > 0:
            # Index 0 is a learned null/dropped condition. Real global ids are
            # shifted by +1 and remain separate from VQ codebook vectors.
            self.guide_global_emb = nn.Embedding(self.num_global_codes + 1, hidden_dim)
            self.target_global_emb = nn.Embedding(self.num_global_codes + 1, hidden_dim)
            if self.max_global_edges > 0:
                self.global_edge_emb = nn.Embedding(self.max_global_edges + 1, hidden_dim)
            else:
                self.global_edge_emb = None
            self.global_condition_proj = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.guide_global_emb = None
            self.target_global_emb = None
            self.global_edge_emb = None
            self.global_condition_proj = None
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.node_head = MLP([hidden_dim, hidden_dim * 2, node_feature_dim], dropout=dropout)
        self.active_head = nn.Linear(hidden_dim, 1)
        self.split_head = nn.Linear(hidden_dim, 1)
        self.attachment_head = nn.Linear(hidden_dim, 1)
        self.edge_head = MLP([hidden_dim * 4, hidden_dim * 2, edge_dim], dropout=dropout)

    def forward(
        self,
        x_t: torch.Tensor,
        edge_t: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor,
        relation: torch.Tensor,
        guide_x: torch.Tensor | None = None,
        guide_edge: torch.Tensor | None = None,
        guide_mask: torch.Tensor | None = None,
        has_guide: torch.Tensor | None = None,
        guide_global_id: torch.Tensor | None = None,
        target_global_id: torch.Tensor | None = None,
        global_edge_id: torch.Tensor | None = None,
        has_global_condition: torch.Tensor | None = None,
    ) -> DenoiserOutput:
        h = self.encoder(x_t, edge_t, mask, t, relation)
        if self.guide_global_emb is not None and target_global_id is not None:
            if guide_global_id is None:
                guide_global_id = torch.full_like(target_global_id, -1)
            if has_global_condition is None:
                has_global_condition = torch.ones_like(target_global_id, dtype=torch.bool)

            def shifted(values: torch.Tensor) -> torch.Tensor:
                values = values.long() + 1
                values = torch.where(has_global_condition, values, torch.zeros_like(values))
                return values.clamp(min=0, max=self.num_global_codes)

            cond = self.target_global_emb(shifted(target_global_id))
            cond = cond + self.guide_global_emb(shifted(guide_global_id))
            if self.global_edge_emb is not None and global_edge_id is not None:
                edge_values = global_edge_id.long() + 1
                edge_values = torch.where(has_global_condition, edge_values, torch.zeros_like(edge_values))
                cond = cond + self.global_edge_emb(edge_values.clamp(min=0, max=self.max_global_edges))
            h = h + self.global_condition_proj(cond).unsqueeze(1) * mask.unsqueeze(-1).to(dtype=h.dtype)
        if guide_x is not None and guide_edge is not None and guide_mask is not None:
            if has_guide is None:
                has_guide = guide_mask.any(dim=1)
            safe_guide_mask = guide_mask.clone()
            no_guide = ~safe_guide_mask.any(dim=1)
            if no_guide.any():
                safe_guide_mask[no_guide, 0] = True
            guide_t = torch.zeros_like(t)
            guide_h = self.guide_encoder(guide_x, guide_edge, safe_guide_mask, guide_t, relation)
            guide_h = torch.where(safe_guide_mask.unsqueeze(-1), guide_h, torch.zeros_like(guide_h))
            attn, _ = self.cross_attn(
                query=h,
                key=guide_h,
                value=guide_h,
                key_padding_mask=~safe_guide_mask,
                need_weights=False,
            )
            gate = has_guide.view(-1, 1, 1).to(dtype=h.dtype)
            h = self.cross_norm(h + attn * gate)
            h = h * mask.unsqueeze(-1).to(dtype=h.dtype)

        node_features = self.node_head(h)
        active_logits = self.active_head(h).squeeze(-1)
        split_logits = self.split_head(h).squeeze(-1)
        attachment_logits = self.attachment_head(h).squeeze(-1)

        hi = h.unsqueeze(2).expand(-1, -1, h.size(1), -1)
        hj = h.unsqueeze(1).expand(-1, h.size(1), -1, -1)
        pair = torch.cat([hi, hj, torch.abs(hi - hj), hi * hj], dim=-1)
        edge_logits = self.edge_head(pair)
        edge_logits = 0.5 * (edge_logits + edge_logits.transpose(1, 2))
        return DenoiserOutput(
            node_features=node_features,
            edge_logits=edge_logits,
            active_logits=active_logits,
            split_logits=split_logits,
            attachment_logits=attachment_logits,
        )
