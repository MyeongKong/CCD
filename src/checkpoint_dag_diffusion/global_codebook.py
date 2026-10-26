from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .artifacts import (
    ASSIGNMENTS_FORMAT,
    ArtifactVersionError,
    CERTIFIED_OBJECT_FORMAT,
    artifact_fingerprint,
    load_corpus_manifest,
    load_corpus_records,
    validate_codebook,
)
from .checkpoint_state import CheckpointState
from .metrics import ring_count
from .model import GraphStateEncoder, MLP
from .tensorize import TensorSpec, checkpoint_to_tensors, collate_tensor_dicts


ZERO_RELATION = torch.tensor([0, 0, 0, 0], dtype=torch.long)


@dataclass(frozen=True)
class CheckpointPoolItem:
    record_index: int
    checkpoint_index: int
    level: int
    checkpoint_hash: str
    state: CheckpointState


class CheckpointPoolDataset(Dataset):
    """Flat pool of checkpoint states extracted from a DAG corpus."""

    def __init__(self, corpus_dir: str | Path, split: str = "train", spec: TensorSpec | None = None) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.split = split
        manifest = load_corpus_manifest(self.corpus_dir)
        self.records: list[dict[str, Any]] = load_corpus_records(self.corpus_dir, split)
        corpus_spec = TensorSpec(
            atom_vocab=tuple(manifest["atom_vocab"]),
            bond_vocab=tuple(manifest["bond_vocab"]),
            max_nodes=int(manifest["max_nodes"]),
        )
        if spec is not None and spec != corpus_spec:
            raise ArtifactVersionError(
                "Codebook TensorSpec does not match the v2 corpus manifest; "
                "regenerate the codebook from this corpus."
            )
        self.spec = corpus_spec
        self.index: list[tuple[int, int]] = []
        for rec_idx, record in enumerate(self.records):
            for cp_idx, _ in enumerate(record["checkpoints"]):
                self.index.append((rec_idx, cp_idx))

    def __len__(self) -> int:
        return len(self.index)

    def item_meta(self, idx: int) -> CheckpointPoolItem:
        rec_idx, cp_idx = self.index[idx]
        checkpoint = self.records[rec_idx]["checkpoints"][cp_idx]
        return CheckpointPoolItem(
            record_index=rec_idx,
            checkpoint_index=cp_idx,
            level=int(checkpoint["level"]),
            checkpoint_hash=str(checkpoint["hash"]),
            state=CheckpointState.from_dict(checkpoint["state"]),
        )

    def __getitem__(self, idx: int) -> dict:
        meta = self.item_meta(idx)
        rec_idx, cp_idx = self.index[idx]
        checkpoint = self.records[rec_idx]["checkpoints"][cp_idx]
        tensors = checkpoint_to_tensors(checkpoint, self.spec)
        cert = checkpoint.get("certified_object", {})
        certificate_hash = str(cert.get("certificate_hash", meta.checkpoint_hash))
        return {
            "state": tensors,
            "relation": ZERO_RELATION.clone(),
            "hash": certificate_hash,
            "record_index": torch.tensor(meta.record_index, dtype=torch.long),
            "checkpoint_index": torch.tensor(meta.checkpoint_index, dtype=torch.long),
            "level": torch.tensor(meta.level, dtype=torch.long),
            "ring_count": torch.tensor(ring_count(meta.state), dtype=torch.float32),
            "core_count": torch.tensor(sum(bool(v) for v in meta.state.core_flags), dtype=torch.float32),
        }


def collate_checkpoint_pool(items: list[dict]) -> dict:
    return {
        "state": collate_tensor_dicts([item["state"] for item in items]),
        "relation": torch.stack([item["relation"] for item in items], dim=0),
        "hash": [item["hash"] for item in items],
        "record_index": torch.stack([item["record_index"] for item in items], dim=0),
        "checkpoint_index": torch.stack([item["checkpoint_index"] for item in items], dim=0),
        "level": torch.stack([item["level"] for item in items], dim=0),
        "ring_count": torch.stack([item["ring_count"] for item in items], dim=0),
        "core_count": torch.stack([item["core_count"] for item in items], dim=0),
    }


class CheckpointGraphPoolEncoder(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_nodes: int = 32,
    ) -> None:
        super().__init__()
        self.state_encoder = GraphStateEncoder(
            node_feature_dim=node_feature_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_timesteps=1,
            max_level=1,
            max_delta=1,
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)
        self.max_nodes = int(max_nodes)

    def forward(self, x: torch.Tensor, edge: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        t = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        relation = torch.zeros(batch_size, 4, dtype=torch.long, device=x.device)
        h = self.state_encoder(x, edge, mask, t, relation)
        weights = mask.unsqueeze(-1).to(dtype=h.dtype)
        pooled = (h * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.norm(pooled)


class CheckpointCodebookModel(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        edge_dim: int,
        max_nodes: int = 32,
        codebook_size: int = 512,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.hidden_dim = int(hidden_dim)
        self.max_nodes = int(max_nodes)
        self.node_feature_dim = int(node_feature_dim)
        self.edge_dim = int(edge_dim)
        self.encoder = CheckpointGraphPoolEncoder(
            node_feature_dim=node_feature_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_nodes=max_nodes,
        )
        self.codebook = nn.Parameter(torch.randn(codebook_size, hidden_dim) * 0.02)
        self.node_decoder = MLP([hidden_dim, hidden_dim * 2, max_nodes * node_feature_dim], dropout=dropout)
        self.active_decoder = MLP([hidden_dim, hidden_dim, max_nodes], dropout=dropout)
        self.edge_decoder = MLP([hidden_dim, hidden_dim * 2, max_nodes * max_nodes * edge_dim], dropout=dropout)
        self.topology_head = MLP([hidden_dim, hidden_dim, 2], dropout=dropout)

    def encode(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encoder(state["x"], state["edge"], state["mask"])

    def quantize(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distances = torch.cdist(embedding, self.codebook, p=2.0) ** 2
        ids = torch.argmin(distances, dim=1)
        quantized = self.codebook[ids]
        return ids, quantized, distances.gather(1, ids.view(-1, 1)).squeeze(1)

    def decode(self, quantized: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = quantized.size(0)
        return {
            "x": self.node_decoder(quantized).view(batch_size, self.max_nodes, self.node_feature_dim),
            "active_logits": self.active_decoder(quantized).view(batch_size, self.max_nodes),
            "edge_logits": self.edge_decoder(quantized).view(batch_size, self.max_nodes, self.max_nodes, self.edge_dim),
            "topology": self.topology_head(quantized),
        }

    def forward(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        embedding = self.encode(state)
        ids, quantized, distances = self.quantize(embedding)
        quantized_st = embedding + (quantized - embedding).detach()
        decoded = self.decode(quantized_st)
        decoded.update({"embedding": embedding, "quantized": quantized, "code_ids": ids, "assignment_distance": distances})
        return decoded


def edge_reconstruction_loss(edge_logits: torch.Tensor, edge_target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pair_mask = torch.triu(mask.unsqueeze(1) & mask.unsqueeze(2), diagonal=1)
    if not bool(pair_mask.any().item()):
        return edge_logits.sum() * 0.0
    return F.cross_entropy(edge_logits[pair_mask], edge_target[pair_mask].long())


def contrastive_hash_loss(embedding: torch.Tensor, hashes: list[str], margin: float = 1.0) -> torch.Tensor:
    if len(hashes) < 2:
        return embedding.sum() * 0.0
    device = embedding.device
    same = torch.zeros(len(hashes), len(hashes), dtype=torch.bool, device=device)
    for i, h_i in enumerate(hashes):
        for j, h_j in enumerate(hashes):
            if i != j and h_i == h_j:
                same[i, j] = True
    dist = torch.cdist(embedding, embedding, p=2.0)
    positive = (dist[same] ** 2).mean() if bool(same.any().item()) else embedding.sum() * 0.0
    negative_mask = ~same & ~torch.eye(len(hashes), dtype=torch.bool, device=device)
    negative = F.relu(margin - dist[negative_mask]).pow(2).mean() if bool(negative_mask.any().item()) else positive * 0.0
    return positive + negative


def codebook_loss(
    output: dict[str, torch.Tensor],
    batch: dict,
    beta: float = 0.25,
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = weights or {}
    state = batch["state"]
    mask = state["mask"]
    x_loss = (((output["x"] - state["x"]) ** 2) * mask.unsqueeze(-1).float()).sum() / mask.sum().clamp_min(1).float()
    active_loss = F.binary_cross_entropy_with_logits(output["active_logits"], mask.float())
    edge_loss = edge_reconstruction_loss(output["edge_logits"], state["edge"], mask)
    topology_target = torch.stack([batch["ring_count"], batch["core_count"]], dim=1).to(output["topology"].device)
    topo_loss = F.mse_loss(output["topology"], topology_target)
    embedding = output["embedding"]
    quantized = output["quantized"]
    vq_loss = F.mse_loss(quantized, embedding.detach())
    commit_loss = F.mse_loss(embedding, quantized.detach())
    con_loss = contrastive_hash_loss(embedding, batch["hash"])
    total = (
        weights.get("rec", weights.get("reconstruction", 1.0)) * (x_loss + active_loss + edge_loss)
        + weights.get("topology", 0.05) * topo_loss
        + weights.get("vq", 1.0) * vq_loss
        + weights.get("commitment", beta) * commit_loss
        + weights.get("hash", weights.get("contrastive", 0.05)) * con_loss
    )
    logs = {
        "loss": float(total.detach().cpu()),
        "x": float(x_loss.detach().cpu()),
        "active": float(active_loss.detach().cpu()),
        "edge": float(edge_loss.detach().cpu()),
        "topology": float(topo_loss.detach().cpu()),
        "vq": float(vq_loss.detach().cpu()),
        "commit": float(commit_loss.detach().cpu()),
        "hash": float(con_loss.detach().cpu()),
        "active_codes": float(output["code_ids"].unique().numel()),
        "assignment_distance": float(output["assignment_distance"].mean().detach().cpu()),
    }
    return total, logs


def train_codebook(
    corpus_dir: str | Path,
    out_dir: str | Path,
    split: str = "train",
    codebook_size: int = 512,
    hidden_dim: int = 128,
    num_layers: int = 2,
    num_heads: int = 4,
    dropout: float = 0.1,
    epochs: int = 1,
    batch_size: int = 64,
    lr: float = 2e-4,
    beta: float = 0.25,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    loss_weights: dict[str, float] | None = None,
    device: str | None = None,
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    corpus = Path(corpus_dir)
    manifest = load_corpus_manifest(corpus)
    spec = TensorSpec(
        atom_vocab=tuple(manifest["atom_vocab"]),
        bond_vocab=tuple(manifest["bond_vocab"]),
        max_nodes=int(manifest["max_nodes"]),
    )
    dataset = CheckpointPoolDataset(corpus, split=split, spec=spec)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_checkpoint_pool)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = CheckpointCodebookModel(
        node_feature_dim=spec.node_feature_dim,
        edge_dim=spec.edge_dim,
        max_nodes=spec.max_nodes,
        codebook_size=codebook_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        sums: dict[str, float] = {}
        steps = 0
        for batch in loader:
            state = {k: v.to(dev) for k, v in batch["state"].items()}
            batch_dev = dict(batch)
            batch_dev["state"] = state
            batch_dev["ring_count"] = batch["ring_count"].to(dev)
            batch_dev["core_count"] = batch["core_count"].to(dev)
            out = model(state)
            loss, logs = codebook_loss(out, batch_dev, beta=beta, weights=loss_weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            for key, value in logs.items():
                sums[key] = sums.get(key, 0.0) + value
            steps += 1
        epoch_logs = {key: value / max(1, steps) for key, value in sums.items()}
        epoch_logs["epoch"] = float(epoch)
        history.append(epoch_logs)
        print(json.dumps(epoch_logs, sort_keys=True))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "spec": {
            "atom_vocab": list(spec.atom_vocab),
            "bond_vocab": list(spec.bond_vocab),
            "max_nodes": spec.max_nodes,
        },
        "config": {
            "codebook_size": codebook_size,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "dropout": dropout,
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "beta": beta,
            "weight_decay": weight_decay,
            "grad_clip": grad_clip,
            "seed": int(seed),
            "loss_weights": dict(loss_weights or {}),
        },
        "history": history,
        "architecture": {
            "input": "T(O(H))",
            "certified_object_format": CERTIFIED_OBJECT_FORMAT,
            "node_feature_dim": spec.node_feature_dim,
            "edge_dim": spec.edge_dim,
            "masked_mean_pooling": True,
            "codebook_capacity": codebook_size,
            "code_dimension": hidden_dim,
        },
        "format": "checkpoint_dag_diffusion.global_codebook.v2",
    }
    checkpoint["artifact_fingerprint"] = artifact_fingerprint(checkpoint)
    path = out / "codebook.pt"
    torch.save(checkpoint, path)
    (out / "codebook_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"codebook": str(path), "history": history}


def load_codebook_model(path: str | Path, device: torch.device) -> tuple[CheckpointCodebookModel, TensorSpec, dict]:
    ckpt = validate_codebook(
        torch.load(path, map_location=device),
        source=f"global codebook {Path(path)}",
    )
    spec = TensorSpec(
        atom_vocab=tuple(ckpt["spec"]["atom_vocab"]),
        bond_vocab=tuple(ckpt["spec"]["bond_vocab"]),
        max_nodes=int(ckpt["spec"]["max_nodes"]),
    )
    cfg = ckpt["config"]
    model = CheckpointCodebookModel(
        node_feature_dim=spec.node_feature_dim,
        edge_dim=spec.edge_dim,
        max_nodes=spec.max_nodes,
        codebook_size=int(cfg["codebook_size"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_layers=int(cfg["num_layers"]),
        num_heads=int(cfg["num_heads"]),
        dropout=float(cfg.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, spec, ckpt


@torch.no_grad()
def assign_codebook(
    corpus_dir: str | Path,
    codebook_path: str | Path,
    out_dir: str | Path,
    split: str = "train",
    batch_size: int = 128,
    device: str | None = None,
) -> dict:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, spec, ckpt = load_codebook_model(codebook_path, dev)
    dataset = CheckpointPoolDataset(corpus_dir, split=split, spec=spec)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_checkpoint_pool)
    entries: list[dict[str, Any]] = []
    support = Counter()
    distance_sum = Counter()
    for batch in loader:
        state = {k: v.to(dev) for k, v in batch["state"].items()}
        embedding = model.encode(state)
        ids, _, distances = model.quantize(embedding)
        for row in range(ids.numel()):
            code_id = int(ids[row].cpu())
            dist = float(distances[row].cpu())
            support[code_id] += 1
            distance_sum[code_id] += dist
            entries.append(
                {
                    "record_index": int(batch["record_index"][row]),
                    "checkpoint_index": int(batch["checkpoint_index"][row]),
                    "level": int(batch["level"][row]),
                    "hash": str(batch["hash"][row]),
                    "global_id": code_id,
                    "assignment_distance": dist,
                }
            )
    active = len(support)
    total = sum(support.values())
    probs = torch.tensor([support.get(idx, 0) / max(1, total) for idx in range(int(ckpt["config"]["codebook_size"]))])
    entropy = float(-(probs[probs > 0] * probs[probs > 0].log()).sum().item())
    payload = {
        "format": ASSIGNMENTS_FORMAT,
        "method": "learned_codebook",
        "split": split,
        "codebook_path": str(codebook_path),
        "codebook_size": int(ckpt["config"]["codebook_size"]),
        "source_codebook_fingerprint": str(ckpt["artifact_fingerprint"]),
        "assignments": entries,
        "summary": {
            "num_assignments": len(entries),
            "active_codes": active,
            "active_code_fraction": active / max(1, int(ckpt["config"]["codebook_size"])),
            "code_entropy": entropy,
            "support_counts": dict(sorted((str(k), int(v)) for k, v in support.items())),
            "mean_assignment_distance": {
                str(k): float(distance_sum[k] / max(1, support[k])) for k in sorted(support)
            },
        },
    }
    payload["artifact_fingerprint"] = artifact_fingerprint(payload)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "global_assignments.pt"
    torch.save(payload, path)
    (out / "global_assignments_summary.json").write_text(json.dumps(payload["summary"], indent=2), encoding="utf-8")
    return {"assignments": str(path), "summary": payload["summary"]}
