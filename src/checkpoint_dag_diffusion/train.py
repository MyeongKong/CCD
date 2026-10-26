from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from .dataset import CheckpointDAGEdgeDataset, collate_edges
from .artifacts import CERTIFIED_OBJECT_FORMAT, artifact_fingerprint, load_corpus_manifest
from .model import CheckpointDAGDenoiser
from .noise import DiscreteGraphNoise, estimate_activity_marginal, estimate_marginals
from .tensorize import TensorSpec


def load_config(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)
    return json.loads(text)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_diffusion_contract(config: dict[str, Any]) -> None:
    """Reject configurations that request a non-paper corruption contract."""

    diffusion = dict(config.get("diffusion", {}))
    if diffusion.get("node_corruption", "mixed_state") != "mixed_state":
        raise ValueError("CCD requires diffusion.node_corruption=mixed_state.")
    for field in (
        "level0_atom_categorical",
        "binary_object_logits",
        "categorical_activity",
        "categorical_reverse_posterior",
        "activity_capacity_mask",
        "factorized_edge_activity_posterior",
    ):
        if field in diffusion and diffusion[field] is not True:
            raise ValueError(f"CCD requires diffusion.{field}=true.")


def _masked_node_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(dtype=pred.dtype)
    return (((pred - target) ** 2) * weights).sum() / weights.sum().clamp_min(1.0)


def _mixed_node_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    relation: torch.Tensor,
    spec: TensorSpec,
) -> torch.Tensor:
    atom_dim = spec.atom_feature_dim
    level0 = relation[:, 1].view(-1, 1) == 0
    categorical_mask = mask & level0
    coarse_mask = mask & ~level0
    numerator = pred.sum() * 0.0
    if categorical_mask.any():
        atom_target = target[..., :atom_dim].argmax(dim=-1)
        numerator = numerator + F.cross_entropy(
            pred[..., :atom_dim][categorical_mask],
            atom_target[categorical_mask],
            reduction="sum",
        )
    if coarse_mask.any():
        numerator = numerator + ((pred[..., :atom_dim] - target[..., :atom_dim]) ** 2)[coarse_mask].sum()

    binary_indices = tuple(int(index) for index in spec.categorical_node_indices)
    if binary_indices:
        binary_pred = pred[..., list(binary_indices)]
        binary_target = target[..., list(binary_indices)]
        binary_loss = F.binary_cross_entropy_with_logits(binary_pred, binary_target, reduction="none")
        numerator = numerator + (binary_loss * mask.unsqueeze(-1)).sum()

    numerical_indices = [
        index
        for index in range(atom_dim, spec.node_feature_dim)
        if index not in binary_indices and index not in spec.reserved_node_indices
    ]
    if numerical_indices:
        numerical_error = (pred[..., numerical_indices] - target[..., numerical_indices]) ** 2
        numerator = numerator + (numerical_error * mask.unsqueeze(-1)).sum()
    return numerator / mask.sum().clamp_min(1).to(dtype=pred.dtype)


def _edge_ce(edge_logits: torch.Tensor, edge_target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
    pair_mask = torch.triu(pair_mask, diagonal=1)
    if not pair_mask.any():
        return edge_logits.sum() * 0.0
    return F.cross_entropy(edge_logits[pair_mask], edge_target[pair_mask].long())


def _masked_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if mask is None:
        return loss.mean()
    weights = mask.to(dtype=loss.dtype)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def _guide_aggregation_loss(outputs, batch: dict, spec: TensorSpec) -> torch.Tensor:
    """Enforce the full coarse-to-fine node part of correspondence ``mu``.

    One guide block may expand into several fine nodes.  Direct node copying
    is valid only for the preserved one-to-one subset; for every other mapped
    group we compare additive object descriptors after aggregating the
    predicted fine nodes.  The expected number of newly visible internal
    edges is added back to the predicted hidden-bond counts before comparison.
    """

    correspondence_mask = batch.get("correspondence_mask")
    target_to_guide = batch.get("target_to_guide")
    if correspondence_mask is None or target_to_guide is None:
        return outputs.node_features.sum() * 0.0
    losses: list[torch.Tensor] = []
    atom_dim = spec.atom_feature_dim
    none_idx = spec.bond_vocab.index("none")
    edge_prob = torch.softmax(outputs.edge_logits, dim=-1)
    for batch_index in range(outputs.node_features.size(0)):
        if not bool(batch["has_guide"][batch_index].item()):
            continue
        valid_targets = torch.nonzero(
            correspondence_mask[batch_index] & batch["target"]["mask"][batch_index],
            as_tuple=False,
        ).flatten()
        if valid_targets.numel() == 0:
            continue
        for guide_index in torch.unique(target_to_guide[batch_index, valid_targets]).tolist():
            guide_index = int(guide_index)
            if guide_index < 0 or not bool(batch["guide"]["mask"][batch_index, guide_index].item()):
                continue
            children = valid_targets[target_to_guide[batch_index, valid_targets] == guide_index]
            predicted = outputs.node_features[batch_index, children]
            if int(batch["relation"][batch_index, 1].item()) == 0:
                atom_prediction = torch.softmax(predicted[:, :atom_dim], dim=-1)
            else:
                atom_prediction = predicted[:, :atom_dim]
            guide_x = batch["guide"]["x"][batch_index, guide_index]
            aggregate_prediction = torch.cat(
                [
                    atom_prediction.sum(dim=0),
                    predicted[:, atom_dim].sum().view(1),
                    predicted[:, atom_dim + 2].sum().view(1),
                    predicted[:, atom_dim + 3].sum().view(1),
                ]
            )
            aggregate_target = torch.cat(
                [
                    guide_x[:atom_dim],
                    guide_x[atom_dim].view(1),
                    guide_x[atom_dim + 2].view(1),
                    guide_x[atom_dim + 3].view(1),
                ]
            )
            losses.append(F.mse_loss(aggregate_prediction, aggregate_target))

            hidden_prediction = predicted[:, atom_dim + 1].sum()
            if children.numel() > 1:
                left, right = torch.triu_indices(children.numel(), children.numel(), offset=1)
                child_left = children[left]
                child_right = children[right]
                hidden_prediction = hidden_prediction + (
                    1.0 - edge_prob[batch_index, child_left, child_right, none_idx]
                ).sum()
            losses.append((hidden_prediction - guide_x[atom_dim + 1]).square())
    if not losses:
        return outputs.node_features.sum() * 0.0
    return torch.stack(losses).mean()


def compute_loss(outputs, batch: dict, spec: TensorSpec, weights: dict[str, float]) -> tuple[torch.Tensor, dict[str, float]]:
    target = batch["target"]
    mask = target["mask"]
    node_loss = _mixed_node_loss(outputs.node_features, target["x"], mask, batch["relation"], spec)
    edge_loss = _edge_ce(outputs.edge_logits, target["edge"], mask)
    active_loss = _masked_bce(outputs.active_logits, mask.float())

    atom_dim = len(spec.atom_vocab)
    unresolved = (target["x"][..., atom_dim + 4] > 0.5).float()
    split_loss = _masked_bce(outputs.split_logits, unresolved, mask)
    attachment_target = target.get("attachment", torch.zeros_like(target["core"]))
    attachment_loss = _masked_bce(outputs.attachment_logits, attachment_target, mask)

    has_guide = batch["has_guide"]
    overlap_mask = batch.get("overlap_mask", torch.zeros_like(mask))
    overlap_mask = overlap_mask & mask & has_guide.view(-1, 1)
    if overlap_mask.any():
        target_to_guide = batch.get("target_to_guide")
        if target_to_guide is None:
            preservation_target = target["x"]
        else:
            gather_index = target_to_guide.clamp_min(0).unsqueeze(-1).expand(-1, -1, target["x"].size(-1))
            preservation_target = torch.gather(batch["guide"]["x"], dim=1, index=gather_index)
        guide_preservation = _mixed_node_loss(
            outputs.node_features,
            preservation_target,
            overlap_mask,
            batch["relation"],
            spec,
        )
    else:
        guide_preservation = node_loss * 0.0
    guide_aggregation = _guide_aggregation_loss(outputs, batch, spec)
    boundary_mask = batch.get("boundary_expansion_mask", torch.zeros_like(mask))
    boundary_mask = boundary_mask & mask & has_guide.view(-1, 1)
    boundary_correspondence = (
        _masked_bce(outputs.attachment_logits, attachment_target, boundary_mask)
        if boundary_mask.any()
        else node_loss * 0.0
    )
    guide_consistency = guide_preservation + guide_aggregation + boundary_correspondence

    total = (
        weights.get("node", 1.0) * node_loss
        + weights.get("edge", 1.0) * edge_loss
        + weights.get("active", 0.25) * active_loss
        + weights.get("split", 0.05) * split_loss
        + weights.get("attachment", 0.05) * attachment_loss
        + weights.get("consistency", 0.1) * guide_consistency
    )
    logs = {
        "loss": float(total.detach().cpu()),
        "node": float(node_loss.detach().cpu()),
        "edge": float(edge_loss.detach().cpu()),
        "active": float(active_loss.detach().cpu()),
        "split": float(split_loss.detach().cpu()),
        "attachment": float(attachment_loss.detach().cpu()),
        "consistency": float(guide_consistency.detach().cpu()),
        "guide_preservation": float(guide_preservation.detach().cpu()),
        "guide_aggregation": float(guide_aggregation.detach().cpu()),
        "boundary_correspondence": float(boundary_correspondence.detach().cpu()),
    }
    return total, logs


def _move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            out[key] = {k: v.to(device) for k, v in value.items()}
        elif torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def train_from_config(config_path: str | Path, corpus_dir: str | Path | None = None, run_dir: str | Path | None = None) -> dict:
    cfg = load_config(config_path)
    validate_diffusion_contract(cfg)
    seed_all(int(cfg.get("seed", 0)))
    corpus = Path(corpus_dir or cfg["data"]["corpus_dir"])
    run = Path(run_dir or cfg["train"].get("run_dir", "runs/checkpoint_dag_diffusion"))
    run.mkdir(parents=True, exist_ok=True)
    manifest = load_corpus_manifest(corpus)
    spec = TensorSpec(
        atom_vocab=tuple(manifest["atom_vocab"]),
        bond_vocab=tuple(manifest["bond_vocab"]),
        max_nodes=int(manifest["max_nodes"]),
    )
    max_level = int(manifest.get("max_level", 8)) + 1
    max_delta = max(int(k) for k in manifest.get("edge_delta_counts", {"1": 1}).keys())

    train_set = CheckpointDAGEdgeDataset(corpus, split="train", spec=spec)
    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg["train"].get("batch_size", 16)),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        collate_fn=collate_edges,
    )
    node_marginal, edge_marginal = estimate_marginals(train_set, spec.node_feature_dim, spec.edge_dim)
    activity_marginal = estimate_activity_marginal(train_set)
    device = torch.device(cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = CheckpointDAGDenoiser(
        node_feature_dim=spec.node_feature_dim,
        edge_dim=spec.edge_dim,
        hidden_dim=int(cfg["model"].get("hidden_dim", 192)),
        num_layers=int(cfg["model"].get("num_layers", 4)),
        guide_num_layers=int(cfg["model"].get("guide_num_layers", 2)),
        num_heads=int(cfg["model"].get("num_heads", 6)),
        dropout=float(cfg["model"].get("dropout", 0.1)),
        max_timesteps=int(cfg["diffusion"].get("timesteps", 200)),
        max_level=max_level,
        max_delta=max_delta,
    ).to(device)
    noise = DiscreteGraphNoise(
        timesteps=int(cfg["diffusion"].get("timesteps", 200)),
        node_marginal=node_marginal,
        edge_marginal=edge_marginal,
        node_noise_scale=float(cfg["diffusion"].get("node_noise_scale", 0.01)),
        categorical_node_indices=spec.categorical_node_indices,
        reserved_node_indices=spec.reserved_node_indices,
        activity_marginal=activity_marginal,
        atom_feature_dim=spec.atom_feature_dim,
    )
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"].get("lr", 2e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )
    cond_drop_prob = float(cfg["train"].get("cond_drop_prob", 0.1))
    loss_weights = dict(cfg.get("loss", {}))
    history: list[dict[str, float]] = []
    epochs = int(cfg["train"].get("epochs", 1))
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))

    for epoch in range(epochs):
        model.train()
        sums: dict[str, float] = {}
        steps = 0
        for batch in train_loader:
            batch = _move_batch(batch, device)
            target = batch["target"]
            noisy = noise.corrupt(
                target["x"],
                target["edge"],
                target["mask"],
                target_levels=batch["relation"][:, 1],
            )
            has_guide = batch["has_guide"].clone()
            if cond_drop_prob > 0:
                drop = torch.rand_like(has_guide.float()) < cond_drop_prob
                has_guide = has_guide & ~drop
            outputs = model(
                x_t=noisy.x_t,
                edge_t=noisy.edge_t,
                # Every padded slot is a denoising candidate.  Noisy activity
                # is encoded by zero node fields/no-edge incidence, while the
                # capacity mask lets inactive candidates receive t/relation
                # and conditioning signals needed for reactivation.
                mask=torch.ones_like(noisy.active_t),
                t=noisy.t,
                relation=batch["relation"],
                guide_x=batch["guide"]["x"],
                guide_edge=batch["guide"]["edge"],
                guide_mask=batch["guide"]["mask"],
                has_guide=has_guide,
            )
            loss, logs = compute_loss(outputs, batch, spec=spec, weights=loss_weights)
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

    checkpoint = {
        "model_state": model.state_dict(),
        "config": cfg,
        "spec": {
            "atom_vocab": list(spec.atom_vocab),
            "bond_vocab": list(spec.bond_vocab),
            "max_nodes": spec.max_nodes,
        },
        "node_marginal": node_marginal,
        "edge_marginal": edge_marginal,
        "activity_marginal": activity_marginal,
        "max_level": max_level,
        "max_delta": max_delta,
        "history": history,
        "architecture": {
            "input": "corrupted T(O(H_f)), clean T(O(H_g))",
            "certified_object_format": CERTIFIED_OBJECT_FORMAT,
            "relation_order": ["guide_level", "target_level", "delta", "operator_id"],
            "mixed_state_corruption": True,
            "categorical_node_indices": list(spec.categorical_node_indices),
            "categorical_activity": True,
            "categorical_reverse_posterior": True,
            "level0_atom_categorical": True,
            "atom_feature_dim": spec.atom_feature_dim,
            "binary_object_logits": True,
            "activity_capacity_mask": True,
            "factorized_edge_activity_posterior": True,
            "cfg_all_heads": True,
            "semantic_slot_alignment": True,
            "full_correspondence_supervision": True,
            "nested_generated_lineage": True,
            "boundary_slot_aliases": True,
        },
        "format": "checkpoint_dag_diffusion.local_denoiser.v2",
    }
    checkpoint["artifact_fingerprint"] = artifact_fingerprint(checkpoint)
    ckpt_dir = run / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "last.pt"
    torch.save(checkpoint, ckpt_path)
    (run / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"checkpoint": str(ckpt_path), "history": history, "run_dir": str(run)}
