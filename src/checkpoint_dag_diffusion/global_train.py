from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from .artifacts import CERTIFIED_OBJECT_FORMAT, artifact_fingerprint
from .global_dataset import GlobalCheckpointEdgeDataset, collate_global_edges
from .global_graph import code_count, load_global_graph, validate_level_augmented_dag
from .model import CheckpointDAGDenoiser
from .noise import DiscreteGraphNoise, estimate_activity_marginal, estimate_marginals
from .tensorize import TensorSpec
from .train import compute_loss, validate_diffusion_contract


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
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


def train_global_from_config(
    config_path: str | Path | None,
    corpus_dir: str | Path,
    global_graph_path: str | Path,
    assignments_path: str | Path,
    run_dir: str | Path,
    epochs: int | None = None,
    batch_size: int | None = None,
    device: str | None = None,
) -> dict:
    cfg = load_config(config_path)
    validate_diffusion_contract(cfg)
    seed_all(int(cfg.get("seed", 0)))
    graph = load_global_graph(global_graph_path)
    if graph.get("planning_nodes"):
        validate_level_augmented_dag(graph)
    spec = TensorSpec(
        atom_vocab=tuple(graph["spec"]["atom_vocab"]),
        bond_vocab=tuple(graph["spec"]["bond_vocab"]),
        max_nodes=int(graph["spec"]["max_nodes"]),
    )
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    train_cfg = dict(cfg.get("train", {}))
    model_cfg = dict(cfg.get("model", {}))
    diffusion_cfg = dict(cfg.get("diffusion", {}))
    loss_cfg = dict(cfg.get("loss", {}))

    dataset = GlobalCheckpointEdgeDataset(
        corpus_dir=corpus_dir,
        global_graph_path=global_graph_path,
        assignments_path=assignments_path,
        split="train",
        spec=spec,
    )
    effective_batch_size = int(batch_size or train_cfg.get("batch_size", 16))
    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collate_global_edges,
    )
    node_marginal, edge_marginal = estimate_marginals(dataset, spec.node_feature_dim, spec.edge_dim)
    activity_marginal = estimate_activity_marginal(dataset)
    dev = torch.device(device or train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    canonical_edges = list(graph.get("initialization_edges", [])) + list(graph.get("planning_edges", []))
    if canonical_edges:
        max_delta = max(int(edge.get("delta", edge.get("relation", {}).get("delta", 1))) for edge in canonical_edges)
    else:
        max_delta = max(int(k) for edge in graph["global_edges"] for k in edge.get("delta_hist", {"1": 1}).keys())
    if graph.get("planning_nodes"):
        max_level = max(int(node["level"]) for node in graph["planning_nodes"]) + 1
    else:
        max_level = max(int(k) for node in graph["global_nodes"] for k in node.get("level_hist", {"0": 0}).keys()) + 1
    num_condition_edges = int(graph.get("num_condition_edges", graph["num_global_edges"]))
    model = CheckpointDAGDenoiser(
        node_feature_dim=spec.node_feature_dim,
        edge_dim=spec.edge_dim,
        hidden_dim=int(model_cfg.get("hidden_dim", 192)),
        num_layers=int(model_cfg.get("num_layers", 4)),
        guide_num_layers=int(model_cfg.get("guide_num_layers", 2)),
        num_heads=int(model_cfg.get("num_heads", 6)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        max_timesteps=int(diffusion_cfg.get("timesteps", 200)),
        max_level=max_level,
        max_delta=max_delta,
        num_global_codes=code_count(graph),
        max_global_edges=num_condition_edges,
    ).to(dev)
    noise = DiscreteGraphNoise(
        timesteps=int(diffusion_cfg.get("timesteps", 200)),
        node_marginal=node_marginal,
        edge_marginal=edge_marginal,
        node_noise_scale=float(diffusion_cfg.get("node_noise_scale", 0.01)),
        categorical_node_indices=spec.categorical_node_indices,
        reserved_node_indices=spec.reserved_node_indices,
        activity_marginal=activity_marginal,
        atom_feature_dim=spec.atom_feature_dim,
    )
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    cond_drop_prob = float(train_cfg.get("cond_drop_prob", 0.1))
    global_drop_prob = float(train_cfg.get("global_drop_prob", cond_drop_prob))
    total_epochs = int(epochs or train_cfg.get("epochs", 1))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    train_cfg.update(
        {
            "batch_size": effective_batch_size,
            "device": str(dev),
            "epochs": total_epochs,
        }
    )
    history: list[dict[str, float]] = []

    for epoch in range(total_epochs):
        model.train()
        sums: dict[str, float] = {}
        steps = 0
        for batch in loader:
            batch = _move_batch(batch, dev)
            target = batch["target"]
            noisy = noise.corrupt(
                target["x"],
                target["edge"],
                target["mask"],
                target_levels=batch["relation"][:, 1],
            )
            has_guide = batch["has_guide"].clone()
            has_global = torch.ones_like(has_guide, dtype=torch.bool)
            if cond_drop_prob > 0:
                has_guide = has_guide & ~(torch.rand_like(has_guide.float()) < cond_drop_prob)
            if global_drop_prob > 0:
                has_global = has_global & ~(torch.rand_like(has_global.float()) < global_drop_prob)
            outputs = model(
                x_t=noisy.x_t,
                edge_t=noisy.edge_t,
                mask=torch.ones_like(noisy.active_t),
                t=noisy.t,
                relation=batch["relation"],
                guide_x=batch["guide"]["x"],
                guide_edge=batch["guide"]["edge"],
                guide_mask=batch["guide"]["mask"],
                has_guide=has_guide,
                guide_global_id=batch["guide_condition_id"],
                target_global_id=batch["target_condition_id"],
                global_edge_id=batch["global_edge_id"],
                has_global_condition=has_global,
            )
            loss, logs = compute_loss(outputs, batch, spec=spec, weights=loss_cfg)
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
        "config": {
            **cfg,
            "model": model_cfg,
            "train": train_cfg,
            "diffusion": diffusion_cfg,
            "loss": loss_cfg,
        },
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
        "num_global_codes": code_count(graph),
        "max_global_edges": num_condition_edges,
        "global_graph_path": str(global_graph_path),
        "assignments_path": str(assignments_path),
        "global_graph_fingerprint": str(graph["artifact_fingerprint"]),
        "assignments_fingerprint": str(dataset.assignments["artifact_fingerprint"]),
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
            "target_layers": int(model_cfg.get("num_layers", 4)),
            "guide_layers": int(model_cfg.get("guide_num_layers", 2)),
            "hidden_dim": int(model_cfg.get("hidden_dim", 192)),
            "heads": int(model_cfg.get("num_heads", 6)),
            "level_augmented_global_dag": bool(graph.get("planning_nodes")),
            "compact_condition_ids": bool(graph.get("num_global_conditions") is not None),
        },
        "format": "checkpoint_dag_diffusion.global_denoiser.v2",
    }
    checkpoint["artifact_fingerprint"] = artifact_fingerprint(checkpoint)
    ckpt_dir = run / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "last.pt"
    torch.save(checkpoint, ckpt_path)
    (run / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"checkpoint": str(ckpt_path), "history": history, "run_dir": str(run)}
