from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random
from typing import Any

import torch

from .artifacts import validate_local_denoiser
from .certified_object import certified_object_from_state
from .checkpoint_state import CheckpointState
from .dataset import CheckpointDAGEdgeDataset
from .metrics import summarize_samples
from .model import CheckpointDAGDenoiser, DenoiserOutput
from .noise import DiscreteGraphNoise
from .tensorize import TensorSpec, empty_condition, relation_to_tensor, state_to_tensors, tensors_to_state


def load_model_checkpoint(path: str | Path, device: torch.device) -> tuple[CheckpointDAGDenoiser, TensorSpec, dict[str, Any]]:
    ckpt = validate_local_denoiser(
        torch.load(path, map_location=device),
        source=f"local denoiser {Path(path)}",
    )
    spec = TensorSpec(
        atom_vocab=tuple(ckpt["spec"]["atom_vocab"]),
        bond_vocab=tuple(ckpt["spec"]["bond_vocab"]),
        max_nodes=int(ckpt["spec"]["max_nodes"]),
    )
    cfg = ckpt["config"]
    model = CheckpointDAGDenoiser(
        node_feature_dim=spec.node_feature_dim,
        edge_dim=spec.edge_dim,
        hidden_dim=int(cfg["model"].get("hidden_dim", 192)),
        num_layers=int(cfg["model"].get("num_layers", 4)),
        guide_num_layers=int(cfg["model"].get("guide_num_layers", 2)),
        num_heads=int(cfg["model"].get("num_heads", 6)),
        dropout=float(cfg["model"].get("dropout", 0.1)),
        max_timesteps=int(cfg["diffusion"].get("timesteps", 200)),
        max_level=int(ckpt.get("max_level", 16)),
        max_delta=int(ckpt.get("max_delta", 16)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, spec, ckpt


def _conditioned_forward(
    model: CheckpointDAGDenoiser,
    x: torch.Tensor,
    edge: torch.Tensor,
    mask: torch.Tensor,
    t: torch.Tensor,
    relation: torch.Tensor,
    guide: dict[str, torch.Tensor],
    has_guide: torch.Tensor,
    guidance_weight: float,
) -> DenoiserOutput:
    cond = model(x, edge, mask, t, relation, guide["x"], guide["edge"], guide["mask"], has_guide)
    if guidance_weight == 1.0 or not bool(has_guide.any().item()):
        return cond
    uncond = model(x, edge, mask, t, relation, guide["x"], guide["edge"], guide["mask"], torch.zeros_like(has_guide))
    return DenoiserOutput(
        node_features=uncond.node_features + guidance_weight * (cond.node_features - uncond.node_features),
        edge_logits=uncond.edge_logits + guidance_weight * (cond.edge_logits - uncond.edge_logits),
        active_logits=uncond.active_logits + guidance_weight * (cond.active_logits - uncond.active_logits),
        split_logits=uncond.split_logits + guidance_weight * (cond.split_logits - uncond.split_logits),
        attachment_logits=uncond.attachment_logits
        + guidance_weight * (cond.attachment_logits - uncond.attachment_logits),
    )


def _sample_edges(
    edge_logits: torch.Tensor,
    mask: torch.Tensor,
    none_idx: int,
    temperature: float,
    none_bias: float,
    max_degree: int,
) -> torch.Tensor:
    logits = edge_logits.clone()
    logits[..., none_idx] += none_bias
    logits = logits / max(temperature, 1e-6)
    probs = torch.softmax(logits, dim=-1)
    flat = probs.reshape(-1, probs.size(-1))
    sampled = torch.multinomial(flat, 1).view(*probs.shape[:-1])
    sampled = torch.triu(sampled, diagonal=1)
    sampled = sampled + sampled.transpose(1, 2)
    diag = torch.arange(sampled.size(1), device=sampled.device)
    sampled[:, diag, diag] = none_idx
    valid_pair = mask.unsqueeze(1) & mask.unsqueeze(2)
    sampled = torch.where(valid_pair, sampled, torch.full_like(sampled, none_idx))

    if max_degree > 0:
        for batch_idx in range(sampled.size(0)):
            active = mask[batch_idx].nonzero(as_tuple=False).flatten().tolist()
            for node in active:
                incident = [nbr for nbr in active if nbr != node and int(sampled[batch_idx, node, nbr].item()) != none_idx]
                while len(incident) > max_degree:
                    drop = random.choice(incident)
                    sampled[batch_idx, node, drop] = none_idx
                    sampled[batch_idx, drop, node] = none_idx
                    incident.remove(drop)
    return sampled


def _terminal_graph_state(
    noise: DiscreteGraphNoise,
    spec: TensorSpec,
    device: torch.device,
    *,
    categorical_atom: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample the terminal node, edge, and activity marginals.

    Atom channels represent categorical atom identities only for a level-0
    target.  At coarser levels they remain numerical block-composition
    summaries, so they must not be projected onto a one-hot simplex.
    """

    mask = torch.rand(1, spec.max_nodes, device=device) < noise.activity_marginal
    if not bool(mask.any().item()):
        mask[:, 0] = True
    x = noise.node_marginal.to(device).view(1, 1, -1).expand(1, spec.max_nodes, -1).clone()
    x = x + torch.randn_like(x) * noise.node_noise_scale
    if categorical_atom:
        atom_dim = int(noise.atom_feature_dim)
        if atom_dim <= 0:
            raise ValueError("Level-0 categorical atom sampling requires atom_feature_dim > 0.")
        if atom_dim != spec.atom_feature_dim:
            raise ValueError(
                "Noise/TensorSpec atom dimensions disagree: "
                f"{atom_dim} != {spec.atom_feature_dim}."
            )
        atom_marginal = noise.node_marginal[:atom_dim].to(device=device).clamp_min(0.0)
        atom_marginal = atom_marginal / atom_marginal.sum().clamp_min(1e-8)
        atom_ids = torch.multinomial(atom_marginal, spec.max_nodes, replacement=True).view(1, -1)
        x[..., :atom_dim] = torch.nn.functional.one_hot(
            atom_ids, num_classes=atom_dim
        ).to(dtype=x.dtype)
    for index in spec.categorical_node_indices:
        probability = float(noise.node_marginal[index].clamp(0.0, 1.0))
        x[..., index] = (torch.rand(1, spec.max_nodes, device=device) < probability).to(dtype=x.dtype)
    for index in spec.reserved_node_indices:
        x[..., index] = 0.0
    x = x * mask.unsqueeze(-1).to(dtype=x.dtype)

    values = torch.multinomial(
        noise.edge_marginal.to(device),
        spec.max_nodes * spec.max_nodes,
        replacement=True,
    )
    edge = values.view(1, spec.max_nodes, spec.max_nodes).long()
    edge = torch.triu(edge, diagonal=1)
    edge = edge + edge.transpose(1, 2)
    none_idx = spec.bond_vocab.index("none")
    diagonal = torch.arange(spec.max_nodes, device=device)
    edge[:, diagonal, diagonal] = none_idx
    valid_pair = mask.unsqueeze(1) & mask.unsqueeze(2)
    edge = torch.where(valid_pair, edge, torch.full_like(edge, none_idx))
    return x, edge, mask


def _reverse_schedule(timesteps: int, evaluations: int, device: torch.device) -> list[tuple[int, int]]:
    count = max(1, min(int(timesteps), int(evaluations)))
    values = torch.linspace(timesteps, 1, count, device=device).round().long().tolist()
    times: list[int] = []
    for value in values:
        value = int(value)
        if not times or value < times[-1]:
            times.append(value)
    return list(zip(times, times[1:] + [0]))


def _representative_to_current_slots(state: CheckpointState) -> dict[int, int]:
    representative_slots = state.tensor_slot_ids or tuple(range(state.num_nodes))
    return {
        int(slot): current
        for current, slot in enumerate(representative_slots)
        if int(slot) >= 0
    }


def _realization_status(
    state: CheckpointState,
    guide_state: CheckpointState | None = None,
) -> dict[str, Any]:
    aliases = _representative_to_current_slots(state)
    known_nodes = len(aliases)
    if guide_state is None:
        nested_lineage = True
    else:
        guide_blocks = [set(int(value) for value in block) for block in guide_state.original_atom_ids]
        target_blocks = [set(int(value) for value in block) for block in state.original_atom_ids]
        nested_lineage = (
            bool(guide_blocks)
            and set().union(*guide_blocks) == set().union(*target_blocks)
            and all(any(block <= guide for guide in guide_blocks) for block in target_blocks)
        )
    known_boundary_slots = sum(
        slot.representative_slot_id is not None and int(slot.representative_slot_id) >= 0
        for slot in state.attachment_slots
    )
    return {
        "aligned_nodes": int(known_nodes),
        "total_nodes": int(state.num_nodes),
        "representative_alignment_complete": bool(known_nodes == state.num_nodes),
        "nested_partition_lineage": bool(nested_lineage),
        "aligned_boundary_slots": int(known_boundary_slots),
        "total_boundary_slots": int(len(state.attachment_slots)),
        "boundary_alignment_complete": bool(known_boundary_slots == len(state.attachment_slots)),
    }


@torch.no_grad()
def sample_edge_transition(
    model: CheckpointDAGDenoiser,
    spec: TensorSpec,
    noise: DiscreteGraphNoise,
    relation_edge: dict,
    num_nodes: int,
    guide_state: CheckpointState | None,
    device: torch.device,
    target_template_state: CheckpointState | None = None,
    guidance_weight: float = 1.3,
    steps: int | None = None,
    temperature: float = 0.9,
    none_bias: float = 2.0,
    max_degree: int = 4,
) -> CheckpointState:
    del num_nodes  # Target activity is generated, not leaked through a clean mask.
    categorical_atom = int(relation_edge["target_level"]) == 0
    x, edge, mask = _terminal_graph_state(
        noise,
        spec,
        device,
        categorical_atom=categorical_atom,
    )

    if guide_state is None:
        guide = {key: value.unsqueeze(0).to(device) for key, value in empty_condition(spec).items()}
        has_guide = torch.tensor([False], device=device)
    else:
        guide = {key: value.unsqueeze(0).to(device) for key, value in state_to_tensors(guide_state, spec).items()}
        has_guide = torch.tensor([True], device=device)
    relation = relation_to_tensor(relation_edge).view(1, -1).to(device)
    none_idx = spec.bond_vocab.index("none")
    out: DenoiserOutput | None = None
    preserved_target_slots: set[int] = set()
    for t_value, s_value in _reverse_schedule(noise.timesteps, int(steps or 32), device):
        t = torch.tensor([t_value], dtype=torch.long, device=device)
        capacity_mask = torch.ones_like(mask)
        out = _conditioned_forward(model, x, edge, capacity_mask, t, relation, guide, has_guide, guidance_weight)
        x, edge, mask = noise.reverse_step(
            clean_x=out.node_features,
            clean_edge_logits=out.edge_logits,
            clean_active_logits=out.active_logits,
            current_x=x,
            current_edge=edge,
            current_active=mask,
            t=t_value,
            s=s_value,
            none_idx=none_idx,
            edge_temperature=temperature,
            none_bias=none_bias,
            max_degree=max_degree,
            categorical_atom=categorical_atom,
        )
        if guide_state is not None:
            correspondence = relation_edge.get("correspondence", {})
            preserved = correspondence.get("preserved_target_to_guide", {})
            aligned: dict[int, int] = {}
            guide_slot_map = _representative_to_current_slots(guide_state)
            for target_idx, guide_idx in preserved.items():
                target_i = int(target_idx)
                guide_i = guide_slot_map.get(int(guide_idx), -1)
                if 0 <= target_i < spec.max_nodes and 0 <= guide_i < guide_state.num_nodes:
                    aligned[target_i] = guide_i
                    preserved_target_slots.add(target_i)
                    mask[:, target_i] = True
                    x[:, target_i] = guide["x"][:, guide_i]
            for target_i, guide_i in aligned.items():
                for target_j, guide_j in aligned.items():
                    edge[:, target_i, target_j] = guide["edge"][:, guide_i, guide_j]
    if out is None:  # pragma: no cover - the reverse schedule is never empty
        raise RuntimeError("The reverse denoising schedule produced no evaluations.")
    return tensors_to_state(
        x.squeeze(0),
        edge.squeeze(0),
        mask.squeeze(0),
        spec=spec,
        split_logits=out.split_logits.squeeze(0),
        attachment_logits=out.attachment_logits.squeeze(0),
        template_state=target_template_state,
        target_level=int(relation_edge["target_level"]),
        preserved_target_slots=preserved_target_slots,
        guide_state=guide_state,
        correspondence=relation_edge.get("correspondence", {}),
    )


def _load_shape_records(corpus_dir: str | Path, spec: TensorSpec, split: str = "train") -> list[dict]:
    dataset = CheckpointDAGEdgeDataset(corpus_dir, split=split, spec=spec)
    return dataset.records


def _choose_path(record: dict) -> list[dict]:
    by_guide: dict[int, list[dict]] = {}
    noise_edges: list[dict] = []
    for edge in record["dag_edges"]:
        if edge.get("guide") is None:
            noise_edges.append(edge)
        else:
            by_guide.setdefault(int(edge["guide"]), []).append(edge)
    if not noise_edges:
        raise ValueError("Record has no noise edge.")
    start = random.choice(noise_edges)
    path = [start]
    current = int(start["target"])
    while current != 0:
        candidates = [edge for edge in by_guide.get(current, []) if int(edge["target"]) < current]
        if not candidates:
            candidates = [edge for edge in record["dag_edges"] if edge.get("guide") == current - 1]
        if not candidates:
            break
        weights = [1.0 / max(1, int(edge["delta"])) for edge in candidates]
        edge = random.choices(candidates, weights=weights, k=1)[0]
        path.append(edge)
        current = int(edge["target"])
    return path


@torch.no_grad()
def sample_pipeline(
    checkpoint: str | Path,
    out_dir: str | Path,
    num_samples: int = 16,
    corpus_dir: str | Path | None = None,
    device: str | None = None,
    guidance: float = 1.3,
    temperature: float = 0.9,
    none_bias: float = 2.0,
    max_degree: int = 4,
    seed: int = 0,
) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, spec, ckpt = load_model_checkpoint(checkpoint, dev)
    noise = DiscreteGraphNoise(
        timesteps=int(ckpt["config"]["diffusion"].get("timesteps", 200)),
        node_marginal=ckpt["node_marginal"],
        edge_marginal=ckpt["edge_marginal"],
        node_noise_scale=float(ckpt["config"]["diffusion"].get("node_noise_scale", 0.01)),
        categorical_node_indices=spec.categorical_node_indices,
        reserved_node_indices=spec.reserved_node_indices,
        activity_marginal=float(ckpt.get("activity_marginal", 0.5)),
        atom_feature_dim=spec.atom_feature_dim,
    )
    local_denoiser_steps = int(ckpt["config"]["diffusion"].get("denoiser_evaluations", 32))
    if corpus_dir is None:
        raise ValueError("DAG sampling requires corpus_dir for empirical path shapes.")
    shape_records = _load_shape_records(corpus_dir, spec)
    final_samples: list[CheckpointState] = []
    traces: list[list[dict]] = []
    for _ in range(num_samples):
        record = random.choice(shape_records)
        path = _choose_path(record)
        current_state: CheckpointState | None = None
        trace: list[dict] = []
        for step_edge in path:
            guide_before_transition = current_state
            target_meta = record["checkpoints"][int(step_edge["target"])]
            target_nodes = len(target_meta["state"]["node_summaries"])
            current_state = sample_edge_transition(
                model=model,
                spec=spec,
                noise=noise,
                relation_edge=step_edge,
                num_nodes=target_nodes,
                guide_state=current_state,
                device=dev,
                target_template_state=CheckpointState.from_dict(target_meta["state"]),
                guidance_weight=1.0 if step_edge.get("guide") is None else guidance,
                steps=local_denoiser_steps,
                temperature=temperature,
                none_bias=none_bias,
                max_degree=max_degree,
            )
            trace.append(
                {
                    "target_level": int(step_edge["target_level"]),
                    "guide_level": int(step_edge["guide_level"]),
                    "delta": int(step_edge["delta"]),
                    "operator_id": int(step_edge["operator_id"]),
                    "target_nodes": target_nodes,
                    "realized_checkpoint_object": certified_object_from_state(
                        current_state,
                        level=int(step_edge["target_level"]),
                        mode=str(target_meta.get("mode", "unknown")),
                        radius=(
                            None
                            if target_meta.get("radius") is None
                            else int(target_meta["radius"])
                        ),
                    ).to_dict(),
                    "realization_status": _realization_status(
                        current_state,
                        guide_before_transition,
                    ),
                }
            )
        if current_state is not None:
            final_samples.append(current_state)
            traces.append(trace)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = [
        {"sample": sample.to_dict(), "path_trace": trace}
        for sample, trace in zip(final_samples, traces)
    ]
    (out / "samples.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    path_signatures = Counter(tuple((s["target_level"], s["guide_level"], s["delta"]) for s in trace) for trace in traces)
    summary = summarize_samples(final_samples)
    summary["path_diversity"] = {
        "num_unique_paths": len(path_signatures),
        "unique_path_fraction": len(path_signatures) / max(1, len(traces)),
        "path_length_hist": dict(Counter(len(trace) for trace in traces)),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
