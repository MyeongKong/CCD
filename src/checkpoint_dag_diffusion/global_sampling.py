from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from typing import Any, Sequence

import torch

from .artifacts import ArtifactVersionError, validate_global_denoiser
from .certified_object import certified_object_from_state
from .checkpoint_state import CheckpointState
from .global_graph import load_global_graph, validate_level_augmented_dag
from .metrics import structural_validity, summarize_samples
from .model import CheckpointDAGDenoiser, DenoiserOutput
from .noise import DiscreteGraphNoise
from .sampling import (
    _realization_status,
    _representative_to_current_slots,
    _reverse_schedule,
    _terminal_graph_state,
)
from .tensorize import TensorSpec, empty_condition, relation_to_tensor, state_to_tensors, tensors_to_state


def _parse_code(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.upper().startswith("C"):
        text = text[1:]
    return int(text)


def load_global_model_checkpoint(path: str | Path, device: torch.device) -> tuple[CheckpointDAGDenoiser, TensorSpec, dict[str, Any]]:
    ckpt = validate_global_denoiser(
        torch.load(path, map_location=device),
        source=f"global denoiser {Path(path)}",
    )
    spec = TensorSpec(
        atom_vocab=tuple(ckpt["spec"]["atom_vocab"]),
        bond_vocab=tuple(ckpt["spec"]["bond_vocab"]),
        max_nodes=int(ckpt["spec"]["max_nodes"]),
    )
    cfg = ckpt["config"]
    model_cfg = cfg.get("model", {})
    diffusion_cfg = cfg.get("diffusion", {})
    model = CheckpointDAGDenoiser(
        node_feature_dim=spec.node_feature_dim,
        edge_dim=spec.edge_dim,
        hidden_dim=int(model_cfg.get("hidden_dim", 192)),
        num_layers=int(model_cfg.get("num_layers", 4)),
        guide_num_layers=int(model_cfg.get("guide_num_layers", 2)),
        num_heads=int(model_cfg.get("num_heads", 6)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        max_timesteps=int(diffusion_cfg.get("timesteps", 200)),
        max_level=int(ckpt.get("max_level", 16)),
        max_delta=int(ckpt.get("max_delta", 16)),
        num_global_codes=int(ckpt.get("num_global_codes", 0)),
        max_global_edges=int(ckpt.get("max_global_edges", 0)),
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
    guide_global_id: torch.Tensor,
    target_global_id: torch.Tensor,
    global_edge_id: torch.Tensor,
    guidance_weight: float,
) -> DenoiserOutput:
    cond = model(
        x,
        edge,
        mask,
        t,
        relation,
        guide["x"],
        guide["edge"],
        guide["mask"],
        has_guide,
        guide_global_id=guide_global_id,
        target_global_id=target_global_id,
        global_edge_id=global_edge_id,
        has_global_condition=torch.ones_like(target_global_id, dtype=torch.bool),
    )
    if guidance_weight == 1.0:
        return cond
    uncond = model(
        x,
        edge,
        mask,
        t,
        relation,
        guide["x"],
        guide["edge"],
        guide["mask"],
        torch.zeros_like(has_guide),
        guide_global_id=guide_global_id,
        target_global_id=target_global_id,
        global_edge_id=global_edge_id,
        has_global_condition=torch.zeros_like(target_global_id, dtype=torch.bool),
    )
    return DenoiserOutput(
        node_features=uncond.node_features + guidance_weight * (cond.node_features - uncond.node_features),
        edge_logits=uncond.edge_logits + guidance_weight * (cond.edge_logits - uncond.edge_logits),
        active_logits=uncond.active_logits + guidance_weight * (cond.active_logits - uncond.active_logits),
        split_logits=uncond.split_logits + guidance_weight * (cond.split_logits - uncond.split_logits),
        attachment_logits=uncond.attachment_logits
        + guidance_weight * (cond.attachment_logits - uncond.attachment_logits),
    )


@torch.no_grad()
def sample_global_transition(
    model: CheckpointDAGDenoiser,
    spec: TensorSpec,
    noise: DiscreteGraphNoise,
    relation_edge: dict,
    global_edge_id: int,
    target_condition_id: int,
    guide_condition_id: int,
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
    del num_nodes  # The active-node state is part of the reverse process.
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
    guide_gid = torch.tensor([guide_condition_id], dtype=torch.long, device=device)
    target_gid = torch.tensor([target_condition_id], dtype=torch.long, device=device)
    global_eid = torch.tensor([global_edge_id], dtype=torch.long, device=device)
    none_idx = spec.bond_vocab.index("none")
    out: DenoiserOutput | None = None
    preserved_target_slots: set[int] = set()
    for t_value, s_value in _reverse_schedule(noise.timesteps, int(steps or 32), device):
        t = torch.tensor([t_value], dtype=torch.long, device=device)
        out = _conditioned_forward(
            model,
            x,
            edge,
            torch.ones_like(mask),
            t,
            relation,
            guide,
            has_guide,
            guide_gid,
            target_gid,
            global_eid,
            guidance_weight,
        )
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


def _weighted_choice(items: list[dict], count_key: str = "count", temperature: float = 1.0) -> dict:
    if not items:
        raise ValueError("Cannot sample from an empty list.")
    temp = max(temperature, 1e-6)
    weights = [float(item.get(count_key, 1)) ** (1.0 / temp) for item in items]
    return random.choices(items, weights=weights, k=1)[0]


def _node_lookup(graph: dict) -> dict[int, dict]:
    entries = graph.get("global_entries", graph.get("global_nodes", []))
    return {int(node["global_id"]): node for node in entries}


def _planning_node_lookup(graph: dict) -> dict[int, dict]:
    nodes = graph.get("planning_nodes", [])
    if not nodes:
        # Compatibility projection for v1 artifacts. New artifacts always carry
        # explicit level-augmented planning nodes.
        projected = []
        for node in graph.get("global_nodes", []):
            level_hist = node.get("level_hist", {"0": 1})
            level = int(max(level_hist.items(), key=lambda item: int(item[1]))[0])
            projected.append({**node, "planning_id": int(node["global_id"]), "level": level})
        nodes = projected
    return {int(node["planning_id"]): node for node in nodes}


def _node_certificate(node: dict) -> dict[str, Any]:
    cert = node.get("prototype_certificate")
    return cert if isinstance(cert, dict) else {}


def _states_that_can_reach(outgoing: dict[int, list[dict]], targets: set[int]) -> set[int]:
    reverse: dict[int, set[int]] = defaultdict(set)
    for guide, edges in outgoing.items():
        for edge in edges:
            reverse[int(edge["target_planning_id"])].add(int(guide))
    reachable = set(int(value) for value in targets)
    frontier = list(reachable)
    while frontier:
        current = frontier.pop()
        for parent in reverse.get(current, set()):
            if parent not in reachable:
                reachable.add(parent)
                frontier.append(parent)
    return reachable


def _trace_row(
    edge: dict,
    planning_nodes: dict[int, dict],
    target_planning_id: int,
    guide_planning_id: int,
) -> dict[str, Any]:
    target_node = planning_nodes.get(int(target_planning_id), {})
    cert = _node_certificate(target_node)
    return {
        "global_edge_id": int(edge["global_edge_id"]),
        "guide_planning_id": int(guide_planning_id),
        "target_planning_id": int(target_planning_id),
        "guide_global_id": int(edge.get("guide_global_id", -1)),
        "target_global_id": int(target_node.get("global_id", edge.get("target_global_id", -1))),
        "guide_level": int(edge["guide_level"]),
        "target_level": int(target_node.get("level", edge["target_level"])),
        "count": int(edge.get("count", 0)),
        "target_certificate_hash": str(cert.get("certificate_hash", "")),
        "target_semantic_hash": str(cert.get("semantic_hash", "")),
        "target_quotient_graph_hash": str(cert.get("quotient_graph_hash", "")),
        "target_semantic_label": str(target_node.get("semantic_label", "")),
        "target_certificate": cert,
    }


def _realized_object(state: CheckpointState, planning_node: dict[str, Any]) -> dict[str, Any]:
    """Serialize the sampled intermediate as the inspectable object O(H)."""

    requested = _node_certificate(planning_node)
    semantic = requested.get("semantic", {}) if isinstance(requested, dict) else {}
    return certified_object_from_state(
        state,
        level=int(planning_node["level"]),
        mode=str(semantic.get("mode", "unknown")),
        radius=None if semantic.get("radius") is None else int(semantic["radius"]),
    ).to_dict()


def _prototype_num_nodes(node: dict) -> int:
    return len(node["prototype_state"]["node_summaries"])


def _edge_level(edge: dict) -> int:
    return int(edge.get("target_level", edge.get("relation", {}).get("target_level", 0)))


def _parse_state_spec(value: str | int | tuple[int, int]) -> tuple[int, int | None]:
    if isinstance(value, tuple):
        return int(value[0]), int(value[1])
    if isinstance(value, int):
        return int(value), None
    text = str(value).strip()
    level: int | None = None
    for separator in ("@L", "@l", ":"):
        if separator in text:
            code_text, level_text = text.split(separator, 1)
            text = code_text
            level = int(level_text)
            break
    return int(_parse_code(text)), level


def _matching_planning_ids(
    planning_nodes: dict[int, dict],
    code: int,
    level: int | None,
) -> set[int]:
    return {
        planning_id
        for planning_id, node in planning_nodes.items()
        if int(node["global_id"]) == int(code)
        and (level is None or int(node["level"]) == int(level))
    }


@torch.no_grad()
def sample_global_pipeline(
    checkpoint: str | Path,
    global_graph_path: str | Path,
    out_dir: str | Path,
    num_samples: int = 16,
    start_code: str | int | None = None,
    start_level: int | None = None,
    waypoint_code: str | int | None = None,
    waypoint_level: int | None = None,
    path: Sequence[str | int | tuple[int, int]] | None = None,
    device: str | None = None,
    guidance: float = 1.3,
    transition_temperature: float = 1.0,
    denoise_temperature: float = 0.9,
    none_bias: float = 2.0,
    max_degree: int = 4,
    max_path_steps: int = 8,
    max_refinement_retries: int = 2,
    variant: str = "ccd",
    control_metadata: dict[str, Any] | None = None,
    seed: int = 0,
) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, spec, ckpt = load_global_model_checkpoint(checkpoint, dev)
    graph = load_global_graph(global_graph_path)
    if graph.get("planning_edges") is not None:
        # Sampling is defined on the canonical level-augmented DAG. Validate
        # its semantic edge contract before binding it to a model artifact.
        validate_level_augmented_dag(graph)
    if str(ckpt["global_graph_fingerprint"]) != str(graph["artifact_fingerprint"]):
        raise ArtifactVersionError(
            "Denoiser and global graph come from different artifact chains; "
            "retrain or supply their matching v2 artifacts."
        )
    if str(ckpt["assignments_fingerprint"]) != str(graph["source_assignments_fingerprint"]):
        raise ArtifactVersionError(
            "Denoiser and global graph reference different checkpoint assignments; "
            "regenerate the pipeline together."
        )
    graph_spec = graph["spec"]
    if (
        tuple(graph_spec["atom_vocab"]) != spec.atom_vocab
        or tuple(graph_spec["bond_vocab"]) != spec.bond_vocab
        or int(graph_spec["max_nodes"]) != spec.max_nodes
    ):
        raise ArtifactVersionError(
            "Denoiser and global graph use different tensor specifications; regenerate them together."
        )
    expected_conditions = int(graph.get("num_global_conditions", 0))
    if expected_conditions and int(ckpt.get("num_global_codes", expected_conditions)) != expected_conditions:
        raise ValueError("Denoiser/global-graph condition-table size mismatch; regenerate the v2 artifacts together.")
    expected_edges = int(graph.get("num_condition_edges", 0))
    if expected_edges and int(ckpt.get("max_global_edges", expected_edges)) != expected_edges:
        raise ValueError("Denoiser/global-graph edge-table size mismatch; regenerate the v2 artifacts together.")
    if variant not in {"ccd", "exact_plan_final_diffusion"}:
        raise ValueError(f"Unknown sampling variant: {variant}")
    entries = _node_lookup(graph)
    condition_ids = {
        int(global_id): int(entry.get("condition_id", global_id))
        for global_id, entry in entries.items()
    }
    planning_nodes = _planning_node_lookup(graph)
    noise = DiscreteGraphNoise(
        timesteps=int(ckpt["config"].get("diffusion", {}).get("timesteps", 200)),
        node_marginal=ckpt["node_marginal"],
        edge_marginal=ckpt["edge_marginal"],
        node_noise_scale=float(ckpt["config"].get("diffusion", {}).get("node_noise_scale", 0.01)),
        categorical_node_indices=spec.categorical_node_indices,
        reserved_node_indices=spec.reserved_node_indices,
        activity_marginal=float(ckpt.get("activity_marginal", 0.5)),
        atom_feature_dim=spec.atom_feature_dim,
    )
    local_denoiser_steps = int(
        ckpt["config"].get("diffusion", {}).get("denoiser_evaluations", 32)
    )
    if graph.get("planning_edges") is not None:
        noise_edges = list(graph.get("initialization_edges", []))
        planning_edges = list(graph.get("planning_edges", []))
    else:
        noise_edges = []
        planning_edges = []
        for legacy in graph["global_edges"]:
            edge = dict(legacy)
            edge["guide_planning_id"] = int(edge["guide_global_id"])
            edge["target_planning_id"] = int(edge["target_global_id"])
            edge["guide_level"] = int(
                max(edge.get("guide_level_hist", {"1": 1}).items(), key=lambda item: int(item[1]))[0]
            )
            edge["target_level"] = _edge_level(edge)
            if int(edge["guide_global_id"]) < 0:
                noise_edges.append(edge)
            else:
                planning_edges.append(edge)
    if not noise_edges:
        raise ValueError("Global graph has no noise-to-checkpoint initialization edges.")
    outgoing: dict[int, list[dict]] = defaultdict(list)
    for edge in planning_edges:
        outgoing[int(edge["guide_planning_id"])].append(edge)

    start_id = _parse_code(start_code)
    waypoint_id = _parse_code(waypoint_code)
    if start_id is not None:
        start_candidates = _matching_planning_ids(planning_nodes, start_id, start_level)
        if not start_candidates:
            raise ValueError(f"No planning state matches start C{start_id} at level {start_level}.")
    else:
        start_candidates = set()
    if waypoint_id is not None:
        waypoint_targets = _matching_planning_ids(planning_nodes, waypoint_id, waypoint_level)
        if not waypoint_targets:
            raise ValueError(f"No planning state matches waypoint C{waypoint_id} at level {waypoint_level}.")
        reachable_to_waypoint = _states_that_can_reach(outgoing, waypoint_targets)
    else:
        waypoint_targets = set()
        reachable_to_waypoint = set()

    fixed_specs = [_parse_state_spec(value) for value in path] if path is not None else []
    if fixed_specs:
        first_code, first_level = fixed_specs[0]
        fixed_start_candidates = _matching_planning_ids(planning_nodes, first_code, first_level)
        if not fixed_start_candidates:
            raise ValueError(f"Fixed path starts at an unknown state C{first_code}@L{first_level}.")
        if start_id is not None and not (fixed_start_candidates & start_candidates):
            raise ValueError("The fixed path conflicts with the requested start state.")
        start_candidates = fixed_start_candidates
        if max(0, len(fixed_specs) - 1) > max_path_steps:
            raise ValueError("Fixed path exceeds the maximum number of checkpoint refinement edges.")

    initialization_by_target: dict[int, list[dict]] = defaultdict(list)
    for edge in noise_edges:
        initialization_by_target[int(edge["target_planning_id"])].append(edge)
    initializable = set(initialization_by_target)
    if start_candidates:
        start_candidates &= initializable
        if not start_candidates:
            raise ValueError("Requested start state has no trained noise-to-checkpoint edge.")

    if graph.get("coarsest_planning_prior"):
        prior_items = [
            {"planning_id": int(key), "count": int(value)}
            for key, value in graph["coarsest_planning_prior"].items()
        ]
    else:
        prior_items = [
            {"planning_id": int(key), "count": int(value)}
            for key, value in graph.get("coarsest_prior", {}).items()
        ]
    if start_candidates:
        prior_items = [item for item in prior_items if int(item["planning_id"]) in start_candidates]
    if waypoint_id is not None:
        prior_items = [item for item in prior_items if int(item["planning_id"]) in reachable_to_waypoint]
    if not prior_items:
        raise ValueError("No initial planning state satisfies the requested controls.")

    def realize(edge: dict, node: dict, guide_state: CheckpointState | None) -> tuple[CheckpointState, int, bool]:
        if variant == "exact_plan_final_diffusion" and int(node["level"]) > 0:
            return CheckpointState.from_dict(node["prototype_state"]), 0, False
        attempts = 0
        candidate: CheckpointState | None = None
        for attempts in range(1, max(1, max_refinement_retries + 1) + 1):
            candidate = sample_global_transition(
                model=model,
                spec=spec,
                noise=noise,
                relation_edge=edge["relation"],
                global_edge_id=int(edge["global_edge_id"]),
                target_condition_id=condition_ids[int(node["global_id"])],
                guide_condition_id=(
                    -1
                    if int(edge.get("guide_global_id", -1)) < 0
                    else condition_ids[int(edge["guide_global_id"])]
                ),
                num_nodes=_prototype_num_nodes(node),
                guide_state=guide_state,
                device=dev,
                target_template_state=CheckpointState.from_dict(node["prototype_state"]),
                # Noise-to-checkpoint transitions have no graph guide, but
                # they still carry target/global-edge conditions.  Apply CFG
                # to those identifiers as well as to later guided segments.
                guidance_weight=guidance,
                steps=local_denoiser_steps,
                temperature=denoise_temperature,
                none_bias=none_bias,
                max_degree=max_degree,
            )
            if structural_validity(candidate):
                return candidate, attempts, False
        assert candidate is not None
        return candidate, attempts, True

    final_samples: list[CheckpointState] = []
    payload: list[dict[str, Any]] = []
    for _ in range(num_samples):
        chosen = _weighted_choice(prior_items, temperature=transition_temperature)
        current_planning_id = int(chosen["planning_id"])
        start_edge = _weighted_choice(
            initialization_by_target[current_planning_id], temperature=transition_temperature
        )
        current_node = planning_nodes[current_planning_id]
        current_state, attempts, repair_failed = realize(start_edge, current_node, None)
        trace = [
            {
                **_trace_row(start_edge, planning_nodes, current_planning_id, -1),
                "realization_attempts": attempts,
                "repair_failed": repair_failed,
                "realized_checkpoint_object": _realized_object(current_state, current_node),
                "realization_status": _realization_status(current_state),
            }
        ]
        reached_level0 = int(current_node["level"]) == 0
        waypoint_reached = waypoint_id is None or current_planning_id in waypoint_targets
        fixed_position = 1
        # ``max_path_steps`` counts coarse-to-fine checkpoint refinements. The
        # noise-to-start initialization is recorded but is not a planning edge.
        planning_steps = 0
        while planning_steps < max_path_steps:
            if reached_level0:
                break
            candidate_edges = list(outgoing.get(current_planning_id, []))
            if fixed_specs:
                if fixed_position >= len(fixed_specs):
                    break
                next_code, next_level = fixed_specs[fixed_position]
                allowed = _matching_planning_ids(planning_nodes, next_code, next_level)
                candidate_edges = [
                    edge for edge in candidate_edges if int(edge["target_planning_id"]) in allowed
                ]
                if not candidate_edges:
                    raise ValueError(
                        f"Fixed path has no observed edge at position {fixed_position}: "
                        f"C{next_code}@L{next_level}."
                    )
                fixed_position += 1
            elif waypoint_id is not None and not waypoint_reached:
                candidate_edges = [
                    edge
                    for edge in candidate_edges
                    if int(edge["target_planning_id"]) in reachable_to_waypoint
                ]
                if not candidate_edges:
                    raise RuntimeError("Waypoint constraint became unreachable; refusing to relax it.")
            if not candidate_edges:
                break
            edge = _weighted_choice(candidate_edges, temperature=transition_temperature)
            target_planning_id = int(edge["target_planning_id"])
            target_node = planning_nodes[target_planning_id]
            previous_planning_id = current_planning_id
            guide_before_transition = current_state
            current_state, attempts, repair_failed = realize(edge, target_node, current_state)
            current_planning_id = target_planning_id
            reached_level0 = int(target_node["level"]) == 0
            waypoint_reached = waypoint_reached or current_planning_id in waypoint_targets
            trace.append(
                {
                    **_trace_row(edge, planning_nodes, current_planning_id, previous_planning_id),
                    "realization_attempts": attempts,
                    "repair_failed": repair_failed,
                    "realized_checkpoint_object": _realized_object(current_state, target_node),
                    "realization_status": _realization_status(current_state, guide_before_transition),
                }
            )
            planning_steps += 1
        if fixed_specs and fixed_position != len(fixed_specs):
            raise ValueError("Generation stopped before completing the fixed checkpoint path.")
        if fixed_specs and not reached_level0:
            raise ValueError("Complete-path control must terminate at a level-0 checkpoint.")
        if waypoint_id is not None and not waypoint_reached:
            raise RuntimeError("Generation terminated without visiting the requested waypoint.")
        final_samples.append(current_state)
        final_planning_id = int(trace[-1]["target_planning_id"])
        final_code = int(trace[-1]["target_global_id"])
        target_cert = _node_certificate(planning_nodes.get(final_planning_id, {}))
        payload.append(
            {
                "sample": current_state.to_dict(),
                "global_path": [int(row["target_global_id"]) for row in trace],
                "planning_path": [int(row["target_planning_id"]) for row in trace],
                "path_levels": [int(row["target_level"]) for row in trace],
                "path_edge_counts": [int(row["count"]) for row in trace],
                "path_certificate_hashes": [str(row["target_certificate_hash"]) for row in trace],
                "requested_path_objects": [dict(row["target_certificate"]) for row in trace],
                "start_global_id": int(trace[0]["target_global_id"]),
                "start_planning_id": int(trace[0]["target_planning_id"]),
                "requested_start_global_id": None if start_id is None else int(start_id),
                "requested_start_level": None if start_level is None else int(start_level),
                "requested_waypoint_global_id": None if waypoint_id is None else int(waypoint_id),
                "requested_waypoint_level": None if waypoint_level is None else int(waypoint_level),
                "requested_path": None if not fixed_specs else [[code, level] for code, level in fixed_specs],
                "target_global_id": final_code,
                "target_planning_id": final_planning_id,
                "target_certificate_hash": str(target_cert.get("certificate_hash", "")),
                "target_semantic_hash": str(target_cert.get("semantic_hash", "")),
                "target_quotient_graph_hash": str(target_cert.get("quotient_graph_hash", "")),
                "target_certificate": target_cert,
                "variant": variant,
                "repair_budget": int(max_refinement_retries),
                "control_metadata": dict(control_metadata or {}),
                "assignment_distances": [
                    float(planning_nodes[int(row["target_planning_id"])].get("mean_assignment_distance", 0.0))
                    for row in trace
                ],
                "checkpoint_consistency": {
                    "followed_existing_edges": True,
                    "reached_level0": bool(reached_level0),
                    "waypoint_reached": bool(waypoint_reached),
                    "complete_path_followed": bool(not fixed_specs or fixed_position == len(fixed_specs)),
                    "structural_repair_failed": bool(any(row["repair_failed"] for row in trace)),
                    "repair_failed": bool(any(row["repair_failed"] for row in trace)),
                },
                "path_trace": trace,
            }
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "samples.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = summarize_samples(final_samples)
    path_signatures = Counter(tuple(item["planning_path"]) for item in payload)
    summary["path_diversity"] = {
        "num_unique_paths": len(path_signatures),
        "unique_path_fraction": len(path_signatures) / max(1, len(payload)),
        "path_length_hist": dict(Counter(len(item["global_path"]) for item in payload)),
        "novel_path_fraction": None,
    }
    summary["checkpoint_consistency"] = {
        "followed_existing_edges": 1.0,
        "reached_level0": sum(bool(item["checkpoint_consistency"]["reached_level0"]) for item in payload) / max(1, len(payload)),
        "waypoint_reached": sum(bool(item["checkpoint_consistency"]["waypoint_reached"]) for item in payload) / max(1, len(payload)),
    }
    summary["start_global_id_hist"] = dict(Counter(int(item["start_global_id"]) for item in payload))
    summary["start_planning_id_hist"] = dict(Counter(int(item["start_planning_id"]) for item in payload))
    summary["repair_failure"] = sum(
        bool(item["checkpoint_consistency"]["repair_failed"]) for item in payload
    ) / max(1, len(payload))
    if waypoint_id is not None:
        summary["requested_waypoint_global_id"] = int(waypoint_id)
    if control_metadata:
        summary["control_metadata"] = dict(control_metadata)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
