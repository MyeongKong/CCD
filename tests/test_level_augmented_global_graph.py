from __future__ import annotations

from copy import deepcopy
import json

import pytest
import torch

from checkpoint_dag_diffusion.artifacts import ASSIGNMENTS_FORMAT, artifact_fingerprint
from checkpoint_dag_diffusion.checkpoint_state import CheckpointState
from checkpoint_dag_diffusion.global_graph import (
    build_global_graph,
    code_count,
    load_global_graph,
    validate_level_augmented_dag,
)
from checkpoint_dag_diffusion.preprocess import (
    build_records_from_graphs,
    load_graph_json,
    write_corpus,
)
from checkpoint_dag_diffusion.tensorize import TensorSpec
import checkpoint_dag_diffusion.global_sampling as global_sampling


def _colliding_assignments(tmp_path):
    graphs = load_graph_json("data/toy_graphs.json", max_molecules=3)
    records = build_records_from_graphs(graphs, max_nodes=16)
    corpus = tmp_path / "corpus"
    write_corpus(records, corpus, max_nodes=16)

    # Force every local checkpoint, including checkpoints from different
    # levels, into one code.  This creates self-loops in the legacy code-only
    # projection and is exactly the case level augmentation must resolve.
    assignments = []
    for record_index, record in enumerate(records):
        for checkpoint_index, checkpoint in enumerate(record["checkpoints"]):
            certificate_hash = str(checkpoint["certified_object"]["certificate_hash"])
            assignments.append(
                {
                    "record_index": record_index,
                    "checkpoint_index": checkpoint_index,
                    "level": int(checkpoint["level"]),
                    "hash": certificate_hash,
                    "global_id": 7,
                    "assignment_distance": float(100 - 10 * record_index - checkpoint_index),
                }
            )
    assignment_payload = {
        "format": ASSIGNMENTS_FORMAT,
        "method": "forced_collision",
        "split": "train",
        "codebook_size": 8,
        "assignments": assignments,
        "summary": {"num_assignments": len(assignments)},
        "source_codebook_fingerprint": "forced-collision-codebook",
    }
    assignment_payload["artifact_fingerprint"] = artifact_fingerprint(assignment_payload)
    assignment_path = tmp_path / "assignments.pt"
    torch.save(assignment_payload, assignment_path)
    result = build_global_graph(corpus, assignment_path, tmp_path / "global")
    return records, assignments, load_global_graph(result["global_graph"])


def test_level_augmented_states_resolve_cross_level_code_collisions(tmp_path):
    records, assignments, graph = _colliding_assignments(tmp_path)

    assert graph["format"].endswith(".v2")
    assert graph["planning_is_dag"] is True
    assert validate_level_augmented_dag(graph)

    # C_7 remains one reusable code-level entry, while its appearances at
    # different levels become distinct planning nodes (C_7, level).
    assert len(graph["global_entries"]) == 1
    assert int(graph["global_entries"][0]["global_id"]) == 7
    assert int(graph["global_entries"][0]["condition_id"]) == 0
    assert code_count(graph) == 1
    assert graph["global_entries"] == graph["global_nodes"]
    observed_levels = {int(item["level"]) for item in assignments}
    assert len(graph["planning_nodes"]) == len(observed_levels)
    assert {
        (int(node["global_id"]), int(node["level"])) for node in graph["planning_nodes"]
    } == {(7, level) for level in observed_levels}

    # Code-only edges can self-loop after clustering; canonical planning edges
    # never do and strictly decrease the contraction level.
    assert any(
        int(edge["guide_global_id"]) == int(edge["target_global_id"]) == 7
        for edge in graph["global_edges"]
        if int(edge["guide_global_id"]) >= 0
    )
    assert graph["planning_edges"]
    assert all(
        int(edge["guide_planning_id"]) != int(edge["target_planning_id"])
        and int(edge["guide_level"]) > int(edge["target_level"])
        for edge in graph["planning_edges"]
    )

    nonnoise_observations = sum(
        edge.get("guide") is not None
        for record in records
        for edge in record["dag_edges"]
    )
    noise_observations = sum(
        edge.get("guide") is None
        for record in records
        for edge in record["dag_edges"]
    )
    assert sum(int(edge["count"]) for edge in graph["planning_edges"]) == nonnoise_observations
    assert sum(int(edge["count"]) for edge in graph["initialization_edges"]) == noise_observations
    assert sum(int(v) for v in graph["coarsest_planning_prior"].values()) == noise_observations
    assert sum(int(row["count"]) for row in graph["coarsest_state_prior"]) == noise_observations


def test_representatives_are_nearest_and_validation_rejects_non_decreasing_edge(tmp_path):
    _, assignments, graph = _colliding_assignments(tmp_path)

    expected_code = min(
        assignments,
        key=lambda item: (
            float(item["assignment_distance"]),
            int(item["record_index"]),
            int(item["checkpoint_index"]),
        ),
    )
    entry = graph["global_entries"][0]
    assert int(entry["prototype_record_index"]) == int(expected_code["record_index"])
    assert int(entry["prototype_checkpoint_index"]) == int(expected_code["checkpoint_index"])
    assert float(entry["prototype_assignment_distance"]) == float(expected_code["assignment_distance"])

    for node in graph["planning_nodes"]:
        level = int(node["level"])
        expected = min(
            (item for item in assignments if int(item["level"]) == level),
            key=lambda item: (
                float(item["assignment_distance"]),
                int(item["record_index"]),
                int(item["checkpoint_index"]),
            ),
        )
        assert int(node["prototype_record_index"]) == int(expected["record_index"])
        assert int(node["prototype_checkpoint_index"]) == int(expected["checkpoint_index"])
        assert float(node["prototype_assignment_distance"]) == float(expected["assignment_distance"])

    invalid = deepcopy(graph)
    invalid["planning_edges"][0]["guide_level"] = invalid["planning_edges"][0]["target_level"]
    with pytest.raises(ValueError):
        validate_level_augmented_dag(invalid)


def test_complete_path_control_uses_exact_level_states_and_observed_edges(tmp_path, monkeypatch):
    _, _, graph = _colliding_assignments(tmp_path)
    graph_path = tmp_path / "global_graph.pt"
    torch.save(graph, graph_path)

    planning_nodes = {int(node["planning_id"]): node for node in graph["planning_nodes"]}
    initializable = {int(edge["target_planning_id"]) for edge in graph["initialization_edges"]}
    first_edge = next(
        edge
        for edge in graph["planning_edges"]
        if int(edge["guide_planning_id"]) in initializable and int(edge["target_level"]) == 0
    )
    start_pid = int(first_edge["guide_planning_id"])
    target_pid = int(first_edge["target_planning_id"])
    start_node = planning_nodes[start_pid]
    target_node = planning_nodes[target_pid]
    fixed_path = [
        (int(start_node["global_id"]), int(start_node["level"])),
        (int(target_node["global_id"]), int(target_node["level"])),
    ]

    spec = TensorSpec(
        atom_vocab=tuple(graph["spec"]["atom_vocab"]),
        bond_vocab=tuple(graph["spec"]["bond_vocab"]),
        max_nodes=int(graph["spec"]["max_nodes"]),
    )
    fake_checkpoint = {
        "config": {"diffusion": {"timesteps": 1, "denoiser_evaluations": 1}},
        "node_marginal": torch.zeros(spec.node_feature_dim),
        "edge_marginal": torch.ones(spec.edge_dim),
        "activity_marginal": 0.5,
        "global_graph_fingerprint": graph["artifact_fingerprint"],
        "assignments_fingerprint": graph["source_assignments_fingerprint"],
    }
    monkeypatch.setattr(
        global_sampling,
        "load_global_model_checkpoint",
        lambda path, device: (object(), spec, fake_checkpoint),
    )
    final_state = CheckpointState.from_dict(target_node["prototype_state"])
    monkeypatch.setattr(
        global_sampling,
        "sample_global_transition",
        lambda **kwargs: final_state,
    )

    out_dir = tmp_path / "samples"
    global_sampling.sample_global_pipeline(
        checkpoint=tmp_path / "unused.pt",
        global_graph_path=graph_path,
        out_dir=out_dir,
        num_samples=1,
        start_code=fixed_path[0][0],
        start_level=fixed_path[0][1],
        waypoint_code=fixed_path[-1][0],
        waypoint_level=fixed_path[-1][1],
        path=fixed_path,
        variant="exact_plan_final_diffusion",
        device="cpu",
        max_path_steps=2,
    )
    sample = json.loads((out_dir / "samples.json").read_text(encoding="utf-8"))[0]
    assert sample["planning_path"] == [start_pid, target_pid]
    assert sample["path_levels"] == [int(start_node["level"]), 0]
    assert sample["checkpoint_consistency"]["complete_path_followed"] is True
    assert sample["checkpoint_consistency"]["waypoint_reached"] is True

    # A code-level repetition at the wrong level is not silently projected to
    # the observed coarse-to-level-0 edge, even when both states share one C_k.
    with pytest.raises(ValueError, match="no observed edge"):
        global_sampling.sample_global_pipeline(
            checkpoint=tmp_path / "unused.pt",
            global_graph_path=graph_path,
            out_dir=tmp_path / "invalid_samples",
            num_samples=1,
            path=[fixed_path[0], fixed_path[0]],
            variant="exact_plan_final_diffusion",
            device="cpu",
            max_path_steps=2,
        )

    # Consumer-side validation rejects a cached graph whose relation no longer
    # agrees with its strict level-decreasing edge metadata.
    malformed = deepcopy(graph)
    malformed["planning_edges"][0]["relation"]["target_level"] += 1
    # Re-sign the deliberately malformed fixture so the semantic DAG check,
    # rather than the preceding integrity check, is what rejects it.
    malformed["artifact_fingerprint"] = artifact_fingerprint(malformed)
    malformed_path = tmp_path / "malformed_graph.pt"
    torch.save(malformed, malformed_path)
    with pytest.raises(ValueError, match="relation field"):
        global_sampling.sample_global_pipeline(
            checkpoint=tmp_path / "unused.pt",
            global_graph_path=malformed_path,
            out_dir=tmp_path / "malformed_samples",
            num_samples=1,
            device="cpu",
        )
