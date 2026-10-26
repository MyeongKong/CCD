from __future__ import annotations

import json

import pytest
import torch

from checkpoint_dag_diffusion.artifacts import (
    ASSIGNMENTS_FORMAT,
    ArtifactVersionError,
    GLOBAL_GRAPH_FORMAT,
    artifact_fingerprint,
    load_assignments,
    load_corpus_records,
    validate_assignments_against_records,
)
from checkpoint_dag_diffusion.dataset import CheckpointDAGEdgeDataset
from checkpoint_dag_diffusion.global_codebook import load_codebook_model
from checkpoint_dag_diffusion.global_graph import load_global_graph
from checkpoint_dag_diffusion.global_sampling import load_global_model_checkpoint
from checkpoint_dag_diffusion.preprocess import (
    build_records_from_graphs,
    load_graph_json,
    write_corpus,
)
from checkpoint_dag_diffusion.sampling import load_model_checkpoint


def _toy_corpus(tmp_path):
    records = build_records_from_graphs(
        load_graph_json("data/toy_graphs.json", max_molecules=2),
        max_nodes=16,
    )
    corpus = tmp_path / "corpus"
    write_corpus(records, corpus, max_nodes=16)
    return corpus


@pytest.mark.parametrize(
    ("loader", "stale_format", "filename"),
    [
        (load_model_checkpoint, "checkpoint_dag_diffusion.local_denoiser.v1", "local.pt"),
        (load_global_model_checkpoint, "checkpoint_dag_diffusion.global_denoiser.v1", "global.pt"),
        (load_codebook_model, "checkpoint_dag_diffusion.global_codebook.v1", "codebook.pt"),
    ],
)
def test_model_loaders_reject_v1_before_reading_stale_fields(tmp_path, loader, stale_format, filename):
    path = tmp_path / filename
    torch.save({"format": stale_format, "model_state": {}}, path)
    with pytest.raises(ArtifactVersionError, match="v1/stale artifacts"):
        loader(path, torch.device("cpu"))


def test_local_loader_rejects_stale_v2_without_mixed_state_contract(tmp_path):
    path = tmp_path / "stale_local_v2.pt"
    torch.save(
        {
            "format": "checkpoint_dag_diffusion.local_denoiser.v2",
            "model_state": {},
            "config": {},
            "spec": {},
            "node_marginal": torch.zeros(1),
            "edge_marginal": torch.ones(1),
            # The old payload lacks activity_marginal and architecture flags.
            "max_level": 1,
            "max_delta": 1,
            "architecture": {},
        },
        path,
    )
    with pytest.raises(ArtifactVersionError, match="activity_marginal"):
        load_model_checkpoint(path, torch.device("cpu"))


def test_global_loader_rejects_stale_v2_without_compact_condition_ids(tmp_path):
    path = tmp_path / "stale_global_v2.pt"
    payload = {
            "format": "checkpoint_dag_diffusion.global_denoiser.v2",
            "model_state": {},
            "config": {"model": {}, "diffusion": {}},
            "spec": {"atom_vocab": ["C"], "bond_vocab": ["none"], "max_nodes": 1},
            "node_marginal": torch.zeros(10),
            "edge_marginal": torch.ones(1),
            "activity_marginal": 0.5,
            "max_level": 1,
            "max_delta": 1,
            "num_global_codes": 1,
            "max_global_edges": 1,
            "global_graph_fingerprint": "graph-fingerprint",
            "assignments_fingerprint": "assignment-fingerprint",
            "architecture": {
                "certified_object_format": "checkpoint_dag_diffusion.certified_object.v2",
                "relation_order": ["guide_level", "target_level", "delta", "operator_id"],
                "mixed_state_corruption": True,
                "categorical_activity": True,
                "categorical_reverse_posterior": True,
                "categorical_node_indices": [5, 6, 7, 8],
                "level0_atom_categorical": True,
                "atom_feature_dim": 1,
                "binary_object_logits": True,
                "activity_capacity_mask": True,
                "factorized_edge_activity_posterior": True,
                "cfg_all_heads": True,
                "semantic_slot_alignment": True,
                "full_correspondence_supervision": True,
                "nested_generated_lineage": True,
                "boundary_slot_aliases": True,
                # compact_condition_ids is deliberately absent.
            },
        }
    payload["artifact_fingerprint"] = artifact_fingerprint(payload)
    torch.save(payload, path)
    with pytest.raises(ArtifactVersionError, match="compact v2 global condition ids"):
        load_global_model_checkpoint(path, torch.device("cpu"))


def test_denoiser_loader_rejects_pre_categorical_atom_v2_checkpoint(tmp_path):
    path = tmp_path / "pre_atom_categorical.pt"
    payload = {
            "format": "checkpoint_dag_diffusion.local_denoiser.v2",
            "model_state": {},
            "config": {"model": {}, "diffusion": {}},
            "spec": {"atom_vocab": ["C"], "bond_vocab": ["none"], "max_nodes": 1},
            "node_marginal": torch.zeros(10),
            "edge_marginal": torch.ones(1),
            "activity_marginal": 0.5,
            "max_level": 1,
            "max_delta": 1,
            "architecture": {
                "certified_object_format": "checkpoint_dag_diffusion.certified_object.v2",
                "relation_order": ["guide_level", "target_level", "delta", "operator_id"],
                "mixed_state_corruption": True,
                "categorical_activity": True,
                "categorical_reverse_posterior": True,
                "categorical_node_indices": [5, 6, 7, 8],
            },
        }
    payload["artifact_fingerprint"] = artifact_fingerprint(payload)
    torch.save(payload, path)
    with pytest.raises(ArtifactVersionError, match="level0_atom_categorical"):
        load_model_checkpoint(path, torch.device("cpu"))


def test_corpus_consumer_rejects_v1_manifest(tmp_path):
    corpus = _toy_corpus(tmp_path)
    manifest_path = corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format"] = "checkpoint_dag_diffusion.corpus.v1"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactVersionError, match="v1/stale artifacts"):
        CheckpointDAGEdgeDataset(corpus, split="train")


def test_corpus_consumer_rejects_stale_v2_records_without_certified_objects(tmp_path):
    corpus = _toy_corpus(tmp_path)
    records_path = corpus / "train" / "records.pt"
    records = torch.load(records_path, map_location="cpu")
    records[0]["checkpoints"][0].pop("certified_object")
    torch.save(records, records_path)

    with pytest.raises(ArtifactVersionError, match="certified_object"):
        CheckpointDAGEdgeDataset(corpus, split="train")


def test_assignment_and_global_graph_loaders_reject_unversioned_or_v1_payloads(tmp_path):
    assignments_path = tmp_path / "assignments.pt"
    torch.save(
        {
            "method": "legacy",
            "split": "train",
            "codebook_size": 1,
            "assignments": [],
            "summary": {},
        },
        assignments_path,
    )
    with pytest.raises(ArtifactVersionError, match="<missing>"):
        load_assignments(assignments_path)

    graph_path = tmp_path / "global_graph.pt"
    torch.save({"format": "checkpoint_dag_diffusion.global_graph.v1"}, graph_path)
    with pytest.raises(ArtifactVersionError, match="v1/stale artifacts"):
        load_global_graph(graph_path)


def test_assignment_table_is_bound_to_exact_corpus_objects(tmp_path):
    corpus = _toy_corpus(tmp_path)
    records = load_corpus_records(corpus, "train")
    rows = []
    for record_index, record in enumerate(records):
        for checkpoint_index, checkpoint in enumerate(record["checkpoints"]):
            rows.append(
                {
                    "record_index": record_index,
                    "checkpoint_index": checkpoint_index,
                    "level": int(checkpoint["level"]),
                    "hash": str(checkpoint["certified_object"]["certificate_hash"]),
                }
            )
    validate_assignments_against_records({"assignments": rows}, records)

    wrong = [dict(row) for row in rows]
    wrong[0]["hash"] = "different-corpus-certificate"
    with pytest.raises(ArtifactVersionError, match="does not match corpus checkpoint"):
        validate_assignments_against_records({"assignments": wrong}, records)

    with pytest.raises(ArtifactVersionError, match="does not completely cover"):
        validate_assignments_against_records({"assignments": rows[:-1]}, records)


def test_stale_v2_assignments_and_graph_require_current_schema_fields():
    with pytest.raises(ArtifactVersionError, match="assignment_distance"):
        from checkpoint_dag_diffusion.artifacts import validate_assignments

        payload = {
                "format": ASSIGNMENTS_FORMAT,
                "method": "legacy-v2",
                "split": "train",
                "codebook_size": 1,
                "assignments": [
                    {
                        "record_index": 0,
                        "checkpoint_index": 0,
                        "level": 0,
                        "hash": "h",
                        "global_id": 0,
                    }
                ],
                "summary": {},
                "source_codebook_fingerprint": "fixture-codebook",
            }
        payload["artifact_fingerprint"] = artifact_fingerprint(payload)
        validate_assignments(payload)

    from checkpoint_dag_diffusion.artifacts import validate_global_graph_payload

    with pytest.raises(ArtifactVersionError, match="num_global_conditions"):
        validate_global_graph_payload(
            {
                "format": GLOBAL_GRAPH_FORMAT,
                "method": "stale-v2",
                "split": "train",
                "global_entries": [],
                "planning_nodes": [],
                "planning_edges": [],
                "initialization_edges": [],
                "coarsest_planning_prior": {},
                "num_condition_edges": 0,
                "planning_is_dag": True,
                "spec": {"atom_vocab": ["C"], "bond_vocab": ["none"], "max_nodes": 1},
                "num_global_nodes": 0,
                "num_planning_nodes": 0,
                "num_planning_edges": 0,
                "num_initialization_edges": 0,
            }
        )
