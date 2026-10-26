from __future__ import annotations

"""Strict readers for the mutually dependent CCD v2 artifacts.

The v2 implementation changed the meaning of several tensors and identifiers
(certified-object inputs, mixed graph states, level-augmented planning states,
and compact global condition ids).  Loading an older payload with permissive
defaults can therefore produce a run that is syntactically valid but
scientifically different.  These helpers reject such payloads at the first
consumer boundary and tell the caller to regenerate the artifact chain.
"""

import json
import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


CORPUS_FORMAT = "checkpoint_dag_diffusion.corpus.v2"
CERTIFIED_OBJECT_FORMAT = "checkpoint_dag_diffusion.certified_object.v2"
CODEBOOK_FORMAT = "checkpoint_dag_diffusion.global_codebook.v2"
ASSIGNMENTS_FORMAT = "checkpoint_dag_diffusion.global_assignments.v2"
GLOBAL_GRAPH_FORMAT = "checkpoint_dag_diffusion.global_graph.v2"
LOCAL_DENOISER_FORMAT = "checkpoint_dag_diffusion.local_denoiser.v2"
GLOBAL_DENOISER_FORMAT = "checkpoint_dag_diffusion.global_denoiser.v2"
RELATION_ORDER = ["guide_level", "target_level", "delta", "operator_id"]


def artifact_fingerprint(payload: Mapping[str, Any]) -> str:
    """Deterministically bind an artifact's schema, tables, and tensors."""

    def normalize(value: Any) -> Any:
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            return {
                "__tensor__": True,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
            }
        if isinstance(value, Mapping):
            return {
                str(key): normalize(item)
                for key, item in sorted(value.items(), key=lambda row: str(row[0]))
                if str(key) != "artifact_fingerprint"
            }
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)

    encoded = json.dumps(normalize(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verify_fingerprint(payload: Mapping[str, Any], label: str) -> None:
    expected = str(payload.get("artifact_fingerprint", ""))
    if not expected:
        raise ArtifactVersionError(f"{label} lacks its pipeline-binding artifact fingerprint.")
    observed = artifact_fingerprint(payload)
    if observed != expected:
        raise ArtifactVersionError(f"{label} fingerprint mismatch; the artifact is stale or modified.")


class ArtifactVersionError(ValueError):
    """Raised when an artifact is stale or does not satisfy the v2 contract."""


def _mapping(payload: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ArtifactVersionError(f"{label} must be a mapping, got {type(payload).__name__}.")
    return payload


def _require_keys(payload: Mapping[str, Any], keys: Iterable[str], label: str) -> None:
    missing = sorted(str(key) for key in keys if key not in payload)
    if missing:
        raise ArtifactVersionError(
            f"{label} is missing required v2 field(s): {', '.join(missing)}. "
            "Regenerate this artifact and its downstream artifacts."
        )


def _require_format(payload: Mapping[str, Any], expected: str, label: str) -> None:
    actual = payload.get("format")
    if actual != expected:
        rendered = "<missing>" if actual is None else repr(actual)
        raise ArtifactVersionError(
            f"Unsupported {label} format {rendered}; expected {expected!r}. "
            "v1/stale artifacts are not compatible with the current model; regenerate them."
        )


def _require_relation_contract(architecture: Mapping[str, Any], label: str) -> None:
    if architecture.get("relation_order") != RELATION_ORDER:
        raise ArtifactVersionError(
            f"{label} does not use the required v2 relation order {RELATION_ORDER!r}."
        )


def validate_corpus_manifest(payload: Any, *, source: str = "corpus manifest") -> dict[str, Any]:
    manifest = dict(_mapping(payload, source))
    _require_format(manifest, CORPUS_FORMAT, source)
    _require_keys(
        manifest,
        (
            "certified_object_format",
            "relation_order",
            "correspondence_maps",
            "atom_vocab",
            "bond_vocab",
            "max_nodes",
            "splits",
        ),
        source,
    )
    if manifest["certified_object_format"] != CERTIFIED_OBJECT_FORMAT:
        raise ArtifactVersionError(f"{source} does not contain v2 certified checkpoint objects.")
    if manifest["relation_order"] != RELATION_ORDER:
        raise ArtifactVersionError(f"{source} has a stale relation-field order.")
    if manifest["correspondence_maps"] is not True:
        raise ArtifactVersionError(f"{source} must declare explicit correspondence maps.")
    if int(manifest["max_nodes"]) <= 0 or not manifest["atom_vocab"] or not manifest["bond_vocab"]:
        raise ArtifactVersionError(f"{source} has an invalid tensor specification.")
    return manifest


def load_corpus_manifest(corpus_dir: str | Path) -> dict[str, Any]:
    path = Path(corpus_dir) / "manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactVersionError(f"Cannot read corpus manifest {path}: {exc}") from exc
    return validate_corpus_manifest(payload, source=f"corpus manifest {path}")


def validate_corpus_records(records: Any, *, split: str) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        raise ArtifactVersionError(f"Corpus split {split!r} must contain a list of records.")
    for record_index, record in enumerate(records):
        label = f"corpus {split} record {record_index}"
        record = _mapping(record, label)
        _require_keys(record, ("checkpoints", "dag_edges"), label)
        if not isinstance(record["checkpoints"], list) or not isinstance(record["dag_edges"], list):
            raise ArtifactVersionError(f"{label} has malformed checkpoint/DAG collections.")
        for checkpoint_index, checkpoint in enumerate(record["checkpoints"]):
            checkpoint_label = f"{label}, checkpoint {checkpoint_index}"
            checkpoint = _mapping(checkpoint, checkpoint_label)
            _require_keys(
                checkpoint,
                ("state", "level", "mode", "hash", "certified_object"),
                checkpoint_label,
            )
            certified = _mapping(checkpoint["certified_object"], f"{checkpoint_label} certified object")
            _require_format(certified, CERTIFIED_OBJECT_FORMAT, f"{checkpoint_label} certified object")
            _require_keys(
                certified,
                (
                    "quotient_graph",
                    "partition",
                    "structural_summaries",
                    "summary",
                    "boundary_slots",
                    "semantic",
                    "quotient_graph_hash",
                    "semantic_hash",
                    "certificate_hash",
                ),
                f"{checkpoint_label} certified object",
            )
        for edge_index, edge in enumerate(record["dag_edges"]):
            edge_label = f"{label}, DAG edge {edge_index}"
            edge = _mapping(edge, edge_label)
            _require_keys(
                edge,
                ("target", "guide", "guide_level", "target_level", "delta", "operator_id"),
                edge_label,
            )
            if edge.get("guide") is not None:
                correspondence = edge.get("correspondence")
                if not isinstance(correspondence, Mapping):
                    raise ArtifactVersionError(f"{edge_label} lacks its explicit v2 correspondence map.")
                _require_keys(
                    correspondence,
                    (
                        "guide_to_target",
                        "target_to_guide",
                        "preserved_target_to_guide",
                        "guide_boundary_to_target",
                    ),
                    f"{edge_label} correspondence",
                )
    return records


def load_corpus_records(corpus_dir: str | Path, split: str = "train") -> list[dict[str, Any]]:
    corpus = Path(corpus_dir)
    manifest = load_corpus_manifest(corpus)
    if split not in manifest["splits"]:
        raise ArtifactVersionError(f"Corpus manifest does not declare split {split!r}.")
    path = corpus / split / "records.pt"
    try:
        records = torch.load(path, map_location="cpu")
    except (OSError, RuntimeError) as exc:
        raise ArtifactVersionError(f"Cannot read corpus records {path}: {exc}") from exc
    validated = validate_corpus_records(records, split=split)
    declared = int(manifest["splits"][split])
    if len(validated) != declared:
        raise ArtifactVersionError(
            f"Corpus split {split!r} contains {len(validated)} records but its v2 manifest declares {declared}."
        )
    return validated


def validate_assignments(payload: Any, *, source: str = "global assignments") -> dict[str, Any]:
    assignments = dict(_mapping(payload, source))
    _require_format(assignments, ASSIGNMENTS_FORMAT, source)
    _require_keys(
        assignments,
        ("method", "split", "codebook_size", "assignments", "summary", "source_codebook_fingerprint", "artifact_fingerprint"),
        source,
    )
    _verify_fingerprint(assignments, source)
    if not isinstance(assignments["assignments"], list):
        raise ArtifactVersionError(f"{source} has a malformed assignments table.")
    capacity = int(assignments["codebook_size"])
    if capacity <= 0:
        raise ArtifactVersionError(f"{source} has a non-positive codebook size.")
    for index, row in enumerate(assignments["assignments"]):
        label = f"{source} row {index}"
        row = _mapping(row, label)
        _require_keys(
            row,
            ("record_index", "checkpoint_index", "level", "hash", "global_id", "assignment_distance"),
            label,
        )
        global_id = int(row["global_id"])
        if global_id < 0 or global_id >= capacity:
            raise ArtifactVersionError(
                f"{label} uses global id {global_id} outside the declared codebook capacity {capacity}."
            )
    return assignments


def load_assignments(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = torch.load(path, map_location="cpu")
    return validate_assignments(payload, source=f"global assignments {path}")


def validate_assignments_against_records(
    assignments: Mapping[str, Any],
    records: list[dict[str, Any]],
    *,
    source: str = "global assignments",
) -> None:
    """Bind assignments to the exact certified-object corpus split.

    Every local checkpoint must occur exactly once and retain both its level
    and complete certificate identity. This prevents a same-shaped corpus from
    being silently mixed into another learned artifact chain.
    """

    expected: dict[tuple[int, int], tuple[int, str]] = {}
    for record_index, record in enumerate(records):
        for checkpoint_index, checkpoint in enumerate(record["checkpoints"]):
            expected[(record_index, checkpoint_index)] = (
                int(checkpoint["level"]),
                str(checkpoint["certified_object"]["certificate_hash"]),
            )

    observed: dict[tuple[int, int], tuple[int, str]] = {}
    for row_index, row in enumerate(assignments["assignments"]):
        key = (int(row["record_index"]), int(row["checkpoint_index"]))
        if key in observed:
            raise ArtifactVersionError(f"{source} contains duplicate assignment key {key}.")
        observed[key] = (int(row["level"]), str(row["hash"]))
        if key not in expected:
            raise ArtifactVersionError(
                f"{source} row {row_index} references checkpoint {key}, which is absent from the corpus split."
            )

    missing = sorted(set(expected) - set(observed))
    if missing:
        preview = ", ".join(str(key) for key in missing[:5])
        raise ArtifactVersionError(
            f"{source} does not completely cover the corpus split; "
            f"missing {len(missing)} checkpoint(s): {preview}."
        )
    for key, actual in observed.items():
        wanted = expected[key]
        if actual != wanted:
            raise ArtifactVersionError(
                f"{source} does not match corpus checkpoint {key}: "
                f"stored (level, certificate)={actual!r}, expected {wanted!r}."
            )


def validate_global_graph_payload(payload: Any, *, source: str = "global graph") -> dict[str, Any]:
    graph = dict(_mapping(payload, source))
    _require_format(graph, GLOBAL_GRAPH_FORMAT, source)
    _require_keys(
        graph,
        (
            "method",
            "split",
            "certified_object_format",
            "global_entries",
            "planning_nodes",
            "planning_edges",
            "initialization_edges",
            "coarsest_planning_prior",
            "num_global_conditions",
            "num_condition_edges",
            "planning_is_dag",
            "spec",
            "num_global_nodes",
            "num_planning_nodes",
            "num_planning_edges",
            "num_initialization_edges",
            "source_assignments_fingerprint",
            "source_codebook_fingerprint",
            "artifact_fingerprint",
        ),
        source,
    )
    _verify_fingerprint(graph, source)
    if graph["planning_is_dag"] is not True:
        raise ArtifactVersionError(f"{source} is not a validated level-augmented DAG.")
    if graph["certified_object_format"] != CERTIFIED_OBJECT_FORMAT:
        raise ArtifactVersionError(f"{source} does not contain v2 certified checkpoint prototypes.")
    entries = graph["global_entries"]
    planning_nodes = graph["planning_nodes"]
    planning_edges = graph["planning_edges"]
    initialization_edges = graph["initialization_edges"]
    if not all(isinstance(value, list) for value in (entries, planning_nodes, planning_edges, initialization_edges)):
        raise ArtifactVersionError(f"{source} has malformed canonical graph collections.")
    condition_by_global: dict[int, int] = {}
    for index, entry in enumerate(entries):
        label = f"{source} global entry {index}"
        entry = _mapping(entry, label)
        _require_keys(entry, ("global_id", "condition_id", "prototype_state", "prototype_certificate"), label)
        prototype = _mapping(entry["prototype_certificate"], f"{label} prototype certificate")
        _require_format(prototype, CERTIFIED_OBJECT_FORMAT, f"{label} prototype certificate")
        _require_keys(
            prototype,
            ("quotient_graph", "partition", "structural_summaries", "boundary_slots", "semantic"),
            f"{label} prototype certificate",
        )
        condition_by_global[int(entry["global_id"])] = int(entry["condition_id"])
    condition_ids = sorted(condition_by_global.values())
    expected_ids = list(range(len(entries)))
    if condition_ids != expected_ids or int(graph["num_global_conditions"]) != len(entries):
        raise ArtifactVersionError(f"{source} does not contain a compact contiguous v2 condition-id table.")
    for index, node in enumerate(planning_nodes):
        label = f"{source} planning node {index}"
        node = _mapping(node, label)
        _require_keys(
            node,
            ("planning_id", "global_id", "condition_id", "level", "prototype_state", "prototype_certificate"),
            label,
        )
        prototype = _mapping(node["prototype_certificate"], f"{label} prototype certificate")
        _require_format(prototype, CERTIFIED_OBJECT_FORMAT, f"{label} prototype certificate")
        _require_keys(
            prototype,
            ("quotient_graph", "partition", "structural_summaries", "boundary_slots", "semantic"),
            f"{label} prototype certificate",
        )
        global_id = int(node["global_id"])
        if global_id not in condition_by_global or int(node["condition_id"]) != condition_by_global[global_id]:
            raise ArtifactVersionError(f"{label} has a stale or inconsistent condition id.")
    for index, edge in enumerate(planning_edges):
        label = f"{source} planning edge {index}"
        edge = _mapping(edge, label)
        _require_keys(
            edge,
            (
                "global_edge_id",
                "guide_planning_id",
                "target_planning_id",
                "guide_level",
                "target_level",
                "delta",
                "operator_id",
                "relation",
                "correspondence_compatible",
                "representative_support_example",
            ),
            label,
        )
    for index, edge in enumerate(initialization_edges):
        label = f"{source} initialization edge {index}"
        edge = _mapping(edge, label)
        _require_keys(
            edge,
            ("global_edge_id", "target_planning_id", "guide_level", "target_level", "delta", "operator_id", "relation"),
            label,
        )
    if int(graph["num_condition_edges"]) != len(planning_edges) + len(initialization_edges):
        raise ArtifactVersionError(f"{source} has an inconsistent denoiser edge-condition table.")
    for field, actual in (
        ("num_global_nodes", len(entries)),
        ("num_planning_nodes", len(planning_nodes)),
        ("num_planning_edges", len(planning_edges)),
        ("num_initialization_edges", len(initialization_edges)),
    ):
        if int(graph[field]) != actual:
            raise ArtifactVersionError(f"{source} has inconsistent {field!r} metadata.")
    spec = _mapping(graph["spec"], f"{source} tensor spec")
    _require_keys(spec, ("atom_vocab", "bond_vocab", "max_nodes"), f"{source} tensor spec")
    return graph


def _validate_model_common(
    payload: Any,
    *,
    expected_format: str,
    source: str,
    global_model: bool,
) -> dict[str, Any]:
    checkpoint = dict(_mapping(payload, source))
    required = [
        "model_state",
        "config",
        "spec",
        "node_marginal",
        "edge_marginal",
        "activity_marginal",
        "max_level",
        "max_delta",
        "architecture",
        "artifact_fingerprint",
    ]
    if global_model:
        required.extend(
            (
                "num_global_codes",
                "max_global_edges",
                "global_graph_fingerprint",
                "assignments_fingerprint",
            )
        )
    _require_format(checkpoint, expected_format, source)
    _require_keys(checkpoint, required, source)
    _verify_fingerprint(checkpoint, source)
    spec = _mapping(checkpoint["spec"], f"{source} tensor spec")
    _require_keys(spec, ("atom_vocab", "bond_vocab", "max_nodes"), f"{source} tensor spec")
    config = _mapping(checkpoint["config"], f"{source} config")
    _require_keys(config, ("model", "diffusion"), f"{source} config")
    architecture = _mapping(checkpoint["architecture"], f"{source} architecture")
    if architecture.get("certified_object_format") != CERTIFIED_OBJECT_FORMAT:
        raise ArtifactVersionError(f"{source} predates the explicit v2 certified-object model input.")
    _require_relation_contract(architecture, source)
    for flag in ("mixed_state_corruption", "categorical_activity", "categorical_reverse_posterior"):
        if architecture.get(flag) is not True:
            raise ArtifactVersionError(f"{source} lacks required v2 architecture flag {flag!r}.")
    for flag in (
        "level0_atom_categorical",
        "binary_object_logits",
        "activity_capacity_mask",
        "factorized_edge_activity_posterior",
        "cfg_all_heads",
        "semantic_slot_alignment",
        "full_correspondence_supervision",
        "nested_generated_lineage",
        "boundary_slot_aliases",
    ):
        if architecture.get(flag) is not True:
            raise ArtifactVersionError(
                f"{source} lacks current paper-contract architecture flag {flag!r}; retrain it."
            )
    expected_node_dim = len(spec["atom_vocab"]) + 9
    expected_edge_dim = len(spec["bond_vocab"])
    node_marginal = checkpoint["node_marginal"]
    edge_marginal = checkpoint["edge_marginal"]
    if not torch.is_tensor(node_marginal) or int(node_marginal.numel()) != expected_node_dim:
        raise ArtifactVersionError(f"{source} has a stale node marginal or tensor specification.")
    if not torch.is_tensor(edge_marginal) or int(edge_marginal.numel()) != expected_edge_dim:
        raise ArtifactVersionError(f"{source} has a stale edge marginal or tensor specification.")
    expected_categorical = list(range(len(spec["atom_vocab"]) + 4, len(spec["atom_vocab"]) + 8))
    if list(architecture.get("categorical_node_indices", [])) != expected_categorical:
        raise ArtifactVersionError(f"{source} has stale mixed-state categorical fields.")
    if int(architecture.get("atom_feature_dim", -1)) != len(spec["atom_vocab"]):
        raise ArtifactVersionError(f"{source} has a stale level-0 atom-category specification.")
    activity_marginal = float(checkpoint["activity_marginal"])
    if not 0.0 < activity_marginal < 1.0:
        raise ArtifactVersionError(f"{source} has an invalid activity marginal.")
    if global_model and architecture.get("compact_condition_ids") is not True:
        raise ArtifactVersionError(f"{source} predates compact v2 global condition ids; retrain it.")
    return checkpoint


def validate_local_denoiser(payload: Any, *, source: str = "local denoiser") -> dict[str, Any]:
    return _validate_model_common(
        payload,
        expected_format=LOCAL_DENOISER_FORMAT,
        source=source,
        global_model=False,
    )


def validate_global_denoiser(payload: Any, *, source: str = "global denoiser") -> dict[str, Any]:
    return _validate_model_common(
        payload,
        expected_format=GLOBAL_DENOISER_FORMAT,
        source=source,
        global_model=True,
    )


def validate_codebook(payload: Any, *, source: str = "global codebook") -> dict[str, Any]:
    checkpoint = dict(_mapping(payload, source))
    _require_format(checkpoint, CODEBOOK_FORMAT, source)
    _require_keys(checkpoint, ("model_state", "spec", "config", "architecture", "artifact_fingerprint"), source)
    _verify_fingerprint(checkpoint, source)
    spec = _mapping(checkpoint["spec"], f"{source} tensor spec")
    config = _mapping(checkpoint["config"], f"{source} config")
    architecture = _mapping(checkpoint["architecture"], f"{source} architecture")
    _require_keys(spec, ("atom_vocab", "bond_vocab", "max_nodes"), f"{source} tensor spec")
    _require_keys(
        config,
        (
            "codebook_size",
            "hidden_dim",
            "num_layers",
            "num_heads",
            "dropout",
            "epochs",
            "batch_size",
            "lr",
            "beta",
            "weight_decay",
            "grad_clip",
            "seed",
            "loss_weights",
        ),
        f"{source} config",
    )
    _require_keys(
        architecture,
        (
            "input",
            "certified_object_format",
            "node_feature_dim",
            "edge_dim",
            "masked_mean_pooling",
            "codebook_capacity",
            "code_dimension",
        ),
        f"{source} architecture",
    )
    if architecture["input"] != "T(O(H))" or architecture["masked_mean_pooling"] is not True:
        raise ArtifactVersionError(f"{source} predates certified-object v2 encoding; retrain it.")
    if architecture["certified_object_format"] != CERTIFIED_OBJECT_FORMAT:
        raise ArtifactVersionError(f"{source} uses a stale certified-object schema; retrain it.")
    if int(architecture["codebook_capacity"]) != int(config["codebook_size"]):
        raise ArtifactVersionError(f"{source} has inconsistent codebook capacity metadata.")
    if int(architecture["code_dimension"]) != int(config["hidden_dim"]):
        raise ArtifactVersionError(f"{source} has inconsistent code dimension metadata.")
    return checkpoint
