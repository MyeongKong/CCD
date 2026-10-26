from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

from .artifacts import load_corpus_records
from .checkpoint_state import CheckpointState
from .global_graph import load_global_graph
from .metrics import (
    frechet_chemnet_distance,
    molecular_validity,
    ring_count,
    structural_validity,
    summarize_samples,
)
from .properties import evaluate_property_records
from .recontract import recontract_and_compare


def _prob(counter: Counter, keys: list[int]) -> list[float]:
    total = sum(counter.values())
    if total <= 0:
        return [1.0 / max(1, len(keys)) for _ in keys]
    return [counter.get(key, 0) / total for key in keys]


def js_divergence(p: list[float], q: list[float]) -> float:
    eps = 1e-12
    p = [max(eps, float(v)) for v in p]
    q = [max(eps, float(v)) for v in q]
    ps = sum(p)
    qs = sum(q)
    p = [v / ps for v in p]
    q = [v / qs for v in q]
    m = [(a + b) * 0.5 for a, b in zip(p, q)]

    def kl(a: list[float], b: list[float]) -> float:
        return sum(x * math.log(x / y) for x, y in zip(a, b))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def load_sample_states(samples_json: str | Path) -> list[CheckpointState]:
    data = json.loads(Path(samples_json).read_text(encoding="utf-8"))
    return [CheckpointState.from_dict(item["sample"]) for item in data]


def load_reference_states(corpus_dir: str | Path, split: str = "test") -> list[CheckpointState]:
    records = load_corpus_records(corpus_dir, split)
    states = []
    for record in records:
        if record["checkpoints"]:
            states.append(CheckpointState.from_dict(record["checkpoints"][0]["state"]))
    return states


def _node_lookup_from_graph(global_graph_path: str | Path | None) -> dict[int, dict[str, Any]]:
    if global_graph_path is None:
        return {}
    graph = load_global_graph(global_graph_path)
    return {int(node["global_id"]): node for node in graph["global_entries"]}


def _target_certificate(item: dict[str, Any], nodes: dict[int, dict[str, Any]]) -> dict[str, Any] | None:
    cert = item.get("target_certificate")
    if isinstance(cert, dict) and cert.get("certificate_hash"):
        return cert
    target_id = item.get("target_global_id")
    if target_id is None and item.get("global_path"):
        target_id = item["global_path"][-1]
    if target_id is None:
        return None
    node = nodes.get(int(target_id), {})
    cert = node.get("prototype_certificate")
    return cert if isinstance(cert, dict) and cert.get("certificate_hash") else None


def _requested_objects(item: dict[str, Any], nodes: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    path_objects = item.get("requested_path_objects")
    if isinstance(path_objects, list):
        valid = [dict(obj) for obj in path_objects if isinstance(obj, dict) and obj.get("certificate_hash")]
        if valid:
            return valid
    target = _target_certificate(item, nodes)
    return [] if target is None else [target]


def _primary_request_position(item: dict[str, Any], request_count: int) -> int:
    """Select the checkpoint used by mode-specific control metrics.

    Start control is tested at the first requested object and waypoint control
    at the matching path position.  Uncontrolled and complete-path samples use
    the terminal request; complete-path adherence itself is reported by the
    stricter all-request ``path_certification`` metric.
    """

    if request_count <= 0:
        return -1
    waypoint_id = item.get("requested_waypoint_global_id")
    waypoint_level = item.get("requested_waypoint_level")
    if waypoint_id is not None:
        for position, row in enumerate(item.get("path_trace", [])):
            if int(row.get("target_global_id", -1)) != int(waypoint_id):
                continue
            if waypoint_level is not None and int(row.get("target_level", -1)) != int(waypoint_level):
                continue
            if position < request_count:
                return position
    if item.get("requested_start_global_id") is not None:
        return 0
    return request_count - 1


def _certification_metrics(
    data: list[dict[str, Any]],
    generated: list[CheckpointState],
    global_graph_path: str | Path | None,
    *,
    require_rdkit: bool = False,
) -> dict[str, Any]:
    nodes = _node_lookup_from_graph(global_graph_path)
    rows = []
    for sample_index, (item, state) in enumerate(zip(data, generated)):
        requested = _requested_objects(item, nodes)
        if not requested:
            continue
        valid = bool(molecular_validity(state, require_rdkit=require_rdkit))
        recovery = recontract_and_compare(state, requested)
        comparisons = list(recovery.get("comparisons", []))
        primary_position = _primary_request_position(item, len(requested))
        primary = comparisons[primary_position] if 0 <= primary_position < len(comparisons) else {}
        rows.append(
            {
                "sample_index": sample_index,
                "primary_request_position": primary_position,
                "valid": valid,
                "exact": bool(primary.get("exact", False)),
                "semantic": bool(primary.get("semantic", False)),
                "qmatch": bool(primary.get("qmatch", False)),
                "summary_error": primary.get("summary_error"),
                "path_semantic": bool(comparisons) and all(bool(row.get("semantic", False)) for row in comparisons),
                "path_exact": bool(comparisons) and all(bool(row.get("exact", False)) for row in comparisons),
                "recontractable": bool(recovery.get("recontractable", False)),
                # Repair Failure is defined after the fixed sampling/projection
                # budget, using the recovered primary checkpoint condition.
                # Runs with a zero/absent budget report N/A.
                "repair_failed": (
                    None
                    if int(item.get("repair_budget", 0) or 0) <= 0
                    else not (valid and bool(primary.get("semantic", False)))
                ),
                "comparisons": comparisons,
            }
        )
    if not rows:
        return {
            "target_available_fraction": 0.0,
            "valid_target_count": 0,
            "cert_exact": None,
            "cert_semantic": None,
            "qmatch": None,
            "summary_error": None,
            "certified_validity": None,
            "certified_semantic_validity": None,
            "path_certification": None,
            "control_success": None,
            "recontractable_fraction": None,
            "repair_failure": None,
            "recontraction_records": [],
        }
    n = len(rows)
    valid_rows = [row for row in rows if row["valid"]]
    valid_n = len(valid_rows)
    summary_values = [float(row["summary_error"]) for row in valid_rows if row["summary_error"] is not None]
    repair_rows = [row for row in rows if row["repair_failed"] is not None]
    return {
        "target_available_fraction": n / max(1, len(data)),
        "valid_target_count": valid_n,
        # QMatch/Cert@Semantic/Cert@Exact use the valid-only denominator.
        "cert_exact": None if valid_n == 0 else sum(row["exact"] for row in valid_rows) / valid_n,
        "cert_semantic": None if valid_n == 0 else sum(row["semantic"] for row in valid_rows) / valid_n,
        "qmatch": None if valid_n == 0 else sum(row["qmatch"] for row in valid_rows) / valid_n,
        "summary_error": None if not summary_values else sum(summary_values) / len(summary_values),
        # Certified Validity uses all generation attempts and exact identity.
        "certified_validity": sum(row["valid"] and row["exact"] for row in rows) / max(1, len(data)),
        "certified_semantic_validity": sum(row["valid"] and row["semantic"] for row in rows) / max(1, len(data)),
        "path_certification": sum(row["valid"] and row["path_semantic"] for row in rows) / max(1, len(data)),
        "control_success": sum(row["valid"] and row["semantic"] for row in rows) / max(1, len(data)),
        "recontractable_fraction": sum(row["recontractable"] for row in rows) / max(1, len(data)),
        "repair_failure": (
            None
            if not repair_rows
            else sum(bool(row["repair_failed"]) for row in repair_rows) / max(1, len(data))
        ),
        "recontraction_records": rows,
    }


def evaluate_global_samples(
    samples_json: str | Path,
    corpus_dir: str | Path | None = None,
    reference_split: str = "test",
    global_graph_path: str | Path | None = None,
    require_rdkit: bool = False,
    compute_fcd: bool = False,
    fcd_device: str = "cpu",
    fcd_jobs: int = 1,
) -> dict[str, Any]:
    generated = load_sample_states(samples_json)
    molecular_summary = summarize_samples(generated)
    gen_ring = Counter(ring_count(state) for state in generated)
    structural = sum(structural_validity(state) for state in generated) / max(1, len(generated))
    validity = sum(molecular_validity(state, require_rdkit=require_rdkit) for state in generated) / max(1, len(generated))
    result: dict[str, Any] = {
        "num_samples": len(generated),
        "validity": validity,
        "structural_validity": structural,
        "rdkit_validity": molecular_summary["rdkit_validity"],
        "uniqueness": molecular_summary["unique_fraction"],
        "ring_count_hist": dict(sorted((str(k), int(v)) for k, v in gen_ring.items())),
    }
    if corpus_dir is not None:
        reference = load_reference_states(corpus_dir, split=reference_split)
        result["reference_split"] = reference_split
        result["reference_num_samples"] = len(reference)
        if reference:
            ref_ring = Counter(ring_count(state) for state in reference)
            keys = sorted(set(ref_ring) | set(gen_ring))
            result["ring_count_js"] = js_divergence(_prob(ref_ring, keys), _prob(gen_ring, keys))
            if compute_fcd:
                result["fcd"] = frechet_chemnet_distance(
                    generated,
                    reference,
                    device=fcd_device,
                    n_jobs=fcd_jobs,
                )
    data = json.loads(Path(samples_json).read_text(encoding="utf-8"))
    result["certification"] = _certification_metrics(
        data,
        generated,
        global_graph_path,
        require_rdkit=require_rdkit,
    )
    property_result = evaluate_property_records(data)
    if property_result is not None:
        result["property_targeting"] = property_result
    if data and "global_path" in data[0]:
        paths = Counter(tuple(item["global_path"]) for item in data)
        result["path_diversity"] = {
            "num_unique_paths": len(paths),
            "unique_path_fraction": len(paths) / max(1, len(data)),
            "path_length_hist": dict(sorted((str(k), int(v)) for k, v in Counter(len(item["global_path"]) for item in data).items())),
        }
        result["control"] = {
            "waypoint_reached": sum(
                bool(item.get("checkpoint_consistency", {}).get("waypoint_reached", True)) for item in data
            )
            / max(1, len(data)),
            "reached_level0": sum(
                bool(item.get("checkpoint_consistency", {}).get("reached_level0", False)) for item in data
            )
            / max(1, len(data)),
        }
    return result
