from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Iterable

from .checkpoint_state import CheckpointState
from .chemistry import checkpoint_state_to_rdkit_mol, rdkit_available


PROPERTY_TOLERANCES: dict[str, float] = {
    "logP": 0.25,
    "QED": 0.05,
    "TPSA": 5.0,
    "gap": 0.15,
}


def evaluate_rdkit_properties(state: CheckpointState) -> dict[str, float] | None:
    """Evaluate the independently computed molecular properties used in the paper."""

    if not rdkit_available():
        return None
    from rdkit.Chem import Crippen, Descriptors, QED  # type: ignore

    molecule = checkpoint_state_to_rdkit_mol(state, sanitize=True)
    if molecule is None:
        return None
    return {
        "logP": float(Crippen.MolLogP(molecule)),
        "QED": float(QED.qed(molecule)),
        "TPSA": float(Descriptors.TPSA(molecule)),
    }


def evaluate_property_records(
    records: Iterable[dict[str, Any]],
    *,
    tolerances: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    """Compute valid-only MAE and all-attempt Success@Tol.

    A precomputed ``gap`` value (or any externally evaluated value) may be
    supplied in a record's ``evaluated_properties`` field.  RDKit evaluates
    logP, QED and TPSA independently from the generator.
    """

    rows = list(records)
    if not rows or not any(row.get("requested_properties") for row in rows):
        return None
    thresholds = {**PROPERTY_TOLERANCES, **dict(tolerances or {})}
    absolute_errors: dict[str, list[float]] = defaultdict(list)
    successes: dict[str, int] = defaultdict(int)
    request_counts: dict[str, int] = defaultdict(int)
    evaluated_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        requested = {str(key): float(value) for key, value in row.get("requested_properties", {}).items()}
        if not requested:
            continue
        state = CheckpointState.from_dict(row["sample"])
        rdkit_values = evaluate_rdkit_properties(state)
        evaluated: dict[str, float] = {}
        if rdkit_values is not None:
            evaluated.update({str(key): float(value) for key, value in row.get("evaluated_properties", {}).items()})
            evaluated.update(rdkit_values)
        result = {"sample_index": index, "requested": requested, "evaluated": evaluated, "errors": {}}
        for name, target in requested.items():
            request_counts[name] += 1
            if name not in evaluated:
                result["errors"][name] = None
                continue
            error = abs(float(evaluated[name]) - target)
            absolute_errors[name].append(error)
            result["errors"][name] = error
            if error <= float(thresholds[name]):
                successes[name] += 1
        evaluated_rows.append(result)

    metrics: dict[str, Any] = {}
    for name in sorted(request_counts):
        errors = absolute_errors[name]
        metrics[name] = {
            "mae": None if not errors else sum(errors) / len(errors),
            "success_at_tolerance": successes[name] / max(1, request_counts[name]),
            "tolerance": float(thresholds[name]),
            "requested": int(request_counts[name]),
            "evaluable": len(errors),
        }
    return {
        "per_property": metrics,
        "success_at_tolerance": (
            None
            if not metrics
            else sum(value["success_at_tolerance"] for value in metrics.values()) / len(metrics)
        ),
        "records": evaluated_rows,
    }


def rerank_property_records(
    records: Iterable[dict[str, Any]],
    *,
    targets: dict[str, float],
    num_select: int,
    tolerances: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Rerank a candidate budget without changing its checkpoint plan."""

    thresholds = {**PROPERTY_TOLERANCES, **dict(tolerances or {})}
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for index, original in enumerate(records):
        row = dict(original)
        state = CheckpointState.from_dict(row["sample"])
        rdkit_values = evaluate_rdkit_properties(state)
        evaluated = dict(rdkit_values or {})
        if rdkit_values is not None:
            evaluated.update(
                {str(key): float(value) for key, value in row.get("evaluated_properties", {}).items()}
            )
        if rdkit_values is None or any(name not in evaluated for name in targets):
            score = float("inf")
        else:
            score = sum(
                abs(float(evaluated[name]) - float(target)) / float(thresholds[name])
                for name, target in targets.items()
            )
        row["requested_properties"] = {str(key): float(value) for key, value in targets.items()}
        row["evaluated_properties"] = evaluated
        row["property_reranking_score"] = None if math.isinf(score) else score
        scored.append((score, index, row))
    return [row for _, _, row in sorted(scored, key=lambda value: (value[0], value[1]))[: int(num_select)]]
