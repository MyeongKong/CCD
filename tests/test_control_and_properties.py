from __future__ import annotations

import pytest

from checkpoint_dag_diffusion.checkpoint_state import CheckpointState, NodeSummary
from checkpoint_dag_diffusion.control import checkpoint_compatibility
from checkpoint_dag_diffusion import properties


def _state(num_nodes: int) -> CheckpointState:
    return CheckpointState(
        stage="h",
        node_summaries=tuple(NodeSummary.from_atom("C") for _ in range(num_nodes)),
        edges={(idx, idx + 1): "single" for idx in range(num_nodes - 1)},
        core_flags=tuple(True for _ in range(num_nodes)),
        ring_flags=tuple(False for _ in range(num_nodes)),
        original_atom_ids=tuple((idx,) for idx in range(num_nodes)),
    )


def _certificate(*, bond: str = "single", quotient: str = "q") -> dict:
    return {
        "quotient_graph_hash": quotient,
        "certificate_hash": "cert",
        "semantic": {
            "topology_family": "monocyclic",
            "ring_count": 1,
            "ring_size_hist": {"6": 1},
            "core_empty": False,
        },
        "boundary_slots": [
            {
                "multiplicity": 1,
                "boundary_bond_labels": [bond],
            }
        ],
    }


def test_scaffold_filter_checks_topology_boundary_and_optional_exact_quotient():
    requested = _certificate()
    compatible, details = checkpoint_compatibility(requested, _certificate(quotient="other"))
    assert compatible
    assert details["topology_match"] and details["boundary_match"]
    exact, _ = checkpoint_compatibility(requested, _certificate(quotient="other"), exact_quotient=True)
    assert not exact
    wrong_boundary, _ = checkpoint_compatibility(requested, _certificate(bond="double"))
    assert not wrong_boundary


def test_property_denominators_follow_supplement(monkeypatch):
    rows = [
        {"sample": _state(1).to_dict(), "requested_properties": {"logP": 1.0}},
        {"sample": _state(2).to_dict(), "requested_properties": {"logP": 1.0}},
        {"sample": _state(3).to_dict(), "requested_properties": {"logP": 1.0}},
    ]

    def evaluate(state):
        if state.num_nodes == 3:
            return None  # invalid/unevaluable: failure for Success@Tol
        return {"logP": 1.1 if state.num_nodes == 1 else 1.5, "QED": 0.5, "TPSA": 10.0}

    monkeypatch.setattr(properties, "evaluate_rdkit_properties", evaluate)
    result = properties.evaluate_property_records(rows)
    assert result is not None
    logp = result["per_property"]["logP"]
    assert logp["mae"] == pytest.approx((0.1 + 0.5) / 2)
    assert logp["success_at_tolerance"] == pytest.approx(1 / 3)
    assert result["success_at_tolerance"] == pytest.approx(1 / 3)
