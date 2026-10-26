# Checkpoint-Conditioned Diffusion (CCD)

Implementation accompanying **Checkpoint-Conditioned Diffusion: Reusable Quotient-Graph Plans for Controllable Molecular Generation**.

CCD uses explicit quotient-graph checkpoints to plan molecular generation. A shared coarse-to-fine DAG organizes reusable structures, and checkpoint-conditioned diffusion realizes the selected transitions. Start, waypoint, and path requests share the same interface. Agreement with a requested structure is evaluated by recontracting the generated molecule.

## Overview

```text
Molecules → Local checkpoints → Shared checkpoint DAG
         → Structural plan → Conditional diffusion → Recontraction
```

Checkpoint objects combine a visible graph with structural summaries and attachment metadata. The learned codebook organizes these objects into reusable planning entries. Molecular validity and checkpoint agreement are evaluated separately; a selected plan does not guarantee that every generated output will satisfy it.

## Installation

Use Python 3.10 or newer and run from the repository root:

```bash
python -m pip install -e '.[chem,test]'
```

For FCD evaluation:

```bash
python -m pip install -e '.[eval]'
```

## Workflow

Run the stages in order, keeping artifacts from the same training run together.

| Stage | Entry point |
|---|---|
| Prepare local checkpoint graphs | [`preprocess_dag_qm9.py`](scripts/preprocess_dag_qm9.py) |
| Learn checkpoint representations | [`train_global_codebook_qm9.py`](scripts/train_global_codebook_qm9.py) |
| Build the shared planning DAG | [`build_global_graph_qm9.py`](scripts/build_global_graph_qm9.py) |
| Train conditional diffusion | [`train_global_dag_qm9.py`](scripts/train_global_dag_qm9.py) |
| Generate molecules with structural requests | [`sample_global_qm9.py`](scripts/sample_global_qm9.py) |
| Evaluate generated molecules | [`evaluate_global_qm9.py`](scripts/evaluate_global_qm9.py) |

Each script provides its arguments through `--help`, for example:

```bash
python scripts/sample_global_qm9.py --help
```

The `_qm9` filenames are retained for compatibility. Other datasets require matching input preparation and capacity settings. Start from [`configs/qm9_full.yaml`](configs/qm9_full.yaml) and use separate output directories for each dataset and seed. Sampling supports start, waypoint, and complete-path requests, as well as the Exact-Plan variant. Request identifiers must come from the generated planning DAG.

## Experimental configuration

The submission evaluates CCD on QM9, MOSES, and GuacaMol using training-derived checkpoint libraries and held-out evaluation data.

| Setting | Submission protocol |
|---|---|
| Data | Sanitized, deduplicated heavy-atom graphs; largest connected component; oversized molecules removed before splitting |
| Split | 80/10/10 train/validation/test; split seed 2026 |
| Graph capacity | QM9: 32; MOSES: 64; GuacaMol: 96 |
| Training | 100 representation epochs; 200 diffusion epochs; validation-loss model selection |
| Seeds | 11, 29, and 47 |
| Generation | 100 fixed requests × 100 outputs per dataset and seed; no repair or retries |

Architecture and optimization details are given in the paper. The checked-in configuration retains prototype defaults: override the epoch counts and seed, and pass `--max_refinement_retries 0` for submission-style sampling. The supplied trainers do not automatically implement validation-based model selection or mixed precision; the complete benchmark protocol requires additional orchestration.

## Evaluation

Evaluation covers molecular quality, checkpoint matching, and scalar-property targeting. Match@Semantic and Match@Exact measure agreement among valid outputs; Valid Exact Match includes all attempts. PathMatch checks the requested checkpoints against the same final molecule. Exact matching refers to the implemented representation, not complete recovery of hidden molecular structure.

Property tasks are evaluated separately using targets derived from validation data. RDKit evaluates logP, QED, and TPSA; the QM9 gap requires an independent evaluator. Preserve the full attempt set when reporting success rates. The optional property-reranking utility is separate from the submission protocol.

Some API names and output fields retain legacy terminology for compatibility. In evaluation output, `cert_semantic`, `cert_exact`, `certified_validity`, and `path_certification` correspond to Match@Semantic, Match@Exact, Valid Exact Match, and semantic PathMatch, respectively. These identifiers do not imply formal certification.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src pytest -q -p no:cacheprovider
```

Tests cover checkpoint representations, planning, diffusion, artifact consistency, and recontraction-based evaluation.
