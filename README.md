# Checkpoint-Conditioned Diffusion

Paper-aligned implementation of **Checkpoint-Conditioned Diffusion (CCD)**. The
reproducible path is a three-stage pipeline:

```text
verified local objects -> checkpoint encoder + VQ codebook
                        -> level-augmented global checkpoint DAG
                        -> checkpoint-conditioned edge denoiser
```

The global path policy is nonparametric. It samples observed outgoing edges in
proportion to `count ** (1 / temperature)`; one selected global edge invokes a
complete local reverse-diffusion segment.

## Object boundaries

The implementation keeps the five paper concepts separate:

```text
H                         visible labeled quotient graph
O(H)                      verified object (H, pi, sigma, B, eta, h_cert)
T(O(H))                   padded neural tensors (X_obj, A_Q, m)
z_k                       learned 128-dimensional VQ vector
C_k                       explicit reusable dataset-level entry
u = (C_k, contraction_l)  node in the acyclic global planning graph
```

Every local coarse-to-fine edge stores

```text
relation = (guide_level, target_level, delta, operator_id)
mu       = nested-partition node/boundary correspondence
```

Global planning edges strictly decrease the level. A local `mu` is transferred
to global representative orderings only when both endpoints are exactly
isomorphic; incompatible index maps are never applied as positional prefixes.
Codebook, assignment, global-DAG, and denoiser artifacts carry deterministic
fingerprints, and every consumer rejects a same-shaped artifact from a
different training chain.

## Reference architecture

The main YAML configurations encode the supplementary settings:

| Component | Setting |
|---|---|
| Object tensor | max 32 nodes, node width 21, 5 edge categories |
| Checkpoint encoder | width 128, 2 layers, 4 heads, FFN 512, dropout 0.1 |
| Codebook | capacity 512, code width 128 |
| Target / guide encoders | width 192, 4 / 2 layers, 6 heads, FFN 768 |
| Guide fusion | one target-to-guide cross-attention block |
| Diffusion | horizon 200, 32 denoiser evaluations per selected edge |
| Optimization | AdamW, lr 2e-4, weight decay 1e-4, batch 64, clip 1.0 |
| Sampling | CFG 1.3, edge temperature 0.9, no-edge bias 2.0, degree cap 4 |
| Planning | transition temperature 1.0, at most 8 global edges |

Numerical coarse-object summaries are mixed toward empirical marginals. At
level 0, atom identities are corrupted and reversed categorically; binary
object fields, node activity, and edges likewise use categorical marginal
corruption. Edge and activity posteriors are sampled factorwise and then
projected together so inactive endpoints imply no-edge. The clean target size
is not passed to the denoiser as an attention mask.

The codebook loss weights are reconstruction/topology/VQ/commitment/hash =
`1.0/0.05/1.0/0.25/0.05`. The refinement loss weights are
node/bond/activity/split/attachment/guide =
`1.0/1.0/0.25/0.05/0.05/0.10`.

## Installation and tests

```bash
cd checkpoint_dag_diffusion_project
python -m pip install -e '.[chem,test]'
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src pytest -q -p no:cacheprovider
```

FCD is optional:

```bash
python -m pip install -e '.[eval]'
```

The contract tests cover the published checkpoint hash fixture, `T(O(H))`
shape, `H`/metadata separation, permutation equivariance, model parameter
counts, non-prefix `mu`, level-DAG acyclicity, planner controls, mixed-state
reverse transitions, cross-artifact fingerprints, certification denominators,
and recontraction of every requested path object.

## Paper pipeline

The scripts retain `_qm9` in their filenames for compatibility, but accept any
heavy-atom SMILES text file. Dataset-specific files and output directories can
therefore be used for QM9, MOSES, or GuacaMol.

### 1. Build certified local DAG objects

```bash
PYTHONPATH=src python scripts/preprocess_dag_qm9.py \
  --smiles_txt /path/to/train.smi \
  --out_dir runs/qm9_corpus \
  --max_nodes 32 \
  --seed 0
```

Each corpus record contains the source graph, every verified local checkpoint,
the explicit object fields, adjacent/skip refinement edges, and `mu`.

### 2. Learn the shared checkpoint representation

```bash
PYTHONPATH=src python scripts/train_global_codebook_qm9.py \
  --config configs/qm9_full.yaml \
  --corpus_dir runs/qm9_corpus \
  --out_dir runs/qm9_codebook
```

The `representation` and `train` sections of the YAML are authoritative. CLI
arguments can override individual settings. Only populated VQ codes become
explicit global entries; each entry stores a nearest assigned representative,
support examples, structural/certification statistics, and code purity.

### 3. Construct the level-augmented global DAG

```bash
PYTHONPATH=src python scripts/build_global_graph_qm9.py \
  --corpus_dir runs/qm9_corpus \
  --out_dir runs/qm9_global \
  --method learned \
  --codebook runs/qm9_codebook/codebook.pt
```

`global_entries` are the explicit `C_k` objects. `planning_nodes` are distinct
`(C_k, level)` states, and `planning_edges` aggregate observed local
coarse-to-fine transitions and counts. Noise initialization edges are stored
separately. Sparse VQ/global IDs are also assigned compact denoiser-side
`condition_id` values; these learned identifier embeddings remain distinct
from both `C_k` and `z_k`.

### 4. Train checkpoint-conditioned edge diffusion

```bash
PYTHONPATH=src python scripts/train_global_dag_qm9.py \
  --config configs/qm9_full.yaml \
  --corpus_dir runs/qm9_corpus \
  --global_graph runs/qm9_global/global_graph.pt \
  --assignments runs/qm9_global/global_assignments.pt \
  --run_dir runs/qm9_ccd
```

Guide and global-identifier conditions are independently dropped with
probability 0.1. The saved model records its architecture, tensor spec,
empirical node/edge/activity marginals, and graph artifact paths.

### 5. Sample and verify

```bash
PYTHONPATH=src python scripts/sample_global_qm9.py \
  --checkpoint runs/qm9_ccd/checkpoints/last.pt \
  --global_graph runs/qm9_global/global_graph.pt \
  --out_dir runs/qm9_samples \
  --num_samples 1000

PYTHONPATH=src python scripts/evaluate_global_qm9.py \
  --samples_json runs/qm9_samples/samples.json \
  --global_graph runs/qm9_global/global_graph.pt \
  --corpus_dir runs/qm9_corpus \
  --reference_split test \
  --require_rdkit \
  --out runs/qm9_samples/eval.json
```

Evaluation converts the level-0 output back into a molecule, runs the same
contractor at every requested level, and compares recovered objects. QMatch,
Cert@Semantic, and Cert@Exact use the valid-only denominator; Certified
Validity and Path Certification use all generation attempts. Repair Failure is
reported when a fixed resampling budget was used.

To compute FCD on valid generated SMILES (without deduplication) and the held-
out reference set, install the `eval` extra and add `--compute_fcd`.

## Structural controls and model variant

Codes should be specified with a level whenever one code occurs at more than
one level.

```bash
# Start-object control
PYTHONPATH=src python scripts/sample_global_qm9.py \
  --checkpoint runs/qm9_ccd/checkpoints/last.pt \
  --global_graph runs/qm9_global/global_graph.pt \
  --out_dir runs/start_control \
  --start_code C42 --start_level 3

# Waypoint control (unreachable requests fail instead of being relaxed)
PYTHONPATH=src python scripts/sample_global_qm9.py \
  --checkpoint runs/qm9_ccd/checkpoints/last.pt \
  --global_graph runs/qm9_global/global_graph.pt \
  --out_dir runs/waypoint_control \
  --waypoint_code C137 --waypoint_level 1

# Complete path control
PYTHONPATH=src python scripts/sample_global_qm9.py \
  --checkpoint runs/qm9_ccd/checkpoints/last.pt \
  --global_graph runs/qm9_global/global_graph.pt \
  --out_dir runs/path_control \
  --path C42@L3,C137@L1,C301@L0

# Exact-Plan Final-Diffusion ablation
PYTHONPATH=src python scripts/sample_global_qm9.py \
  --checkpoint runs/qm9_ccd/checkpoints/last.pt \
  --global_graph runs/qm9_global/global_graph.pt \
  --out_dir runs/exact_plan \
  --variant exact_plan_final_diffusion
```

Scaffold/boundary control runs the same contractor and encoder, filters active
global objects by topology and attachment compatibility, and selects the
nearest remaining code:

```bash
PYTHONPATH=src python scripts/sample_global_qm9.py \
  --checkpoint runs/qm9_ccd/checkpoints/last.pt \
  --global_graph runs/qm9_global/global_graph.pt \
  --scaffold_codebook runs/qm9_codebook/codebook.pt \
  --scaffold_smiles 'c1ccccc1' \
  --scaffold_mode pure_core \
  --out_dir runs/scaffold_control
```

## Scalar-property reranking

CCD's graph-valued checkpoint condition remains fixed while a larger candidate
budget is reranked using independent RDKit property evaluation:

```bash
PYTHONPATH=src python scripts/rerank_property_samples.py \
  --samples_json runs/qm9_candidates/samples.json \
  --target logP=1.5 \
  --target QED=0.7 \
  --num_select 1000 \
  --out runs/qm9_property/samples.json
```

The evaluator reports valid-only MAE and all-attempt Success@Tol using the
paper tolerances (logP 0.25, QED 0.05, TPSA 5.0, gap 0.15). An external gap
evaluator can place values in `evaluated_properties.gap` before evaluation.

## Versioned artifacts

- `records.pt`: local DAGs with `certified_object.v2` and correspondence maps.
- `codebook.pt`: checkpoint Transformer, VQ vectors, auxiliary decoder, config.
- `global_assignments.pt`: local object-to-code assignments and distances.
- `global_graph.pt`: explicit `C_k` entries plus acyclic `(C_k, level)` graph.
- `last.pt`: global-conditioned denoiser and empirical corruption marginals.
- `samples.json`: final graphs, requested objects, every realized intermediate
  `O(H)`, level/path trace, and repair flags.
- `eval.json`: molecular quality, control, recontraction, and certification metrics.

### Regeneration requirement

Artifacts produced by the earlier prototype are intentionally not upgraded in
place. In particular, old corpora lack the separated structural field
`sigma`, guide-first relations, and `mu`; old global graphs are code-only and
can contain self-loops; and older denoisers do not implement categorical
level-0 atoms, binary-logit prediction, or capacity-aware activity. Regenerate
the corpus, codebook, assignments, global graph, and denoiser in order. Current
consumers validate these contracts and their upstream fingerprints, rejecting
v1, stale-v2, modified, or cross-run payloads rather than silently projecting
them.
