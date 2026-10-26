# Checkpoint-Conditioned Diffusion (CCD)

Implementation accompanying **Checkpoint-Conditioned Diffusion: Reusable
Quotient-Graph Plans for Controllable Molecular Generation**.

CCD represents intermediate molecular structures as explicit quotient checkpoints with structural summaries and attachment metadata. 
A shared coarse-to-fine transition DAG supports start, waypoint, and path requests; checkpoint-conditioned diffusion realizes the selected refinements. 
Final-output recontraction measures agreement with the requested checkpoint representations.
A structural checkpoint is a partially resolved graph; a saved neural-network checkpoint is a separate training artifact.

```text
molecular graphs -> local quotient-checkpoint objects
                 -> checkpoint encoder + VQ codebook
                 -> shared, level-augmented checkpoint DAG
                 -> checkpoint-conditioned diffusion
                 -> final molecule -> recontraction and request comparison
```

Checkpoint agreement and molecular validity are measured separately. The
method does not guarantee chemical correctness, satisfaction of every request,
or compatibility of every path assembled from shared entries.

The global path policy is nonparametric. It samples observed outgoing edges in
proportion to `count ** (1 / temperature)`; one selected global edge invokes a
complete local reverse-diffusion segment.

## Object boundaries

The implementation distinguishes the following objects:

```text
H                         visible labeled quotient graph
O(H)                      checkpoint object (H, pi, sigma, B, eta, h_repr)
T(O(H))                   padded neural tensors (X_obj, A_Q, m)
z_k                       learned 128-dimensional VQ vector
C_k                       explicit reusable dataset-level entry
u = (C_k, contraction_l)  node in the acyclic global planning graph
```

`CheckpointState` is a construction/decoded-state carrier, not the model-facing
definition of bare `H`. Here `pi` is source correspondence, `sigma` contains
structural summaries, `B` contains attachment boundaries, `eta` is the semantic
descriptor, and `h_repr` is the stored representation signature. Source-atom IDs
are retained for provenance but excluded from comparison signatures.

Some Python identifiers and serialized fields retain terminology from the
prototype. These names are compatibility identifiers, not claims of formal
certification:

| Existing code or artifact name | Meaning in the submission |
|---|---|
| `CertifiedCheckpointObject` | Checkpoint object `O(H)` |
| `certified_object_to_tensors` | Tensorization `T(O(H))` |
| `certified_object.v2` | Versioned checkpoint-object storage format |
| `certificate_hash` | Representation signature `h_repr` |
| `semantic_hash` | Semantic signature `h_eta` |
| YAML `certification` section | Recontraction/comparison configuration |

The commands below use the existing implementation names.

Every local coarse-to-fine edge stores

```text
relation = (guide_level, target_level, delta, operator_id)
mu       = nested-partition node/boundary correspondence
```

Global planning edges strictly decrease the level, which ensures acyclicity.
Sharing a VQ code does not imply equality of checkpoint representations or
transferable attachment interfaces. Representative-map transfer checks endpoint
isomorphisms and boundary-slot alignment; a failed transfer leaves an empty map
while retaining the planning edge. This local alignment check does not prove
that an entire merged path has a common molecular realization.

Codebook, assignment, global-DAG, and denoiser artifacts carry fingerprints to
check consistency within a training chain. These are artifact-integrity checks,
separate from final-output structural matching.

## Reference architecture

The checked-in QM9 configuration provides these architecture settings:

| Component | Setting |
|---|---|
| Checkpoint-object tensor | max 32 nodes, node width 21, 5 edge categories |
| Checkpoint encoder | width 128, 2 layers, 4 heads, FFN 512, dropout 0.1 |
| Codebook | capacity 512, code width 128 |
| Target / guide encoders | width 192, 4 / 2 layers, 6 heads, FFN 768 |
| Guide fusion | one target-to-guide cross-attention block |
| Diffusion | horizon 200, 32 reverse updates per diffusion segment |
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

A CCD path with `J` refinement edges also has an initialization segment, giving
`32 * (J + 1)` reverse updates before retries. Classifier-free guidance can
require two denoiser calls per update. The empirical coarsest-state planning
prior is distinct from the terminal-noise marginals used by diffusion.

### Submission settings versus repository defaults

The commands below illustrate the implementation pipeline; they do not by
themselves reproduce the reported benchmark tables. In `configs/qm9_full.yaml`,
`representation.epochs` is 1, `train.epochs` is 100, and `seed` is 0. The sampler
CLI defaults to two refinement retries. The submission describes 100
representation epochs, 200 diffusion epochs, split seed 2026, training/sampling
seeds 11, 29, and 47, and zero local retries. Match those settings explicitly
when preparing a submission experiment.

The 32-node configuration is for QM9. The submission uses capacities 64 and 96
for MOSES and GuacaMol; preprocessing and representation settings must agree.
The submission's sanitization, largest-component selection, deduplication,
request assignments, and evaluation budgets must also be reproduced explicitly.

## Installation and tests

```bash
# Run from the repository root (Python 3.10 or newer).
python -m pip install -e '.[chem,test]'
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src pytest -q -p no:cacheprovider
```

FCD is optional:

```bash
python -m pip install -e '.[eval]'
```

The contract tests cover the checkpoint representation hash fixture, `T(O(H))`
shape, `H`/metadata separation, permutation equivariance, model parameter
counts, non-prefix `mu`, level-DAG acyclicity, planner controls, mixed-state
reverse transitions, cross-artifact fingerprints, matching-metric denominators,
and recontraction of every requested path object.

## Training and evaluation workflow

The scripts retain `_qm9` in their filenames for compatibility, but accept any
heavy-atom SMILES text file. Dataset-specific files and output directories can
therefore be used for QM9, MOSES, or GuacaMol.

### 1. Build local checkpoint DAG objects

```bash
PYTHONPATH=src python scripts/preprocess_dag_qm9.py \
  --smiles_txt /path/to/dataset.smi \
  --out_dir runs/qm9_corpus \
  --max_nodes 32 \
  --seed 2026
```

The preprocessor creates train/validation/test partitions internally. Supply
the intended dataset input rather than treating this command as an importer
for an already fixed training split. Preserve the required benchmark split
protocol and build the learned library only from training records.

Each corpus record contains the source graph, every local checkpoint,
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
support examples, structural and representation-signature statistics, and modal-signature
purity. Modal purity is not the same as agreement with the selected
representative, and neither establishes interface compatibility.

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
  --num_samples 1000 \
  --max_refinement_retries 0

PYTHONPATH=src python scripts/evaluate_global_qm9.py \
  --samples_json runs/qm9_samples/samples.json \
  --global_graph runs/qm9_global/global_graph.pt \
  --corpus_dir runs/qm9_corpus \
  --reference_split test \
  --require_rdkit \
  --out runs/qm9_samples/eval.json
```

Evaluation converts the level-0 output back into a molecule, reconstructs its
checkpoints using the same contractor, and compares them with fixed requests.
Requests use contractor mode/radius where available, with level as a fallback.
QMatch checks labeled quotient isomorphism after a signature prefilter;
Semantic and Exact compare their respective digests. Exact means agreement
under the implemented representation and serialization, not recovery of hidden
connectivity or a collision-free isomorphism proof. Cycle-basis descriptors and
multi-anchor selection can depend on atom ordering.

The evaluator currently emits legacy JSON keys. Interpret them as follows:

| JSON key | Submission terminology | Denominator |
|---|---|---|
| `qmatch` | QMatch | Valid outputs with an available request |
| `cert_semantic` | Match@Semantic | Valid outputs with an available request |
| `cert_exact` | Match@Exact | Valid outputs with an available request |
| `certified_validity` | Valid Exact Match | All supplied attempts |
| `certified_semantic_validity` / `control_success` | Valid semantic match | All supplied attempts |
| `path_certification` | Semantic PathMatch | All supplied attempts |
| `repair_failure` | Failure after the specified retry budget | All supplied attempts; unavailable without a positive budget |

For the submission's valid-only denominators, every registered attempt must
have its intended request recorded: check `target_available_fraction == 1`.
Keep invalid and unsuccessful attempts in the input. With no valid requested
outputs, valid-only rates are undefined (`null`). PathMatch requires every
requested checkpoint to agree after recontraction of the same final molecule.
Per-sample `path_exact` is available in `recontraction_records`; the evaluator
does not expose a separate aggregate exact-PathMatch field. Primary-request
matching, semantic PathMatch, exact PathMatch, and joint structural/property
success must not be substituted for one another.

The sampling example disables retries explicitly. If retries are enabled,
report their budget and cost separately and retain failed attempts in the
accounting. These commands do not automatically implement the submission's
100-requests-by-100-outputs assignment protocol.

To compute FCD on valid generated SMILES (without deduplication) and the held-
out reference set, install the `eval` extra and add `--compute_fcd`.

## Structural controls and model variant

Codes should be specified with a level whenever one code occurs at more than
one level. The IDs below are illustrative: replace them with populated entries
and observed paths from your own `global_graph.pt`. A selected path is a
request; final-output recontraction determines whether it was realized.

```bash
# Start-object control
PYTHONPATH=src python scripts/sample_global_qm9.py \
  --checkpoint runs/qm9_ccd/checkpoints/last.pt \
  --global_graph runs/qm9_global/global_graph.pt \
  --out_dir runs/start_control \
  --start_code C42 --start_level 3 --max_refinement_retries 0

# Waypoint control (unreachable requests fail instead of being relaxed)
PYTHONPATH=src python scripts/sample_global_qm9.py \
  --checkpoint runs/qm9_ccd/checkpoints/last.pt \
  --global_graph runs/qm9_global/global_graph.pt \
  --out_dir runs/waypoint_control \
  --waypoint_code C137 --waypoint_level 1 --max_refinement_retries 0

# Complete path control
PYTHONPATH=src python scripts/sample_global_qm9.py \
  --checkpoint runs/qm9_ccd/checkpoints/last.pt \
  --global_graph runs/qm9_global/global_graph.pt \
  --out_dir runs/path_control \
  --path C42@L3,C137@L1,C301@L0 --max_refinement_retries 0

# Exact-Plan Final-Diffusion ablation
PYTHONPATH=src python scripts/sample_global_qm9.py \
  --checkpoint runs/qm9_ccd/checkpoints/last.pt \
  --global_graph runs/qm9_global/global_graph.pt \
  --out_dir runs/exact_plan \
  --variant exact_plan_final_diffusion --max_refinement_retries 0
```

Exact-Plan retrieves positive-level checkpoint states and diffuses level-0
targets. Its initialization and computational cost differ from CCD, so this
comparison is not an equal-compute comparison by default.

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
  --out_dir runs/scaffold_control \
  --max_refinement_retries 0
```

## Optional scalar-property reranking

This optional utility reranks an existing candidate pool using independent
RDKit property evaluation while preserving each candidate's checkpoint
condition. Generate `runs/qm9_candidates/samples.json` first and retain the full
pool for budget accounting. Reranking is a separate selection protocol; it is
not the submission's one-trajectory-per-attempt evaluation:

```bash
PYTHONPATH=src python scripts/rerank_property_samples.py \
  --samples_json runs/qm9_candidates/samples.json \
  --target logP=1.5 \
  --target QED=0.7 \
  --num_select 1000 \
  --out runs/qm9_property/samples.json
```

Evaluating only the selected file measures the selected subset, not success
over the full candidate budget. To measure all-attempt property success, assign
property targets before generation and evaluate all attempts, including failures.
Per-property Success@Tol uses attempts assigned that property; the aggregate is
a macro-average, not simultaneous satisfaction of all requested properties.

The evaluator reports MAE on valid, evaluable outputs and Success@Tol using the
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
- `eval.json`: molecular quality, control, recontraction, and checkpoint-matching metrics (with legacy JSON keys listed above).

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
