# Resource estimation

Estimation in CUDA-Q Logical is two explicit tiers over the _same_ linked
definitions: device-independent logical counts, and static counts over a
selected QEC realization. One call shape serves both —
`ql.estimate(value, tier=...)`. The value may be a `Build` or an authoring
definition; a definition is first compiled through its normal default pipeline,
and the estimator then validates that the resulting stage matches the requested
tier, failing with a typed diagnostic when they disagree.

| Tier                       | Needs                                                     | Returns                                                                                                                                                         |
| -------------------------- | --------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ql.estimate.Tier.LOGICAL` | a verified P0 program — no device, no code                | `ql.estimate.LogicalProfile`: action/instrument totals, peak logical qubits, idle/discard counts, an action-depth upper bound, synthesis demand                 |
| `ql.estimate.Tier.STATIC`  | a selected P2 build — code, gadgets, and protocols chosen | `ql.estimate.FabricCounts`: per-operation counts, gadget/protocol call totals, resource requests, postselection bookkeeping, syndrome rounds, peak live patches |

`Tier.STATIC` is the default, so `ql.estimate(p2_build)` needs no `tier=`
argument. Both result types are immutable plain-data values with `to_dict()`
projections, and both record the `build_root` and `build_sha256` they were
derived from.

## `Tier.LOGICAL` — cost the algorithm before any QEC choice

The logical tier is valid with no device and no code at all — estimate the
algorithm while it is still portable intent (`examples/01_p0_bell.py`):

```python
import cudaq.logical as ql

@ql.program
def bell() -> tuple[bool, bool]:
    q = ql.allocate(2, state=ql.types.zero)
    q[0] = ql.h(q[0])
    q[0], q[1] = ql.cx(q[0], q[1])
    return ql.measure_z(q[0]), ql.measure_z(q[1])

profile = ql.estimate(ql.compile(bell), tier=ql.estimate.Tier.LOGICAL)
assert profile.logical_qubits_peak == 2
assert profile.actions == {"qlx_standard_h": 1, "qlx_standard_cx": 1}
assert profile.instruments == {
    "qlx_standard_prepare_zero": 2,
    "qlx_standard_measure_z": 2,
}
```

Non-Clifford standard actions (T, T†, CCZ) are counted in `actions` like
everything else and _additionally_ in `profile.synthesis_demand`, so the demand
a Clifford+T synthesis pass will have to meet is visible before any gate-set
commitment (exercised in `python/tests/cudaq/logical/test_quake_import.py`).

## `Tier.STATIC` — count one selected P2 realization

Once codes, gadgets, and protocols are selected, the static tier walks the
executable Fabric closure and counts what would actually run. Example 03's
Steane terminal-memory gadget (`examples/03_code_and_gadget.py`):

<!--
% invisible-code-block: python
%
% gadget_build = load_ql_example(
% "preview/logical/examples/03_code_and_gadget.py", "gadget")
-->

```python
counts = ql.estimate(gadget_build, tier=ql.estimate.Tier.STATIC)
assert counts.patches_peak == 1
assert counts.build_root == "steane_memory"
assert counts.source_stage == "p2"
counts.operation_counts
# {'reset': 2, 'h': 2, 'cx': 2, 'read_syndrome_ancillas': 1, 'mz': 1, 'dealloc': 1}
```

The syndrome-extraction gadget has been lowered to its physical primitives, so
the counts are the reset/`H`/`CX`/measurement work of the actual circuit — not
the one-line `ql.extract_syndrome` the author wrote.

Protocols compose gadgets with resources and postselection, and the static tier
keeps the bookkeeping visible. Example 04's 15-to-1 distillation is estimated
straight from the authoring definition:

<!--
% invisible-code-block: python
%
% distill_15to1 = load_ql_example(
% "preview/logical/examples/04_distillation.py", "distill_15to1")
-->

```python
counts = ql.estimate(distill_15to1, tier=ql.estimate.Tier.STATIC)
assert counts.resource_requests == {"raw_t_state": 15}
assert counts.success_count == 4
assert counts.operation_counts["pack_resource"] == 1
```

`success_count` records how many zero-on-accept checks guard the result, and
resource requests count everything the protocol asks for — including attempts
that postselection may discard. Selection is never averaged away silently.

## The rules that keep estimates honest

- **Folding.** Loops and multiplicities stay symbolic in the IR; the estimator
  applies a `fabric.repeat` trip count or a call multiplicity as a numeric
  multiplier instead of unrolling. A long memory experiment costs the same to
  estimate as a short one.
- **Provenance.** Every estimate records the `build_root` and `build_sha256` it
  refined, plus the source stage and facets. Estimators derive counts from a
  freshly re-verified replay of the build's frozen snapshot — never from the
  mutable module view exposed for inspection.
- **Assumptions in the open.** The native estimation passes attach an explicit
  `assumptions` list and an `evidence` citation to every emitted
  `qlx.estimate_result` (for example,
  `"dynamic branches are counted as a static upper bound"`), so an estimate
  states next to its numbers what it assumed and what it counted.
- **Fail-closed gating.** `Tier.LOGICAL` requires a P0 build and `Tier.STATIC` a
  selected P2 build whose module passes native verification. A stage/tier
  mismatch or an unverifiable module is a typed error, never a partially
  computed number.
- **Declared assumptions.** Where a fact is evidence rather than algebra — a
  code distance, say — the typed evidence constructors ask for a method and a
  provenance, and `ql.analysis` provides the provenance spellings:
  `citation(...)` for a published source, `report(...)` and `computation(...)`
  for internal analyses and recorded tool runs, and `user_assertion(...)` for an
  explicit, unproved statement.

## Direct spellings

`ql.analysis.logical_counts(p0)` and `ql.analysis.count(build)` are the per-tier
function forms of the same estimators. They remain useful when code
intentionally selects one specialized analysis; product flows should prefer the
unified `ql.estimate(...)` front door.

## Sweeps and reproducibility

Design-space sweeps are first-class artifacts:
`ql.compiler.compile_many(points, pipeline=...)` turns a tuple of
`ql.compiler.Experiment` values into an immutable, self-describing
`ExperimentBundle` whose serialized form replays every build bit-identically in
a clean process (`python/tests/cudaq/logical/test_experiments.py`):

```python
import cudaq.logical as ql

@ql.program
def memory() -> bool:
    q = ql.prepare_zero()
    q = ql.idle(q, rounds=3)
    return ql.measure_z(q)

points = tuple(
    ql.compiler.Experiment(root=memory, parameters={"p": p})
    for p in (1e-4, 1e-3, 1e-2))
bundle = ql.compiler.compile_many(
    points, pipeline=ql.compiler.pipelines.logical())
assert len(bundle) == 3
replayed = ql.compiler.ExperimentBundle.replay(bundle.serialize())
```

A serialized bundle needs no ambient Python state, so an estimate quoted in a
paper can be re-derived from the bundle alone.

## Analytical projections live in the open

Physical-qubit, runtime, and retry-risk numbers are _not_ a hidden estimation
tier. They are explicit arithmetic, written in the example or library code where
every assumption is visible and editable, on top of a P0-backed logical profile.
`examples/06_gidney_ekera.py` is the reference workout: a windowed-arithmetic
RSA-2048 resource kernel whose folded logical profile feeds the published
design-point equations, with the same calculation available through
`ql.algorithms.estimate_gidney_ekera`. `examples/07_fermi_hubbard.py` applies
the same pattern to a Trotterized Fermi–Hubbard evolution.

## Estimating ordinary CUDA-Q kernels

CUDA-Q kernels compile through the same stages when a `cudaq.logical` target is
selected (`examples/00_cudaq_logical_resource_estimate.py`,
`examples/05_clifford_t.py`). `cudaq.estimate(kernel)` then returns the per-tier
results as CUDA-Q annotations, and the typed views are rehydrated directly from
them:

<!--
% invisible-code-block: python
%
% import cudaq
% kernel = load_ql_example(
% "preview/logical/examples/00_cudaq_logical_resource_estimate.py",
% "logical_zero_readout")
-->

```python
from cudaq.logical.estimate import FabricCounts, LogicalEstimate

estimates = cudaq.estimate(kernel)
static = FabricCounts.from_annotations(estimates.annotations)
logical = LogicalEstimate.from_annotations(estimates.annotations)
assert static.patches_peak == 1
assert logical.logical_qubits_peak == 1
```
