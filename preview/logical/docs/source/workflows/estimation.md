# Resource estimation

Estimation in CUDA-Q Logical is two explicit tiers over the *same* linked
definitions: device-independent logical counts, and static counts over a
selected QEC realization. One call shape serves both —
`qlx.estimate(value, tier=...)`. The value may be a `Build` or an authoring
definition; a definition is first compiled through its normal default
pipeline, and the estimator then validates that the resulting stage matches
the requested tier, failing with a typed diagnostic when they disagree.

| Tier | Needs | Returns |
|---|---|---|
| `qlx.estimate.Tier.LOGICAL` | a verified P0 program — no device, no code | `qlx.estimate.LogicalProfile`: action/instrument totals, peak logical qubits, idle/discard counts, an action-depth upper bound, synthesis demand |
| `qlx.estimate.Tier.STATIC` | a selected P2 build — code, gadgets, and protocols chosen | `qlx.estimate.FabricCounts`: per-operation counts, gadget/protocol call totals, resource requests, postselection bookkeeping, syndrome rounds, peak live patches |

`Tier.STATIC` is the default, so `qlx.estimate(p2_build)` needs no `tier=`
argument. Both result types are immutable plain-data values with
`to_dict()` projections, and both record the `build_root` and
`build_sha256` they were derived from.

## Tier.LOGICAL — cost the algorithm before any QEC choice

The logical tier is valid with no device and no code at all — estimate the
algorithm while it is still portable intent
(`examples/01_p0_bell.py`):

```python
import cudaq.logical as qlx

@qlx.program
def bell() -> tuple[bool, bool]:
    q = qlx.allocate(2, state=qlx.types.zero)
    q[0] = qlx.h(q[0])
    q[0], q[1] = qlx.cx(q[0], q[1])
    return qlx.measure_z(q[0]), qlx.measure_z(q[1])

profile = qlx.estimate(qlx.compile(bell), tier=qlx.estimate.Tier.LOGICAL)
profile.logical_qubits_peak   # 2
profile.actions               # {'qlx_standard_h': 1, 'qlx_standard_cx': 1}
profile.instruments           # {'qlx_standard_prepare_zero': 2, 'qlx_standard_measure_z': 2}
```

Non-Clifford standard actions (T, T†, CCZ) are counted in `actions` like
everything else and *additionally* in `profile.synthesis_demand`, so the
demand a Clifford+T synthesis pass will have to meet is visible before any
gate-set commitment (exercised in
`python/tests/cudaq/logical/test_quake_import.py`).

## Tier.STATIC — count one selected P2 realization

Once codes, gadgets, and protocols are selected, the static tier walks the
executable Fabric closure and counts what would actually run. Example 03's
Steane terminal-memory gadget (`examples/03_code_and_gadget.py`):

```python
counts = qlx.estimate(gadget_build, tier=qlx.estimate.Tier.STATIC)
counts.operation_counts
# {'reset': 2, 'h': 2, 'cx': 2, 'read_syndrome_ancillas': 1, 'mz': 1, 'dealloc': 1}
counts.patches_peak       # 1 — peak simultaneously live encoded patches
counts.build_root         # 'steane_memory' — every estimate cites its build
counts.source_stage       # 'p2'
```

The syndrome-extraction gadget has been lowered to its physical primitives,
so the counts are the reset/H/CX/measurement work of the actual circuit —
not the one-line `qlx.extract_syndrome` the author wrote.

Protocols compose gadgets with resources and postselection, and the static
tier keeps the bookkeeping visible. Example 04's 15-to-1 distillation is
estimated straight from the authoring definition:

```python
counts = qlx.estimate(distill_15to1, tier=qlx.estimate.Tier.STATIC)
counts.resource_requests    # {'raw_t_state': 15}
counts.success_count        # 4 — the postselection checks that must all pass
counts.operation_counts["pack_resource"]  # 1 — one verified T state out
```

`success_count` records how many zero-on-accept checks guard the result, and
resource requests count everything the protocol asks for — including attempts
that postselection may discard. Selection is never averaged away silently.

## The rules that keep estimates honest

- **Folding.** Loops and multiplicities stay symbolic in the IR; the
  estimator applies a `fabric.repeat` trip count or a call multiplicity as a
  numeric multiplier instead of unrolling. A long memory experiment costs
  the same to estimate as a short one.
- **Provenance.** Every estimate records the `build_root` and
  `build_sha256` it refined, plus the source stage and facets. Estimators
  derive counts from a freshly re-verified replay of the build's frozen
  snapshot — never from the mutable module view exposed for inspection.
- **Assumptions in the open.** The native estimation passes attach an
  explicit `assumptions` list and an `evidence` citation to every emitted
  `qlx.estimate_result` (for example, `"dynamic branches are counted as a
  static upper bound"`), so an estimate states next to its numbers what it
  assumed and what it counted.
- **Fail-closed gating.** `Tier.LOGICAL` requires a P0 build and
  `Tier.STATIC` a selected P2 build whose module passes native
  verification. A stage/tier mismatch or an unverifiable module is a typed
  error, never a partially computed number.
- **Declared assumptions.** Where a fact is evidence rather than algebra —
  a code distance, say — the typed evidence constructors ask for a method
  and a provenance, and `qlx.analysis` provides the provenance spellings:
  `citation(...)` for a published source, `report(...)` and
  `computation(...)` for internal analyses and recorded tool runs, and
  `user_assertion(...)` for an explicit, unproved statement.

## Direct spellings

`qlx.analysis.logical_counts(p0)` and `qlx.analysis.count(build)` are the
per-tier function forms of the same estimators. They remain useful when code
intentionally selects one specialized analysis; product flows should prefer
the unified `qlx.estimate(...)` front door.

## Sweeps and reproducibility

Design-space sweeps are first-class artifacts:
`qlx.compiler.compile_many(points, pipeline=...)` turns a tuple of
`qlx.compiler.Experiment` values into an immutable, self-describing
`ExperimentBundle` whose serialized form replays every build bit-identically
in a clean process (`python/tests/cudaq/logical/test_experiments.py`):

```python
import cudaq.logical as qlx

@qlx.program
def memory() -> bool:
    q = qlx.prepare_zero()
    q = qlx.idle(q, rounds=3)
    return qlx.measure_z(q)

points = tuple(
    qlx.compiler.Experiment(root=memory, parameters={"p": p})
    for p in (1e-4, 1e-3, 1e-2))
bundle = qlx.compiler.compile_many(
    points, pipeline=qlx.compiler.pipelines.logical())
len(bundle)  # 3
replayed = qlx.compiler.ExperimentBundle.replay(bundle.serialize())
```

A serialized bundle needs no ambient Python state, so an estimate quoted in a
paper can be re-derived from the bundle alone.

## Analytical projections live in the open

Physical-qubit, runtime, and retry-risk numbers are *not* a hidden
estimation tier. They are explicit arithmetic, written in the example or
library code where every assumption is visible and editable, on top of a
P0-backed logical profile. `examples/06_gidney_ekera.py` is the reference
workout: a windowed-arithmetic RSA-2048 resource kernel whose folded logical
profile feeds the published design-point equations, with the same
calculation available through `qlx.algorithms.estimate_gidney_ekera`.
`examples/07_fermi_hubbard.py` applies the same pattern to a Trotterized
Fermi–Hubbard evolution.

## Estimating ordinary CUDA-Q kernels

CUDA-Q kernels compile through the same stages when a `cudaq.logical` target
is selected (`examples/00_cudaq_logical_resource_estimate.py`,
`examples/05_clifford_t.py`). `cudaq.estimate(kernel)` then returns the
per-tier results as CUDA-Q annotations, and the typed views are rehydrated
directly from them:

```python
from cudaq.logical.estimate import FabricCounts, LogicalEstimate

estimates = cudaq.estimate(kernel)          # with a cudaq.logical target set
static = FabricCounts.from_annotations(estimates.annotations)
logical = LogicalEstimate.from_annotations(estimates.annotations)
```

For the pass-pipeline spelling of the same two tiers — and for CI systems
that do not embed Python — see the [command-line estimation
workflow](command-line-estimation).
