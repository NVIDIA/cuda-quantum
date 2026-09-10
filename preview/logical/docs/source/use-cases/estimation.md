# Resource estimation

Estimation in CUDA-Q Logical comes in four explicit tiers over the _same_
linked definitions. One call shape serves all four:
`ql.estimate(value, tier=...)`. You can pass a `Build`, schedule, or authoring
definition. A definition is first compiled through its normal default pipeline;
the estimator then checks that the resulting stage matches the requested tier
and fails with a typed diagnostic when they disagree.

| Tier                          | Needs                                                     | Returns                                                                                                                                                         |
| ----------------------------- | --------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ql.estimate.Tier.LOGICAL`    | a verified P0 program — no device, no code                | `ql.estimate.LogicalProfile`: action/instrument totals, peak logical qubits, idle/discard counts, action-depth upper bound, and synthesis demand                 |
| `ql.estimate.Tier.STATIC`     | a selected P2 build — code, gadgets, and protocols chosen | `ql.estimate.FabricCounts`: per-operation counts, gadget/protocol call totals, resource requests, postselection bookkeeping, syndrome rounds, and peak patches  |
| `ql.estimate.Tier.ANALYTICAL` | a selected P2 build and physical operating assumptions    | `ql.estimate.FabricEstimate`: modeled error, acceptance, wall-clock cost, physical-qubit peak, assumptions, and retry demand                                    |
| `ql.estimate.Tier.SCHEDULE`   | a scheduled P3 physical graph                             | `ql.estimate.ScheduleEstimate`: authenticated event counts, makespan, resource-time, physical resources, utilization, and termination behavior                  |

`Tier.STATIC` is the default, so `ql.estimate(p2_build)` needs no `tier=`
argument. Results are immutable typed values, expose plain-data projections
where applicable, and retain the source identity and evidence used to derive
them.

## `Tier.LOGICAL` — cost the algorithm before any QEC choice

The logical tier works with no device and no code at all. You can estimate the
algorithm while it is still portable intent
(`examples/standalone/00_logical_program.py`):

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

Non-Clifford standard actions (T, T†, CCZ) show up in `actions` like
everything else, and _additionally_ in `profile.synthesis_demand`. That way you
can see the demand a Clifford+T synthesis pass will have to meet before you
commit to any gate set (exercised in
`python/tests/cudaq/logical/test_quake_import.py`).

## `Tier.STATIC` — count one selected P2 realization

Once you have selected codes, gadgets, and protocols, the static tier walks
the executable Fabric closure and counts what would actually run. Here is
the standalone Steane terminal-memory gadget
(`examples/standalone/02_code_and_gadget.py`):

<!--
% invisible-code-block: python
%
% gadget_build = load_ql_example(
% "preview/logical/examples/standalone/02_code_and_gadget.py", "gadget")
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
the counts report the reset/`H`/`CX`/measurement work of the actual circuit —
not the one-line `ql.extract_syndrome` the author wrote.

Protocols compose gadgets with resources and postselection, and the static
tier keeps the bookkeeping visible. You can estimate the standalone 15-to-1
distillation example straight from its authoring definition:

<!--
% invisible-code-block: python
%
% distill_15to1 = load_ql_example(
% "preview/logical/examples/standalone/03_magic_state_distillation.py",
% "distill_15to1")
-->

```python
counts = ql.estimate(distill_15to1, tier=ql.estimate.Tier.STATIC)
assert counts.operation_counts["resource_request"] == 15
assert counts.operation_counts["selection"] == 4
assert counts.operation_counts["pack_resource"] == 1
```

The operation counts record how many zero-on-accept selection checks guard the
result, and resource-request operations count everything the protocol asks for
— including attempts that postselection may discard. Selection is never
averaged away silently.

## `Tier.ANALYTICAL` — apply an explicit physical model

The analytical tier starts from a selected P2 operation network and combines
its static counts with a physical error rate, failure budget, scaling law, and
cycle time. A device operating point can supply these values, and callers can
override them for sensitivity studies. The result keeps its assumptions and
reports whether the selected distance meets the requested budget.

`examples/02_surface_code_resource_estimate.py` uses this tier through a CUDA-Q
target, while `examples/standalone/05_gidney_ekera_lookup_addition.py` invokes
it directly on a P2 build.

## `Tier.SCHEDULE` — cost the physical event graph

The schedule tier consumes a scheduled P3 graph. It reports authenticated
physical-resource and event counts, first-attempt/expected/maximum makespans,
resource-time, utilization, exhaustion probability, and the selected
termination policy. `examples/standalone/04_physical_schedule.py` shows the
smallest direct workflow; the surface-code and Carbon CUDA-Q targets expose the
same tier through `cudaq.estimate` annotations.

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
- **Fail-closed gating.** `Tier.LOGICAL` requires P0, `Tier.STATIC` and
  `Tier.ANALYTICAL` require a selected P2 build, and `Tier.SCHEDULE` requires a
  scheduled P3 graph. A stage/tier mismatch, missing operating assumption, or
  unverifiable module is a typed error, never a partially computed number.
- **Declared assumptions.** Where a fact is evidence rather than algebra — a
  code distance, say — the typed evidence constructors ask for a method and a
  provenance, and `ql.analysis` provides the provenance spellings:
  `citation(...)` for a published source, `report(...)` and `computation(...)`
  for internal analyses and recorded tool runs, and `user_assertion(...)` for an
  explicit, unproved statement.

## Direct spellings

`ql.analysis.logical_counts(p0)`, `ql.analysis.count(build)`,
`ql.estimate.analytical(build, ...)`, and `ql.estimate.scheduled(schedule, ...)`
are the per-tier function forms of the same estimators. Reach for them when your
code intentionally selects one specialized analysis; product flows should
prefer the unified `ql.estimate(...)` front door.

## Sweeps and reproducibility

Design-space sweeps are first-class artifacts.
`ql.compiler.compile_many(points, pipeline=...)` turns a tuple of
`ql.compiler.Experiment` values into an immutable, self-describing
`ExperimentBundle`. Its serialized form replays every build bit-identically in
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

A serialized bundle needs no ambient Python state, so you can re-derive an
estimate quoted in a paper from the bundle alone.

## Paper-specific projections live in the open

The built-in analytical tier is not a substitute for a paper-specific system
model. `examples/05_gidney_ekera.py` compiles a windowed-arithmetic RSA-2048
resource kernel to a folded logical profile, then applies the paper's explicit
timing and layout equations. That default fast path deliberately does not call
`Tier.ANALYTICAL`; its assumptions remain visible and editable in the example.
The same example offers `--physical` as an opt-in P3 compilation and scheduling
path. The related library calculation is available through
`ql.algorithms.estimate_gidney_ekera`.

## Estimating ordinary CUDA-Q kernels

When you select a `cudaq.logical` target, ordinary CUDA-Q kernels compile
through the stages that target owns. The logical, Clifford+T, Fermi–Hubbard,
surface-code, and Carbon examples exercise progressively deeper target stacks.
`cudaq.estimate(kernel)` returns the available per-tier results as CUDA-Q
annotations, and typed views rehydrate directly from them:

<!--
% invisible-code-block: python
%
% import cudaq
% kernel, target, kernel_args = load_ql_example(
% "preview/logical/examples/02_surface_code_resource_estimate.py",
% "logical_zero_memory", "baseline_target", "baseline_qubits")
% cudaq.set_target(target)
-->

```python
from cudaq.logical.estimate import FabricCounts, LogicalEstimate

estimates = cudaq.estimate(kernel, kernel_args)
static = FabricCounts.from_annotations(estimates.annotations)
logical = LogicalEstimate.from_annotations(estimates.annotations)
assert static.patches_peak == 1
assert logical.logical_qubits_peak == 1
```
