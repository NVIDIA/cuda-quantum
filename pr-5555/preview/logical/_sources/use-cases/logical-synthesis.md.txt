# Logical Clifford+T synthesis

Use `cudaq.logical.compiler.synthesize` when you want a device-independent
logical program expressed in a specific gate set. The result is an ordinary
immutable P0 `Build`: inspect it, serialize it, or pass it to placement and QEC
compilation later.

## Synthesize a program

The supported gate set is positive-generator Clifford+T:

```python
import cudaq.logical as cql


# Author the rotation at portable P0, before any code or machine is selected.
@cql.program
def ansatz(theta: float) -> bool:
    q = cql.allocate(1, state=cql.types.zero)
    q[0] = cql.rz(q[0], theta)
    return cql.measure_z(q[0])


# Legalize only the logical unitary actions into the requested gate set.
clifford_t = cql.compiler.synthesize(
    ansatz,
    gate_set=cql.compiler.gate_sets.clifford_t,
    parameters={"theta": cql.algebra.pi / 4},
    precision=1e-10,
)

assert clifford_t.stage == cql.stages.P0
assert "pauli_rotation" not in clifford_t.to_mlir()
print(clifford_t.synthesis.t_count)
print(clifford_t.synthesis.clifford_count)
```

You pass no `device=`. Preparation, measurement, discard, structured
classical control, source locations, and linear ownership remain P0 semantics;
only the logical unitary actions are legalized.

`cudaq.logical.compiler.gate_sets.clifford_t` emits:

- `H`;
- positive `S`;
- positive `T`; and
- `CX`.

Inverse phase gates use positive generators. Pauli and Clifford actions, CZ,
and CCZ are decomposed exactly. Static multi-qubit Pauli rotations become local
basis changes, a `CX` parity ladder, one synthesized `Z` rotation, and
uncomputation. The shared CUDA-Q `cudaq-synth`/`gridsynth` implementation
performs the legalization, and the build records it on its
`logical_gate_set_legalization` evidence record. A final independent verifier
(`qlx-verify-clifford-t`) rejects any logical action left outside the requested
gate set.

## Precision

`precision` is the default projective operator-norm bound for each off-lattice
rotation:

```python
result = cql.compiler.synthesize(
    ansatz,
    gate_set=cql.compiler.gate_sets.clifford_t,
    parameters={"theta": 0.3},
    precision=1e-8,
)
```

An explicitly authored rotation precision takes precedence:

```python
@cql.program
def rotation_precision() -> bool:
    q = cql.allocate(1, state=cql.types.zero)
    q[0] = cql.rz(q[0], 0.3, precision=1e-12)
    return cql.measure_z(q[0])

cql.compile(rotation_precision)
```

Exact rational multiples written with `cudaq.logical.algebra.pi` retain exact
source metadata and take the exact-word fast path (see [Magic states and
protocols](magic-states-and-protocols.md)). For float-authored angles,
synthesis uses an exact lattice word only when that word meets the requested
precision.

Grid synthesis needs a static angle. Specialize runtime ABI parameters through
`parameters=` as shown above. Without specialization, synthesis fails rather
than returning a `Build` that still contains an arbitrary rotation.

## Inspect and replay the result

`Build.synthesis` reports the selected gate set, default precision, and gate
counts:

```python
summary = clifford_t.synthesis
print(summary.gate_set)
print(summary.precision)
print(summary.h_count, summary.s_count, summary.t_count, summary.cx_count)

payload = clifford_t.serialize()
replayed = cql.compiler.Build.replay(payload)
assert replayed.synthesis == summary
```

The `Build` evidence records the gate-set obligation, the positive-generator
convention, the precision policy, and the projective error metric. The `Build`
also serializes the resolved legalization pass recipe, so a replayed build
carries the same summary.

## Use the pipeline form

`cudaq.logical.compiler.synthesize` is the convenience API: it executes the
pipeline owned by the gate-set value. Compiler-oriented code can request the
same pipeline directly:

```python
equivalent = cql.compile(
    ansatz,
    pipeline=cql.compiler.pipelines.clifford_t(precision=1e-10),
    parameters={"theta": cql.algebra.pi / 4},
)

assert equivalent.to_mlir() == clifford_t.to_mlir()
```

This is a real P0-to-P0 transformation. If you pass an existing P0 `Build`,
you get a new `Build`, and the source is left unchanged:

```python
source = cql.compile(
    ansatz,
    parameters={"theta": cql.algebra.pi / 4},
)
legalized = cql.compiler.synthesize(
    source,
    gate_set=cql.compiler.gate_sets.clifford_t,
)
```

## Pauli-based computation, still at P0

An estimation flow that consumes Pauli-based computation can add one more
device-free P0 transform after synthesis:

```python
pbc = cql.compile(
    clifford_t,
    pipeline=cql.compiler.pipelines.pbc(),
)

assert pbc.stage == cql.stages.P0
assert "#qlx.action<h>" not in pbc.to_mlir()
assert "#qlx.action<t>" not in pbc.to_mlir()
assert "#qlx.action<pauli_rotation>" in pbc.to_mlir()
```

The `qlx-to-pbc` pass absorbs the Clifford frame and returns an immutable P0
build holding signed π/4 Pauli-product rotations and pairwise-commuting
terminal product measurements. `qlx-verify-pbc` then checks that contract.
Neither pass chooses a code, requests magic states, or inspects a machine.

## When not to synthesize

Do not synthesize early when the intended realization should choose a native
rotation or another non-Clifford strategy at P2 selection time — synthesis is an
early logical commitment that prices every rotation as a Clifford+T word. In
that case, compile the original P0 program instead and let selection see the
rotation.

## Continue from here

- [Magic states and protocols](magic-states-and-protocols.md) — the phase
  convention, exact angles, and where the T states come from.
- [Examples](examples/index.md) — example 01 estimates a CUDA-Q rotation in this
  gate set end to end.
