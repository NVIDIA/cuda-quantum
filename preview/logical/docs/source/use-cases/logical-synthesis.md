# Logical Clifford+T synthesis

Use `ql.compiler.synthesize` when you want a device-independent logical program
expressed in a specific gate set. The result is an ordinary immutable P0
`Build`: inspect it, serialize it, or pass it to placement and QEC compilation
later.

## Synthesize a program

The supported gate set is positive-generator Clifford+T:

```python
import cudaq.logical as ql


# Author the rotation at portable P0, before any code or machine is selected.
@ql.program
def ansatz(theta: float) -> bool:
    q = ql.allocate(2, state=ql.types.zero)
    q[0], q[1] = ql.ops.rotate(
        -(ql.types.X(q[0]) @ ql.types.Z(q[1])),
        angle=theta,
    )
    ql.discard(q[1])
    return ql.measure_z(q[0])


# Legalize only the logical unitary actions into the requested gate set.
clifford_t = ql.compiler.synthesize(
    ansatz,
    gate_set=ql.compiler.gate_sets.clifford_t,
    parameters={"theta": ql.algebra.pi / 4},
    precision=1e-10,
)

assert clifford_t.stage == ql.stages.P0
assert "pauli_rotation" not in clifford_t.to_mlir()
print(clifford_t.synthesis.t_count)
print(clifford_t.synthesis.clifford_count)
```

No `device=` is involved. Preparation, measurement, discard, structured
classical control, source locations, and linear ownership remain P0 semantics;
only logical unitary actions are legalized.

`ql.compiler.gate_sets.clifford_t` emits:

- `H`;
- positive `S`;
- positive `T`; and
- `CX`.

Inverse phase gates use positive generators. Pauli and Clifford actions, CZ, and
CCZ are decomposed exactly. Static multi-qubit Pauli rotations use local basis
changes, a CX parity ladder, one synthesized Z rotation, and uncomputation. The
legalization is performed by the shared CUDA-Q cudaq-synth/gridsynth
implementation — recorded on the build's `logical_gate_set_legalization`
evidence record — and a final independent verifier (`qlx-verify-clifford-t`)
rejects any logical action left outside the requested gate set.

## Precision

`precision` is the default projective operator-norm bound for each off-lattice
rotation:

```python
result = ql.compiler.synthesize(
    ansatz,
    gate_set=ql.compiler.gate_sets.clifford_t,
    parameters={"theta": 0.3},
    precision=1e-8,
)
```

An explicitly authored rotation precision takes precedence:

```python
@ql.program
def rotation_precision() -> bool:
    q = ql.allocate(1, state=ql.types.zero)
    q[0], = ql.ops.rotate(
        ql.types.Z(q[0]),
        angle=0.3,
        precision=1e-12,
    )
    return ql.measure_z(q[0])

ql.compile(rotation_precision)
```

Exact rational multiples written with `ql.algebra.pi` retain exact source
metadata and take the exact-word fast path (see
[Magic states and protocols](magic-states-and-protocols.md)). For float-authored
angles, an exact lattice word is used only when that word meets the requested
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
replayed = ql.compiler.Build.replay(payload)
assert replayed.synthesis == summary
```

The `Build` evidence records the gate-set obligation, positive-generator
convention, precision policy, and projective error metric. The resolved
legalization pass recipe is serialized with the `Build`, so a replayed build
carries the same summary.

## Use the pipeline form

The convenience API executes the pipeline owned by the gate-set value.
Compiler-oriented code can request the same pipeline directly:

```python
equivalent = ql.compile(
    ansatz,
    pipeline=ql.compiler.pipelines.clifford_t(precision=1e-10),
    parameters={"theta": ql.algebra.pi / 4},
)

assert equivalent.to_mlir() == clifford_t.to_mlir()
```

This is a real P0-to-P0 transformation. Passing an existing P0 `Build` creates a
new `Build` and leaves the source unchanged:

```python
source = ql.compile(
    ansatz,
    parameters={"theta": ql.algebra.pi / 4},
)
legalized = ql.compiler.synthesize(
    source,
    gate_set=ql.compiler.gate_sets.clifford_t,
)
```

## Pauli-based computation, still at P0

An estimation flow that consumes Pauli-based computation can insert one more
device-free P0 transform after synthesis:

```python
pbc = ql.compile(
    clifford_t,
    pipeline=ql.compiler.pipelines.pbc(),
)

assert pbc.stage == ql.stages.P0
assert "#qlx.action<h>" not in pbc.to_mlir()
assert "#qlx.action<t>" not in pbc.to_mlir()
assert "#qlx.action<pauli_rotation>" in pbc.to_mlir()
```

The `qlx-to-pbc` pass absorbs the Clifford frame and returns an immutable P0
build containing signed π/4 Pauli-product rotations and pairwise-commuting
terminal product measurements; `qlx-verify-pbc` then checks that contract.
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
- [Examples](examples.md) — example 05 estimates a CUDA-Q rotation in this gate
  set end to end.
