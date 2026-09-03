# Core concepts

Everything in CUDA-Q Logical follows from a small set of ideas. Internalize
these eight and the rest of the system becomes predictable.

## 1. Three stages, one direction

Executable intent is refined through exactly three semantic stages
(`ql.stages.Stage`):

| Stage                   | Question it answers                                              | Primary IR family |
| ----------------------- | ---------------------------------------------------------------- | ----------------- |
| **P0** unplaced logical | What logical computation is requested?                           | `qlx`             |
| **P1** placed logical   | Where may each logical owner reside on a logical machine?        | `lvm`             |
| **P2** QEC realization  | Which code, gadgets, and protocols realize it, and at what cost? | `fabric`          |

A later stage only adds realization facts; it never silently reinterprets the
requested logical behavior. Every stage root is verified and immutable, so a
failed lowering returns to the retained earlier root rather than repairing in
place. `ql.compile` produces the P0 root of a `@ql.program`, `ql.compiler.place`
refines it to a code-agnostic P1 placement, and selecting codes, gadgets, and
protocols produces P2. Stim text emission consumes a P2 build; it is an
interchange product, not a stage.

## 2. Facets, not extra stages

QEC specifications, gadget realizations, protocol networks, and patch graphs are
**facets** (`ql.stages.Facet`: `QEC_SPEC`, `QEC_REALIZATION`,
`PROTOCOL_NETWORK`, `PATCH_GRAPH`): independently verified facts attached to an
immutable stage root, not additional points in the P0–P2 lowering order. Several
facets coexist on one root without recompiling the program. Every pipeline pass
declares the facets it requires, provides, and invalidates, and a facet survives
a pass unless that pass explicitly invalidates or recomputes it. A compiled
gadget build, for example, is a P2 root carrying exactly `QEC_SPEC` and
`QEC_REALIZATION` — see `build.facets` in concept 5.

## 3. Linear ownership

Quantum values are _linear_: one live owner, consumed and re-produced by every
operation.

```python
import cudaq.logical as ql

@ql.program
def ownership() -> tuple[bool, bool]:
    q = ql.allocate(1, state=ql.types.zero)
    q[0] = ql.h(q[0])                    # consume q[0], produce its successor
    q[0], z = ql.mpp(ql.types.Z(q[0]))  # nondestructive: the owner survives
    return z, ql.measure_z(q[0])         # destructive: q[0] is gone
```

Rebinding (`q[0] = ql.h(q[0])`) is the visible spelling of that contract. Using
a consumed value raises `ql.errors.UseAfterConsume` at trace time, and the
canonical IR carries an independent linear-use verification: every linear SSA
value must have exactly one owner along every execution path, so double
consumption, use-after-measure, and leaks are typed failures, not runtime
surprises.

## 4. Codes, profiles, encodings — three different things

- A **`Code`** is validated algebra: physical width `n`, logical width `k`, and
  independent stabilizer and logical operator bases, checked at construction.
  Distance is _evidence_, held as a `ql.codes.Distance`: a bare integer
  normalizes to `claimed` — a recorded assertion, never a proof — while the
  evidence-bearing constructors (`exact`, `lower_bound`, `upper_bound`,
  `circuit`) require a method and provenance.
- A **`CodeProfile`** is an analysis convention over one code: effective
  syndrome generators, meta-checks, and derived boundary maps. Changing the
  convention makes a new profile, not a new code.
- An **`Encoding`** is a reusable logical view: named logical ports, a block ABI
  name, and layout facts. Preparation and conversion are _gadgets_ — an encoding
  never executes circuits.

Every code synthesizes a default profile and encoding; you author one only for a
genuinely different view. The catalog (`ql.codes`) ships `Steane`, `Repetition`,
`rotated_surface(distance)`, `ReedMuller15`, and `BareQubit`:

```python
import cudaq.logical as ql

steane = ql.codes.Steane                      # the [[7,1,3]] CSS code
assert (steane.n, steane.k) == (7, 1)
assert (steane.d.value, steane.d.status) == (3, "claimed")
```

The two structures most worth keeping apart, side by side:

| Structure  | Holds                                                                                                                                                                  |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Code`     | physical width `n`, logical width `k`, `Distance` evidence, independent stabilizers, logical X/Z pairs; CSS constructors are adapters into the same normalized algebra |
| `Encoding` | the protected `code`, a selected `profile`, a `name`, the `block` ABI name, ordered `logical_ports`, and layout facts; sealed after validation                         |

## 5. Objectives, gadgets, protocols — claims and proofs

A **gadget** is a realization — a typed circuit over encoded patches — plus a
claim: `implements=<objective>`. The claim is checked. The compiler derives the
gadget's signed symplectic action, matches it against the objective's Clifford
action, and records the equivalence evidence — or fails with a typed diagnostic:

```python
import cudaq.logical as ql

@ql.objective
def terminal_memory(q: ql.types.logical_qubit) -> None:
    ql.discard(q)

@ql.gadget(implements=terminal_memory)
def steane_memory(block: ql.patch[ql.codes.Steane]) -> None:
    block, _ = ql.extract_syndrome(block)
    block, _ = ql.mz(block.data)
    ql.discard(block)

build = ql.compile(steane_memory)
assert build.stage == ql.stages.P2
assert build.facets == (ql.stages.Facet.QEC_SPEC, ql.stages.Facet.QEC_REALIZATION)
```

When several operand-to-port embeddings verify, the compiler refuses to pick one
silently (`ql.errors.AmbiguousLogicalPortMap`); an explicit `logical_ports=`
mapping is a constraint the verifier checks, never evidence it trusts.

A **protocol** composes gadget calls with operational policy: resource requests,
postselection, and bounded retry with explicit commit points (the 15-to-1
distillation of `examples/04_distillation.py` is the shipped workout). Retry
policy is normalized at construction into immutable, type-checked structures:

| Type          | Fields                                                                                                                     |
| ------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `RetryPolicy` | positive `max_attempts`; a `RetryExhaustion` — `REPORT_FAILURE`, `ABORT`, or `RETURN_LAST`; an optional typed commit point |
| `CommitPoint` | `before_output(endpoint)` or `before_resource_output()`: the boundary before which an attempt may be replayed safely       |

## 6. Evidence follows the program

Every semantic transition emits evidence records — `pass`, `fail`, or
`unresolved` — or an explicit missing-evidence declaration. `build.evidence`
lists them and `build.status` summarizes the build (root, stage, facets, and
counts by result), as in the gadget build above. `build.serialize()` captures
the whole build — selected definitions, evidence, and all — and
`ql.compiler.Build.replay` reopens it in a clean process with no ambient Python
state. The same honesty applies to distances and estimates: a `claimed` distance
stays a claim, and each estimation tier reports exactly which facts it consumed.

## 7. There is no registry — imports are the linker

CUDA-Q Logical deliberately has no global implementation table, because a
registry makes _import order_ part of your program's semantics: two sessions
that import modules in a different order could select different physics, and a
serialized build could not say what was visible when it was compiled.

Instead, discovery is scoped and explicit. Each compilation collects candidates
from exactly three places:

1. definitions bound at module scope in **your program's module**;
2. definitions bound at module scope in **your device's module**;
3. definitions exported by a **`cudaq.logical` library submodule you explicitly
   imported** into one of those modules — the import statement is the link act,
   and bare `import cudaq.logical` links nothing.

The candidate set is derived fresh inside each compilation, filtered by
objective and boundary types, and the _selected_ closure is captured into the
build: `build.source_modules` and `build.definitions` record exactly what was
visible and what won. The consequences you feel day-to-day:

- a gadget defined in a helper file you never imported is invisible — the
  failure is a typed "no feasible P2 implementation", not a mystery winner;
- deleting an import genuinely detaches its implementations;
- replaying a build needs no ambient Python state at all.

## 8. Global phase is not observable

**CUDA-Q Logical treats states and operators as projective: an overall phase is
neither observable nor tracked.** The only observables are measurement outcomes,
and those are invariant under an overall phase. Concretely:

- **Synthesis** targets a projective operator-norm bound
  (`ql.compiler.synthesize(gate_set=..., precision=...)`), so `R_Z(kπ/4) = T^k`
  holds up to phase.
- **Rotations are 4π-periodic exactly and 2π-periodic up to phase.**
  `ql.algebra.Angle` keeps angles as exact rational multiples of π so this stays
  precise — and the authored angle is preserved, never auto-reduced:

```python
import cudaq.logical as ql

assert ql.algebra.Angle(9, 4).pi_fraction == (9, 4)
assert float(ql.algebra.Angle(9, 4) - ql.algebra.Angle(1, 4)) == float(2 * ql.algebra.pi)
```

The boundary: dropping global phase is safe for a linear, classically
conditioned program measured at the end. CUDA-Q Logical's conditionals are
classical (measurement-conditioned), so no shipped surface needs to track phase;
introducing quantum-controlled arbitrary unitaries would change that and is out
of scope.

## Where the pieces live

| You write                          | You get                                          | Canonical home           |
| ---------------------------------- | ------------------------------------------------ | ------------------------ |
| `@ql.program`                      | portable P0 logical program                      | your module              |
| `@ql.machine`                      | logical machine for P1 placement                 | `ql.architecture`, yours |
| `@ql.code`                         | validated `Code` + default profile/encoding      | `ql.codes`, yours        |
| `@ql.gadget` / `@ql.protocol`      | verified realization / composition               | yours, `ql.protocols`    |
| `ql.compile` / `ql.compiler.place` | immutable, replayable `Build` roots              | `ql.compiler`            |
| `ql.estimate(..., tier=...)`       | `Tier.LOGICAL` (P0) or `Tier.STATIC` (P2 counts) | `ql.estimate`            |
| `ql.emit` / `ql.targets`           | Stim circuit text from a P2 build                | `ql.targets`             |

Naming follows PEP 8 throughout — artifact classes are camel case (`Code`,
`Encoding`), while operations, decorators, and constants are snake*case
(`@ql.machine`, `ql.extract_syndrome`). One deliberate near-collision to know
about: `ql.types.X(q)` constructs a Pauli \_factor* for products, while
`ql.x(q)` applies the gate.

## Where to go next

- The [architecture reference](../reference/architecture.md) describes the
  dialect stack and pipeline presets that realize these concepts.
- The [use cases](../use-cases/define-a-code.md) put them to work task by task.
- [Examples](../use-cases/examples/index.md) shows them in runnable, shipped
  code.
