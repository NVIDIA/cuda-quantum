# Architecture

This advanced reference is for contributors and users inspecting dialects,
passes, or intermediate representations. For the Python interface, start with
[Getting started](../getting-started/quickstart.md) and the
[use cases](../use-cases/define-a-code.md).

CUDA-Q Logical has one user surface over three semantic stages, orthogonal
analysis facets, and one primary MLIR dialect family per stage. This page walks
the whole machine: the dialect stack, the artifact model, the ownership
contract, the compile pipeline, and the verification layers that hold it
together.

## The dialect stack

Each semantic stage owns one primary representation family, and the passes that
convert between families are the canonical compiler lowerings. Analysis facets
are ops and results inside the owning representation, not extra semantic stages.

```{figure} ../_static/figures/dialect-map.svg
:alt: The three stage dialects with their responsibilities, the passes that lower between them, and the target-emitter boundary.
:width: 100%

One primary representation family per stage. The named passes on the arrows
are the canonical lowerings; the green strip is the emitter boundary.
```

The division of labor is strict. `qlx` (P0) may not mention a machine or a code.
`lvm` (P1) binds values to machine spaces and slots but does not choose codes.
`fabric` (P2) is the QEC semantic hub: codes, encodings, verified gadgets,
protocols, and the static resource evidence derived from them.

## The semantic spine and its facets

Stages describe semantic commitment: a later stage must discharge the intent its
earlier stages left open, and each transition produces immutable evidence
instead of silently filling in missing physics.

```{figure} ../_static/figures/semantic-spine.svg
:alt: The P0, P1, and P2 stage boxes with their facets and the consumers attached orthogonally.
:width: 100%

The semantic spine. Facets (pink) attach to immutable stage roots; consumers
(green) sit orthogonal to the spine and read the verified stage they need.
```

Four facets attach to stage roots (`cudaq.logical.stages.Facet`): `QEC_SPEC`
(materialized code definitions), `QEC_REALIZATION` (verified gadget
realizations), `PROTOCOL_NETWORK` (a linked, closed protocol call graph), and
`PATCH_GRAPH` (the P2 patch-interaction view, below). Estimate results attach
the same way: typed, schema-versioned records naming their producer, their root,
and their evidence.

## Definitions versus builds

Decorators create reusable definitions:

- `@ql.program` defines an executable application kernel;
- `@ql.objective` defines reusable ideal behavior to claim against;
- `@ql.machine` defines logical spaces, capabilities, and capacity;
- `@ql.code` defines a QEC code as validated data;
- `@ql.gadget` defines one encoded realization of an objective; and
- `@ql.protocol` composes resources, rotations, and postselection into a
  reusable protocol.

(`ql` above is the shipped alias: `import cudaq.logical as ql`.)

`ql.compile` traces a definition into an immutable `Build`; `ql.compiler.place`
continues a P0 build into a placed P1 build without retracing Python. Every
build serializes and replays exactly: `Build.replay(build.serialize())`
reproduces it, witness included.

## The artifact model

Three clusters of artifacts carry the system's semantics.

**The QEC algebra.** A `@ql.code` is authored as data — block shape, stabilizer
checks, logical operators, distance — validated at construction and materialized
into the `fabric` dialect on demand. Reusable families live in
`cudaq.logical.codes` (Steane, rotated surface, repetition, Reed–Muller 15, bare
qubit); `ql.codes.BareQubit` is the honest no-protection boundary used by
protocol factories.

**The gadget stack.** A gadget claims an objective through `implements=` and is
realized by an ordinary authored body over typed patches. Matching compares the
derived Clifford action of the body against the claimed objective — a typed
contract, not a naming convention. Compilation materializes typed gadget records
(`fabric-materialize-record-schemas`) before verification.

**Machines, placement, and builds.** A `@ql.machine` declares regions with
capabilities and capacity; placement constraints such as
`ql.architecture.colocate(...)` guide the solver; the result is a placement
witness recorded on the P1 build (`placement_witness_sha256` in the IR). Builds
are immutable: continuing one never mutates it, and every estimate or emission
reads a private view.

## Data ownership

Logical qubits, encoded patches, and protocol resources are linear owners.
Operations consume the incoming owner and return its successor; measurement or
an explicit `ql.discard` ends ownership. The IR verifiers reject duplication,
stale reuse, and mismatched branch carries — there is no in-place mutation
anywhere in the IR.

```{figure} ../_static/figures/patch-lifecycle.svg
:alt: State machine of a patch's linear ownership from allocation through active execution to terminal readout.
:width: 100%

The linear life of a patch: every arrow consumes the incoming owner and
produces a successor.
```

## The compile pipeline

`ql.compiler.pipelines.*` presets name the canonical pass sequences. Passes
declare the facets they require and provide, so an out-of-order pipeline fails
verification instead of guessing:

| Preset                    | Passes                                                                                                                                | Output                                       |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| `logical()`               | `qlx-normalize-actions`, `qlx-infer-requirements`, `qlx-verify-p0`                                                                    | verified P0                                  |
| `placed()`                | `qlx-verify-p0`, `qlx-place`, `qlx-to-lvm`, `lvm-verify-p1`                                                                           | verified P1 + placement witness              |
| `qec()`                   | `lvm-select`, `lvm-apply-qec-lowerings`, `lvm-to-fabric`, `fabric-materialize-default-encodings`, `fabric-verify-generated-protocols` | verified P2 (realization + protocol network) |
| `clifford_t(precision=…)` | `qlx-synthesize-rotations`, `qlx-verify-clifford-t`                                                                                   | P0 legalized to positive H/S/T/CX            |
| `pbc()`                   | `qlx-to-pbc`, `qlx-verify-pbc`, `qlx-verify-p0`                                                                                       | P0 in Pauli-based-computation form           |

Definition-level recipes verify the other facets: `qec_definitions()`
(`fabric-verify-p2s`, provides `QEC_SPEC`), `gadgets()` (`fabric-verify-p2a`,
requires `QEC_SPEC`, provides `QEC_REALIZATION`), and `protocols()`
(`fabric-link-calls`, `fabric-verify-p2n`, provides `PROTOCOL_NETWORK`).
`device_stack(profile)` verifies an immutable device prefix against the `p1` or
`p2` profile.

Two estimation passes read a stage without leaving it: `qlx-estimate-logical`
attaches a logical profile to a P0 root (`Tier.LOGICAL`), and `fabric-count`
attaches static fabric counts to a selected P2 root (`Tier.STATIC`). Presets are
ordinary `Pipeline` values: `insert_after`, `configure`, `replace`, `append`,
and `remove` compose them without string surgery.

## The P2 patch graph

Topology survives as _inspectable evidence_, not as a physical-lowering stage. A
selected P2 build exposes `build.patch_graph`: a typed, read-only view
(`PatchGraphView`) of the patch instances and logical interactions the
realization implies, derived from canonical `fabric` IR facts. The view converts
to NetworkX or renders to PNG when the optional packages are present. Placement
stays code-agnostic at P1, and the interaction structure that a placement
implies becomes checkable at P2 — that is the whole of the topology story in the
trimmed product.

## Verification layers

Correctness is enforced at three independent layers:

- **Construction**: a code definition is validated as it is authored —
  stabilizer shape, logical operators, and distance evidence must agree before a
  `@ql.code` exists at all.
- **IR verifiers (C++)**: every stage boundary has a verify pass —
  `qlx-verify-p0`, `lvm-verify-p1`, `fabric-verify-p2s` / `-p2a` / `-p2n`,
  `fabric-verify-machine`, plus the gate-set verifiers `qlx-verify-clifford-t`
  and `qlx-verify-pbc`. Missing or inconsistent evidence fails closed. A Python
  linear-use analysis (`compiler/linearity.py`) and strict link completeness
  (`compiler/link_check.py`) back them on the authoring side.
- **Semantic verification**: a gadget's Clifford action is derived from its body
  and compared against the claimed objective
  (`cudaq.logical.gadgets.analysis.clifford_action`); names never select
  semantics.

## Typed inspection

Builds are inspected through typed APIs, not text scraping: `build.stage`,
`build.placement`, `build.definitions`, `build.calls(symbol)`,
`build.protocol_for(objective)`, `build.status`, `build.synthesis`,
`build.patch_graph`, `build.to_mlir()`, `build.content_sha256`, and
`build.serialize()` / `Build.replay(...)`. Estimate results are typed the same
way: `LogicalEstimate`, `FabricCounts`, and `LogicalProfile` read
`cudaq.estimate` annotations back as structured values.

## Package map

```text
cudaq/logical/
  programs/      programs, objectives, selection intents, typed references
  ops/           canonical executable authoring operations
  types/         canonical values, references, annotations, traced proxies
  algebra/       exact angles, Pauli algebra, Clifford actions, GF(2) values
  codes/         code definitions, encodings, profiles, blocks, families
  gadgets/       gadget definitions, typed records, verification, factories
  protocols/     protocol definitions, builders, resource protocols
  architecture/  logical placement and QEC architecture
  devices/       layered devices, resources, regions, reusable local recipes
  compiler/      pipelines, immutable builds, placement, synthesis, topology
  estimate/      the two-tier estimation surface (LOGICAL / STATIC)
  experiments/   immutable, reproducible compilation experiments
  algorithms/    algorithm libraries (the Gidney–Ekerå study)
  analysis/      resource-estimation and evidence APIs
  lower/         target lowering and Stim emission
  targets/       built-in CUDA-Q targets (estimator, clifford_t, surface)
  std/           standard resource kinds and objectives (T_STATE, ...)
  qec/           QEC implementation-lowering contracts
  dialects/      generated MLIR Python bindings (qlx, lvm, fabric)
  stages.py      stage and facet enums
```

## Extensibility

The same typed values the decorator surface produces are writable through the
lower-level builders. Pipelines are composable values, gate sets are declared as
data (`GateSet`: actions plus legalization passes), and new CUDA-Q targets wrap
a backend with `Target.from_backend(...)`. Device- and code-specific libraries
are ordinary Python modules, not global registries.

## Where to go next

- The [core concepts](../getting-started/concepts.md) explain the stage,
  ownership, and evidence model this architecture implements.
- The [use cases](../use-cases/define-a-code.md) apply the Python interface to
  codes, placement, synthesis, estimation, and emission.
