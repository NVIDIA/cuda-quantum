# Architecture

This advanced reference is for contributors and for users who inspect
dialects, passes, or intermediate representations. For the Python interface,
start with [Getting started](../getting-started/quickstart.md) and the
[use cases](../use-cases/define-a-code.md).

CUDA-Q Logical has one user surface over four implemented semantic stages and
a set of reusable MLIR dialects. A stage describes how much the compiler has
committed; it is not another name for a dialect. This page walks through the
dialect stack, artifact model, ownership contract, compile pipeline, and the
verification layers that hold them together.

## The dialect stack

Each dialect owns a coherent vocabulary. The current compiler uses `qlx`,
`lvm`, `fabric`, and `phys` as its principal P0, P1, P2, and P3 vocabularies,
and the passes between their canonical roots are the standard lowerings. This
alignment is useful, but dialect membership is not a stage test: a canonical
root reports its stage through `SemanticRootOpInterface`, and genuinely shared
operations can appear at more than one stage.

```{figure} ../_static/figures/dialect-map.svg
:alt: The four stage dialects with their responsibilities, lowerings, estimators, and emission boundaries.
:width: 100%

One primary representation family per stage. The named passes on the arrows
are the canonical lowerings; the green strip summarizes estimation and
emission products.
```

The division of labor is strict. `qlx` P0 programs may not mention a machine or
a code.
`lvm` (P1) binds values to machine spaces and slots but does not choose codes.
`fabric` (P2) is the QEC semantic hub: codes, encodings, verified gadgets,
protocols, and the static resource evidence derived from them. `phys` (P3)
binds those realizations to physical resources, native events, routes, and
schedules.

## Framework and extension boundaries

CUDA-Q Logical separates three layers whose dependencies point in one
direction:

| Layer | Public responsibility | Representative targets or packages |
| --- | --- | --- |
| Framework | Types, dialects, neutral interfaces, intrinsic verification, and reusable transformations | `MLIRCUDAQLogicalInterfaces`, `MLIRQLXDialect`, `MLIRLVMDialect`, `MLIRFabricDialect`, `MLIRPhysDialect`, `MLIRCflowDialect`, `MLIREventDialect` |
| Reference compiler | Named partial and end-to-end pipelines, pass composition, and command-line tools | transform libraries, compiler providers, `qlx-opt`, `qlx-translate` |
| Broader stack | Python authoring, devices and providers, runtime integration, examples, and evaluations | `cudaq.logical`, target modules, runtime bindings |

The broader stack may use the reference compiler, and the reference compiler
may use the framework. Framework libraries do not depend on either higher
layer. In particular, `MLIRCUDAQLogicalInterfaces` depends only on LLVM/MLIR support;
it does not include or link the QLX, LVM, Fabric, Phys, or hardware dialects.

Two interfaces establish the first common contracts. A canonical executable
root implements `SemanticRootOpInterface`, which returns a typed stage and
root kind derived from the operation class. These values are not additional
IR attributes and cannot disagree with the root. P3 measurement events
implement `PhysicalMeasurementOpInterface`, which exposes their input states,
successor states, classical record, and existing stable record identity.
`phys.measure` and `phys.measure_product` implement this contract directly.

An external physical dialect can implement the same interface without linking
the reference compiler or depending on Phys operation classes. When such an
operation carries physical linear values inside `phys.graph`, the graph
verifier accepts it through the interface only after checking that every
physical state operand and result is reported exactly once and that the record
is the operation's only physical-record result. Each successor must preserve
the physical resource of one input state; relocation requires a separate P3
event. Missing, incomplete, or contradictory interface answers are rejected.
Interface conformance says what an operation means to generic analysis. It
does not make the operation legal for a target, supply a lowering, or attach
scheduling, noise, detector, or resource-estimate data.
The exemption is limited to the reported measurement state: a foreign
operation still cannot carry a physical resource payload or linear event
handle, whether the operation is regionless or owns regions.

The C++ extension surface is currently a build-tree contract. The conformance
fixture is configured as a separate `find_package(CUDAQLogical)` project and
loaded into a fresh `qlx-opt` process. The command retains its dialect-derived
name, but the extension contract belongs to CUDA-Q Logical as a product. The
fixture is deliberately absent from production targets, installs, and wheels.
CUDA-Q Logical does not yet promise a stable installed C++ plugin ABI.

## The semantic spine and its facets

Stages describe semantic commitment: a later stage must discharge the intent its
earlier stages left open, and each transition produces immutable evidence
instead of silently filling in missing physics.

```{figure} ../_static/figures/semantic-spine.svg
:alt: The P0, P1, P2, and P3 stage boxes with their facets and consumers attached orthogonally.
:width: 100%

The semantic spine. Facets (pink) attach to immutable stage roots; consumers
(green) sit orthogonal to the spine and read the verified stage they need.
```

Facets attach independently verified facts to stage roots
(`cudaq.logical.stages.Facet`). P2 facets include `QEC_SPEC`,
`QEC_REALIZATION`, `PROTOCOL_NETWORK`, and `PATCH_GRAPH`; physical lowering and
scheduling attach their own. A build reports the ones it carries, so
`cudaq.logical.stages.PROTOCOL_NETWORK in p2.facets` is the way to ask.
Estimate results attach the same way: typed records naming their producer,
source, and evidence.

## Definitions versus builds

Decorators create reusable definitions:

- `@cudaq.logical.program` defines an executable application kernel;
- `@cudaq.logical.objective` defines reusable ideal behavior to claim against;
- `@cudaq.logical.machine` defines logical spaces, capabilities, and capacity;
- `@cudaq.logical.code` defines a QEC code as validated data;
- `@cudaq.logical.gadget` defines one encoded realization of an objective; and
- `@cudaq.logical.protocol` composes resources, rotations, and postselection
  into a reusable protocol.

(`cql` above is the shipped alias: `import cudaq.logical as cql`.)

`cudaq.logical.compile` traces a definition into an immutable `Build`;
`cudaq.logical.compiler.place` continues a P0 build into a placed P1 build
without retracing Python. Every build serializes and replays exactly:
`Build.replay(build.serialize())` reproduces it, witness included.

## The artifact model

Three clusters of artifacts carry the system's semantics.

**The QEC algebra.** You author a `@cudaq.logical.code` as data — block shape,
stabilizer checks, logical operators, distance. CUDA-Q Logical validates it at
construction and materializes it into the `fabric` dialect on demand. Reusable
families live in `cudaq.logical.codes` (Steane, rotated surface, repetition,
Reed–Muller 15, bare qubit); `cudaq.logical.codes.BareQubit` is the honest
no-protection boundary used by protocol factories.

**The gadget stack.** A gadget claims an objective through `implements=`; an
ordinary authored body over typed patches realizes it. Matching compares the
body's derived Clifford action against the claimed objective — a typed contract,
not a naming convention. Compilation materializes typed gadget records
(`fabric-materialize-record-schemas`) before verification.

**Machines, placement, and builds.** A `@cudaq.logical.machine` declares
logical regions with capabilities and capacity; a layered device binds those
regions to QEC architectures and physical resources. Placement constraints such
as `cudaq.logical.architecture.colocate(...)` guide the P1 solver. QEC
selection produces P2, and physical projection, routing, native legalization,
and scheduling produce P3. Builds are immutable: continuing one never mutates
it, and every estimate or emission reads a private view.

## Data ownership

Logical qubits, encoded patches, and protocol resources are linear owners.
Operations consume the incoming owner and return its successor; measurement or
an explicit `cudaq.logical.discard` ends ownership. The IR verifiers reject
duplication, stale reuse, and mismatched branch carries — there is no in-place
mutation anywhere in the IR.

```{figure} ../_static/figures/patch-lifecycle.svg
:alt: State machine of a patch's linear ownership from allocation through active execution to terminal readout.
:width: 100%

The linear life of a patch: every arrow consumes the incoming owner and
produces a successor.
```

## The compile pipeline

`cudaq.logical.compiler.pipelines.*` presets name the canonical pass sequences.
Passes declare the facets they require and provide, so an out-of-order pipeline
fails verification instead of guessing:

| Preset                    | Passes                                                                                                                                | Output                                       |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| `logical()`               | `qlx-normalize-actions`, `qlx-infer-requirements`, `qlx-verify-p0`                                                                    | verified P0                                  |
| `placed()`                | `qlx-verify-p0`, `qlx-place`, `qlx-to-lvm`, `lvm-verify-p1`                                                                           | verified P1 + placement witness              |
| `qec()`                   | `lvm-select`, `lvm-apply-qec-lowerings`, `lvm-to-fabric`, `fabric-materialize-default-encodings`, `fabric-verify-generated-protocols` | verified P2 (realization + protocol network) |
| `physical()`              | `fabric-derive-patch-graph`, `fabric-map-patches`, `fabric-to-phys`, `phys-route`, `phys-legalize-native-actions`, `phys-verify-p3`     | verified P3 physical event graph             |
| `clifford_t(precision=…)` | `qlx-synthesize-rotations`, `qlx-verify-clifford-t`                                                                                   | P0 legalized to positive H/S/T/CX            |
| `pbc()`                   | `qlx-to-pbc`, `qlx-verify-pbc`, `qlx-verify-p0`                                                                                       | P0 in Pauli-based-computation form           |

Definition-level recipes verify the other facets: `qec_definitions()`
(`fabric-verify-p2s`, provides `QEC_SPEC`), `gadgets()` (`fabric-verify-p2a`,
requires `QEC_SPEC`, provides `QEC_REALIZATION`), and `protocols()`
(`fabric-link-calls`, `fabric-verify-p2n`, provides `PROTOCOL_NETWORK`).
`device_stack(profile)` verifies an immutable device prefix against its `p1`,
`p2`, or `p3` profile.

Four estimation tiers read verified artifacts without changing their semantic
stage. `Tier.LOGICAL` profiles P0; `Tier.STATIC` counts a selected P2 network;
`Tier.ANALYTICAL` combines P2 counts with an explicit physical model; and
`Tier.SCHEDULE` authenticates resources and timing from a scheduled P3 graph.
Presets are ordinary `Pipeline` values: `insert_after`, `configure`, `replace`,
`append`, and `remove` compose them without string surgery.

## The P2 patch graph

Topology survives as _inspectable evidence_ before physical lowering. A
selected P2 build exposes `build.patch_graph`: a typed, read-only view
(`PatchGraphView`) of the patch instances and logical interactions the
realization implies, derived from canonical `fabric` IR facts. You can convert
the view to NetworkX or render it to PNG when the optional packages are
installed. Placement stays code-agnostic at P1, and the interaction structure
that a placement implies becomes checkable at P2. P3 then maps those patches to
physical resources, routes their interactions, legalizes native events, and
records the schedule as separate verified facets.

## Verification layers

Three independent layers enforce correctness:

- **Construction**: CUDA-Q Logical validates a code definition as you author
  it — stabilizer shape, logical operators, and distance evidence must agree
  before a `@cudaq.logical.code` exists at all.
- **IR verifiers (C++)**: every stage boundary has a verify pass —
  `qlx-verify-p0`, `lvm-verify-p1`, `fabric-verify-p2s` / `-p2a` / `-p2n`,
  `fabric-verify-machine`, and `phys-verify-p3`, plus the gate-set verifiers
  `qlx-verify-clifford-t` and `qlx-verify-pbc`. Missing or inconsistent evidence
  fails closed. A Python linear-use analysis (`compiler/linearity.py`) and
  strict link completeness (`compiler/link_check.py`) back them on the
  authoring side.
- **Semantic verification**: the analysis derives a gadget's Clifford action
  from its body and compares it against the claimed objective
  (`cudaq.logical.gadgets.analysis.clifford_action`); names never select
  semantics.

## Typed inspection

You inspect builds through typed APIs, not text scraping: `build.stage`,
`build.placement`, `build.definitions`, `build.calls(symbol)`,
`build.protocol_for(objective)`, `build.status`, `build.synthesis`,
`build.patch_graph`, `build.to_mlir()`, `build.content_sha256`, and
`build.serialize()` / `Build.replay(...)`. Estimate results are typed the same
way: `LogicalEstimate`, `FabricCounts`, `FabricEstimate`, and
`ScheduleEstimate` expose structured results from the four estimation tiers.

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
  architecture/  logical, QEC, and physical architecture
  devices/       layered devices, resources, regions, reusable local recipes
  compiler/      pipelines, immutable builds, placement, synthesis, topology
  estimate/      logical, static, analytical, and schedule estimation
  experiments/   immutable, reproducible compilation experiments
  algorithms/    algorithm libraries (the Gidney–Ekerå study)
  analysis/      resource-estimation and evidence APIs
  lower/         target lowering and Stim emission
  targets/       built-in CUDA-Q targets (estimator, clifford_t, surface)
  std/           standard resource kinds and objectives (T_STATE, ...)
  qec/           QEC implementation-lowering contracts
  dialects/      generated MLIR Python bindings (qlx, lvm, fabric, phys)
  stages.py      stage and facet enums
```

## Extensibility

You can write the same typed values the decorator surface produces through the
lower-level builders. Pipelines are composable values, gate sets are data
(`GateSet`: actions plus legalization passes), and new CUDA-Q targets wrap a
backend with `Target.from_backend(...)`. Device- and code-specific libraries
are ordinary Python modules, not global registries.

## Where to go next

- The [core concepts](../getting-started/concepts.md) explain the stage,
  ownership, and evidence model this architecture implements.
- The [use cases](../use-cases/define-a-code.md) apply the Python interface to
  codes, placement, synthesis, estimation, and emission.
