# Core concepts

A logical program describes a quantum computation. A quantum error-correction
(QEC) code specifies how its logical qubits are protected using physical qubits.
Gadgets implement logical operations on those encoded qubits, and a device
describes the available resources and their capabilities. CUDA-Q Logical
compiles these inputs into builds and estimates resources from those builds.

For a runnable introduction to this workflow, see the
[Bell-pair walkthrough](cudaq-logical-in-practice.md). Examples use
`import cudaq.logical as cql`.

## Logical programs and quantum values

A **logical program** specifies operations on logical qubits independently of
a QEC code or device. Logical qubits represent the computation's quantum
information; a QEC implementation protects it using physical qubits.
Logical **actions** include unitary gates, such as Hadamard and controlled-X,
and **instruments**, such as preparation and measurement.
Measurements produce classical results; the Bell program returns two Boolean
Z-basis outcomes. You can author the workload as a CUDA-Q kernel or directly
with `@cql.program`.

Quantum values have **linear ownership**: each live qubit has one current value.
A gate consumes its input values and returns their successors. Reassigning the
results keeps the program's references current:

```{literalinclude} ../../../examples/steane_bell_e2e_workflow.py
:language: python
:start-at:     qubits = cql.allocate
:end-at:     return cql.measure_z
:dedent: 4
```

`allocate` creates two logical qubits in the zero state. The gates return
successors, and the terminal `measure_z` calls consume the qubits. Reusing a
consumed value is invalid. Encoded patches follow the same ownership rule.

## Definitions, builds, and compilation

A **definition** is a reusable description authored in Python, such as the
`bell` program, `Steane` code, or `steane_h` gadget. A **build** is a compiled
artifact containing the program or definition and the realization choices made
so far.

`cql.materialize(Steane)` creates a compiler representation of the code for
inspection. `cql.compile(bell)` compiles the logical program into a P0 build.
A **pipeline** specifies compiler steps: `cql.compiler.pipelines.qec()` selects
encoded implementations; `cql.compiler.pipelines.physical()` maps them to
physical resources and events.

Compilation adds information through four **stages**:

| Stage | Information added |
| --- | --- |
| **P0: logical program** | Logical operations, values, and measurement results. |
| **P1: logical placement** | Assignments of logical values to logical machine regions and slots. |
| **P2: QEC realization** | Selected codes, encodings, and operation implementations. |
| **P3: physical events** | Physical allocations, native operations, and dependencies; scheduling adds event times. |

Builds are immutable: continuing compilation produces a new build and retains
the earlier one for inspection or estimation.

See the [architecture reference](../reference/architecture.md) for compiler
representations, pipeline details, and build inspection APIs.

## Clifford+T decomposition

**Clifford+T decomposition**, also called **synthesis**, rewrites logical gates
as sequences of Hadamard (`H`), phase (`S`), controlled-X (`CX`), and `T` gates.
`H`, `S`, and `CX` are Clifford gates; `T` is a non-Clifford phase gate with an
angle of π/4. Expressing a workload in this gate set exposes its T-gate demand
before choosing a QEC implementation.

Some operations have exact decompositions; general rotations require
approximation. The synthesis `precision` sets an error bound for each
approximated rotation. A smaller bound usually requires more T gates, so the
chosen precision affects resource counts. Synthesis produces a new P0 build
without selecting a code or device; placement and QEC compilation can follow.

For direct logical programs, use `cql.compiler.synthesize` with
`gate_set=cql.compiler.gate_sets.clifford_t`. For CUDA-Q kernels,
`cql.targets.clifford_t` provides an estimation target;
`cql.targets.clifford_t_target(precision=...)` sets a custom approximation bound.
A device target can include the same step with
`Target.from_device(..., clifford_t_precision=...)`.

This step is optional. The Bell-pair walkthrough uses only `H` and `CX`, so
it needs no rotation synthesis.

See [Logical Clifford+T synthesis](../use-cases/logical-synthesis.md) for
precision semantics and supported operations, and the
[Clifford+T example](../use-cases/examples/clifford-t.md) for resource estimates
at different precisions.

## Codes and operation implementations

A **QEC code** defines how logical information is protected. The notation
`[[n,k,d]]` describes `k` logical qubits encoded in `n` data qubits with
distance `d`: the minimum number of data qubits affected by an undetectable
nontrivial logical Pauli error.

A **CSS code** defines separate X and Z stabilizer checks: products of X or Z
Pauli operators on selected data qubits. The checks commute and define the
allowed encoded states. Their measurements give a
**syndrome**, information about errors without revealing the logical state.
Steane's `hx` and `hz` list each check's data-qubit indices; `lx` and `lz`
specify logical X and Z operators. `@cql.code` checks the algebraic structure;
`d = 3` separately declares the distance.

A code's data width differs from its complete **block** footprint.
`CSSBlock(data=7, sx=3, sz=3)` reserves seven data qubits and six check
ancillas, auxiliary qubits for check measurements. The walkthrough's Steane
block therefore has 13 physical qubits.

An **encoding** defines the interface to a code's logical qubits.
`Steane.default_encoding` supplies it automatically. A **patch** is one live
encoded block; `cql.patch[Steane]` uses the default encoding, and `block.data`
selects its data qubits.

An **action** states ideal logical behavior. A **gadget** implements that
behavior on encoded patches, with `implements=` identifying the action:

```{literalinclude} ../../../examples/steane_bell_e2e_workflow.py
:language: python
:start-at: "@cql.gadget(implements=cql.standard.h)"
:end-at:     return cql.h(block.data)
```

Here, `cql.standard.h` is the ideal logical Hadamard. The body applies Hadamard
to each Steane data qubit, a **transversal** implementation. Construction and
compilation check code algebra and gadget boundaries; logical-action
verification depends on the supported realization class.

A **gadget contract** declares detector relations, logical observables, and
output-syndrome assignments using the realization's measurement records and
encoded boundaries. `cql.gadgets.verify_gadget` checks detector contracts and
supported Clifford unitary actions, single-logical signed X/Z preparations,
and destructive or preserving X/Z measurements. A default call returns only
when both checks pass; unsupported objectives raise with an inspection receipt. See
[Gadgets and verification](../use-cases/gadgets-and-verification.md).

`cql.analysis.derive_edem` derives an extended detector error model from a
gadget and explicit faults. EDEMs retain encoded boundaries for composition;
a closed chain can be projected to a DEM. See
[Derive an error model from gadgets](../use-cases/gadget-edems.md).

A **QEC architecture** assembles an encoding and its available implementations.
`QECArchitecture.link_roots` supplies the Bell program's preparation, gate,
and measurement gadgets. QEC compilation selects compatible implementations
for the placed program.

See [Defining codes](../use-cases/define-a-code.md) and
[Gadgets and verification](../use-cases/gadgets-and-verification.md) for custom
interfaces and verification details.

## Devices and placement

A **logical machine** describes where logical state can reside and which
operations are supported. A **region** is a pool of logical capacity with
capabilities such as computation or measurement. A **slot** holds one logical
qubit. For example, the Bell device's compute region has two slots.

A **physical machine** describes resources, connectivity, native operations,
and timing assumptions. A **carrier** is a physical qubit resource. Native
operations are those implemented directly. The Bell device includes Hadamard
and controlled-X in its Clifford gate set, Z measurements (`MZ`), and
measurements of Pauli-operator products (`MPP`).

A **device** connects these descriptions. `DeviceBuilder` creates logical
regions, `qec.bind` associates them with QEC architectures or encodings, and
`physical.bind` associates encoded regions with physical resources.
`builder.build()` freezes the device description. The Bell device binds two
logical slots to Steane blocks and provides 26 physical carriers.

**Placement** assigns a particular program's logical values to the machine's
regions and slots. `cql.compiler.place` records these assignments in a P1
build. The constraint `colocate(logical.values.data)` keeps the named
allocation in one region, with a distinct slot for each qubit. Code and gadget
selection follows at P2.

Devices are reusable; workload-specific choices belong to builds.
**Retargeting** keeps the logical workload fixed while changing the device or
QEC architecture, producing a new realization for comparison.

See [Devices and placement](../use-cases/devices-and-placement.md) for other
regions, constraints, and physical bindings.

## Schedules and resource estimates

Physical compilation produces an **event graph**: physical operations and
their dependencies. A **schedule** assigns start times and durations while
respecting dependencies and resource conflicts. `cql.compiler.schedule`
constructs the schedule from the P3 build.

A device's **operating point** supplies model assumptions such as timing and
physical error rates. The walkthrough uses an illustrative `cycle_ns = 1.0`.
Its estimate options specify `cycle_time` in seconds, `p_phys` as the
physical-error probability used by the estimation model, and `failure_budget`
as the acceptable total failure probability for the study.

`cql.estimate` analyzes a build or schedule. The **tier** selects which
quantities to report and which inputs are required:

| Tier | Required input | Reports |
| --- | --- | --- |
| LOGICAL | P0 program | Logical-qubit peak and action/instrument counts. |
| **STATIC** | Selected P2 realization | Encoded-patch peak and gadget/operation counts. |
| **ANALYTICAL** | Selected P2 realization and physical-model assumptions | Modeled physical cost, error, acceptance, and runtime. |
| **SCHEDULE** | Scheduled P3 graph | Physical resources, events, runtime, and utilization. |

A **peak** counts the greatest number of resources simultaneously live.
**Makespan** is the elapsed time from the first scheduled event's start to the
last event's completion. Logical actions, gadget calls, and physical events
count work at different stages; their counts need not match.

The Bell pair therefore has two logical qubits, two encoded patches, and 26
reserved physical qubits. Its four-nanosecond makespan follows from the teaching
device's timing model. More physical detail makes dependencies and contention
visible, while the result still depends on the supplied model assumptions.

See [Resource estimation](../use-cases/estimation.md) for analytical models and
the complete result fields.

## CUDA-Q integration

A **compilation target** packages a device's compilation workflow and an
endpoint that consumes the resulting artifacts. `Target.from_device` creates
that target from the device description, with `cql.targets.estimator` as the
default endpoint. In the walkthrough, `estimate_options` supplies the
estimation parameters.

After `cudaq.set_target(target)`, `cudaq.estimate(bell_pair)` compiles the
CUDA-Q kernel through the configured stages. The result's **annotations**
contain per-tier resource estimates. Typed views such as `LogicalEstimate`
and `FabricCounts` expose logical and static fields; the walkthrough reads
physical schedule results from the `SCHEDULE` annotation.

Physical builds also run on CUDA-Q simulators. When the program returns
Booleans, `cudaq.run(build)` executes its P3 build on a simulator target, such
as `stim` or `nvidia`. Each shot returns a `bool` for one result or a
`list[bool]` for several, the same types `cudaq.run` returns for a kernel.
Unsupported P3 forms raise an error.
[Execution boundaries](../reference/capabilities.md#execution-and-documented-boundaries)
lists the main ones.

Use kernels with CUDA-Q's libraries and infrastructure, or author directly
with `cudaq.logical` when integrating another frontend. Both routes use the
same compilation model. Applications can reuse a target for compatible kernels;
direct compiler calls expose individual builds.

Follow the [Bell-pair walkthrough](cudaq-logical-in-practice.md) to apply these
concepts in one runnable example.
