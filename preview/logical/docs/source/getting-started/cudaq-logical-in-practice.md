# The logical programming stack

Follow one Bell-pair computation from a portable logical program to a physical
schedule using the Steane code. First, compile, place, and estimate the program
directly with `cudaq.logical`. Then expose the same device as a CUDA-Q target
so application developers can estimate a kernel with a single
`cudaq.estimate` call.

Install CUDA-Q Logical following the [quick start](quickstart.md). See
[Core concepts](concepts.md) for the definitions used throughout this guide.

The excerpts below come from the runnable
[Steane Bell-pair example](../../../examples/steane_bell_e2e_workflow.py).
From the repository root, run:

```bash
python3 preview/logical/examples/steane_bell_e2e_workflow.py
```

The [compilation stages](concepts.md#definitions-builds-and-compilation) are
**P0**, **P1**, **P2**, and **P3**:

```text
P0: logical program → P1: placement → P2: QEC realization → P3: physical schedule
```

Each stage adds the information needed for more detailed resource estimates.

## 1. Compile and estimate the logical program

Define a [logical program](concepts.md#logical-programs-and-quantum-values) that
prepares and measures a Bell pair. The program specifies the computation
independently of its QEC code and device:

```{literalinclude} ../../../examples/steane_bell_e2e_workflow.py
:language: python
:start-after: START_LOGICAL_PROGRAM
:end-before: END_LOGICAL_PROGRAM
```

The assignments retain the successor values returned by each operation; see
[linear ownership](concepts.md#logical-programs-and-quantum-values) for why
these assignments are required.

`cql.compile` translates the program into a P0
[build](concepts.md#definitions-builds-and-compilation). Pass that build to
`cql.estimate` to count its logical resources:

```{literalinclude} ../../../examples/steane_bell_e2e_workflow.py
:language: python
:start-after: START_LOGICAL_ESTIMATE
:end-before: END_LOGICAL_ESTIMATE
```

```text
Peak logical qubits: 2
Hadamard actions: 1
Controlled-X actions: 1
```

These counts describe the workload. Encoded blocks, physical qubits, and timing
depend on the realization we choose next.

## 2. Choose the code and its implementations

Define the Steane `[[7,1,3]]`
[CSS code](concepts.md#codes-and-operation-implementations) explicitly. It
protects one logical qubit with seven data qubits; `CSSBlock` also reserves
three X-check and three Z-check ancillas. The `hx` and `hz` rows list the
data-qubit indices for each X and Z stabilizer check, while `lx` and `lz` define
the logical X and Z operators.

`@cql.code` validates the code's algebraic structure.
[`cql.materialize`](concepts.md#definitions-builds-and-compilation) creates its compiler
representation, which can be inspected independently of the Bell program:

```{literalinclude} ../../../examples/steane_bell_e2e_workflow.py
:language: python
:start-after: START_STEANE_CODE
:end-before: END_STEANE_CODE
```

The same code is also available as the built-in `cql.codes.Steane`. To use it,
replace the class definition with `Steane = cql.codes.Steane`.

Supply [gadgets](concepts.md#codes-and-operation-implementations) for the Bell
program's Hadamard and controlled-X operations. Each `implements=` clause
names the ideal logical
[action](concepts.md#codes-and-operation-implementations); the body
describes its implementation on encoded patches. For Steane, these operations
can be applied across the data qubits:

```{literalinclude} ../../../examples/steane_bell_e2e_workflow.py
:language: python
:start-after: START_STEANE_GADGETS
:end-before: END_STEANE_GADGETS
```

Combine these gadgets with the supplied preparation and measurement gadgets in
a [QEC architecture](concepts.md#codes-and-operation-implementations).
Together they cover every operation in the Bell program:

```{literalinclude} ../../../examples/steane_bell_e2e_workflow.py
:language: python
:start-after: START_QEC_ARCHITECTURE
:end-before: END_QEC_ARCHITECTURE
```

See [Defining a code](../use-cases/define-a-code.md) and
[Gadgets and verification](../use-cases/gadgets-and-verification.md) for custom
definitions.

## 3. Describe the device

A [device](concepts.md#devices-and-placement) connects the logical machine, QEC
realization, and physical resources. First, give its
[logical machine](concepts.md#devices-and-placement) a compute region with two
slots, then bind that region to the Steane architecture:

```{literalinclude} ../../../examples/steane_bell_e2e_workflow.py
:language: python
:start-after: START_LOGICAL_DEVICE
:end-before: END_LOGICAL_DEVICE
```

Each logical qubit needs one Steane block. Here,
`CSSBlock(data=7, sx=3, sz=3)` reserves seven data qubits and six check ancillas
per block. The device therefore reserves `2 * (7 + 3 + 3) = 26` physical qubits:
14 data qubits and 12 ancillas. The ancillas are part of this example's block
layout. Declare the operations, measurements, and timing available on the
physical machine:

```{literalinclude} ../../../examples/steane_bell_e2e_workflow.py
:language: python
:start-after: START_PHYSICAL_DEVICE
:end-before: END_PHYSICAL_DEVICE
```

This is a teaching device with an illustrative one-nanosecond cycle and native
Clifford operations and Pauli-product measurements. Its resource and timing
results describe these assumptions. See
[Devices and placement](../use-cases/devices-and-placement.md) for other device
configurations.

## 4. Place and realize the program

Use [`cql.compiler.place`](concepts.md#devices-and-placement) to assign the P0
program's logical values to the device's logical machine. The constraint keeps
the named `data` allocation in one region:

```{literalinclude} ../../../examples/steane_bell_e2e_workflow.py
:language: python
:start-after: START_PLACEMENT
:end-before: END_PLACEMENT
```

```text
[('compute', 0), ('compute', 1)]
```

This P1 build records placement. It has not yet selected the QEC
implementations. Continue compilation with the QEC
[pipeline](concepts.md#definitions-builds-and-compilation) to produce P2, then
estimate the encoded resources directly:

```{literalinclude} ../../../examples/steane_bell_e2e_workflow.py
:language: python
:start-after: START_STATIC_ESTIMATE
:end-before: END_STATIC_ESTIMATE
```

```text
Peak encoded patches: 2
Hadamard gadget calls: 1
Controlled-X gadget calls: 1
```

The same Bell program now uses two Steane patches. Its selected implementations
also include two preparations and two logical-Z measurements.

## 5. Schedule and estimate the physical resources

Continue with the physical pipeline and
[schedule](concepts.md#schedules-and-resource-estimates) the resulting P3 events:

```{literalinclude} ../../../examples/steane_bell_e2e_workflow.py
:language: python
:start-after: START_PHYSICAL_SCHEDULE
:end-before: END_PHYSICAL_SCHEDULE
```

`cql.estimate` can now report physical resources and timing. Keep the estimate
options together so we can reuse them in the CUDA-Q target below:

```{literalinclude} ../../../examples/steane_bell_e2e_workflow.py
:language: python
:start-after: START_SCHEDULE_ESTIMATE
:end-before: END_SCHEDULE_ESTIMATE
```

```text
Physical qubits: 26
Scheduled events: 13
Makespan: 4.0 ns
```

The [estimation tier](concepts.md#schedules-and-resource-estimates) determines
which resources to report and which compilation stage is required:

| Tier           | Available from | What it reports                                      |
| -------------- | -------------- | ---------------------------------------------------- |
| **LOGICAL**    | P0             | logical qubits, actions, and instruments             |
| **STATIC**     | P2             | encoded patches, gadget calls, and authored actions  |
| **ANALYTICAL** | P2             | modeled physical cost, error, acceptance, and timing |
| **SCHEDULE**   | P3             | physical resources, timed events, and utilization    |

See [Resource estimation](../use-cases/estimation.md) for the analytical tier
and the complete result fields. The outputs above are estimates for preparing
and measuring the Bell pair under this device model.

## 6. Expose the device as a CUDA-Q target

The explicit calls above are useful when developing a device or inspecting
compilation stages. For application developers, a
[compilation target](concepts.md#cuda-q-integration) packages that workflow so
an ordinary CUDA-Q kernel can use the same code,
gadgets, physical bindings, and estimate options.

Expose `device` through `Target.from_device` and select the resulting target:

```{literalinclude} ../../../examples/steane_bell_e2e_workflow.py
:language: python
:start-after: START_CUDA_Q_TARGET
:end-before: END_CUDA_Q_TARGET
```

This target can now be used with CUDA-Q kernels. For example, the same logical
program can be written as a Bell-pair kernel:

```{literalinclude} ../../../examples/steane_bell_e2e_workflow.py
:language: python
:start-after: START_CUDA_Q_KERNEL
:end-before: END_CUDA_Q_KERNEL
```

The target now handles logical compilation, placement, QEC realization, and
physical scheduling when the application calls `cudaq.estimate`. Recover the
logical and static counts with typed views; the schedule results are available
in the `SCHEDULE` annotation:

```{literalinclude} ../../../examples/steane_bell_e2e_workflow.py
:language: python
:start-after: START_KERNEL_ESTIMATE
:end-before: END_KERNEL_ESTIMATE
```

Both entry points estimate the same Bell-pair workload on the same device. An
application can reuse the target for compatible kernels without repeating the
individual compiler calls.

## Where to go next

- Execute an encoded program with `cudaq.run` in the
  [Steane Bernstein–Vazirani example](../use-cases/examples/steane-bernstein-vazirani.md).
  It supplies an encoding circuit and returns measurement bits per shot.
  The [T-state distillation example](../use-cases/examples/t-factory-evaluation.md)
  runs on distance-1 `BareQubit` patches and adds non-Clifford gates,
  measurement feedback, and noise.
- Read [Core concepts](concepts.md) for definitions of programs, machines,
  codes, gadgets, devices, and targets.
- Work through the guides for [defining a code](../use-cases/define-a-code.md),
  [placing a program](../use-cases/devices-and-placement.md), and
  [estimating resources](../use-cases/estimation.md).
- Browse the [examples](../use-cases/examples/index.md). The
  [Carbon target](../use-cases/examples/carbon-code.md) builds a custom code and
  integrates paired operations with CUDA-Q kernels.
