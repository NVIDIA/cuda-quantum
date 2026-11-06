# The logical programming stack

Follow one Bell pair, written as an ordinary CUDA-Q kernel, from the kernel to
a physical schedule on Steane-encoded qubits, and then run it. First, compile,
place, and estimate the kernel with explicit `cudaq.logical` calls. Then
estimate it with a single `cudaq.estimate` call through a CUDA-Q target built
from the same device. Finally, run the kernel and its physical build with
`cudaq.run`.

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

A *logical program* describes the computation on logical qubits, independent of
the QEC code and device. Each stage adds the information needed for more detailed
resource estimates.

## 1. Write the kernel

Write the Bell pair as a CUDA-Q kernel that returns its two measurements. The
kernel specifies the computation independently of its QEC code and device:

```{literalinclude} ../../../examples/steane_bell_e2e_workflow.py
:language: python
:start-after: START_BELL_KERNEL
:end-before: END_BELL_KERNEL
```

## 2. Choose the code and its implementations

Define the Steane `[[7,1,3]]`
[CSS code](concepts.md#codes-and-operation-implementations) explicitly. It
protects one logical qubit with seven data qubits; `CSSBlock` also reserves
three X-check and three Z-check ancillas. The `hx` and `hz` rows list the
data-qubit indices for each X and Z stabilizer check, while `lx` and `lz` define
the logical X and Z operators.

`@cql.code` validates the code's algebraic structure.
[`cql.materialize`](concepts.md#definitions-builds-and-compilation) creates its compiler
representation, which can be inspected independently of the Bell kernel:

```{literalinclude} ../../../examples/steane_bell_e2e_workflow.py
:language: python
:start-after: START_STEANE_CODE
:end-before: END_STEANE_CODE
```

The same code is also available as the built-in `cql.codes.Steane`. To use it,
replace the class definition with `Steane = cql.codes.Steane`.

Supply [gadgets](concepts.md#codes-and-operation-implementations) for the
kernel's Hadamard and controlled-X operations. Each `implements=` clause
names the ideal logical
[action](concepts.md#codes-and-operation-implementations); the body
describes its implementation on encoded patches. For Steane, these operations
can be applied across the data qubits:

```{literalinclude} ../../../examples/steane_bell_e2e_workflow.py
:language: python
:start-after: START_STEANE_GADGETS
:end-before: END_STEANE_GADGETS
```

Each logical qubit also needs a preparation. Resetting the seven data qubits
gives `|0000000⟩`, which does not satisfy the X stabilizers, so supply an
encoder that prepares logical `|0⟩`.
[`cql.gadgets.verify_gadget`](../use-cases/gadgets-and-verification.md) checks
it. The contract binds the prepared block's syndrome to zero, which requires
every stabilizer to be +1, and the `prepare_zero` claim requires logical Z to
be +1:

```{literalinclude} ../../../examples/steane_bell_e2e_workflow.py
:language: python
:start-after: START_STEANE_ENCODER
:end-before: END_STEANE_ENCODER
```

Combine these gadgets with the library's logical-Z measurement in a
[QEC architecture](concepts.md#codes-and-operation-implementations).
Together they cover every operation in the kernel:

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

A [compilation target](concepts.md#cuda-q-integration) packages the device so
that CUDA-Q can compile and estimate kernels for it. Expose `device` through
`Target.from_device`, with the estimate options that the target and the
explicit estimates below share:

```{literalinclude} ../../../examples/steane_bell_e2e_workflow.py
:language: python
:start-after: START_CUDA_Q_TARGET
:end-before: END_CUDA_Q_TARGET
```

## 4. Compile the kernel for CUDA-Q Logical

`cql.compiler.import_cudaq` compiles the kernel to a logical program, a P0
[build](concepts.md#definitions-builds-and-compilation). The P0 program returns
one result per measurement, in the kernel's return order:

```{literalinclude} ../../../examples/steane_bell_e2e_workflow.py
:language: python
:start-after: START_KERNEL_IMPORT
:end-before: END_KERNEL_IMPORT
```

Pass the build to `cql.estimate` to count its logical resources:

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
depend on the realization chosen above. See
[Capabilities](../reference/capabilities.md#execution-and-documented-boundaries)
for the results a kernel can return.

## 5. Place and realize the program

Use [`cql.compiler.place`](concepts.md#devices-and-placement) to assign the P0
program's logical qubits to the device's logical machine:

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

The Bell pair now uses two Steane patches. Its selected implementations also
include two encoder preparations and two logical-Z measurements.

## 6. Schedule and estimate the physical resources

Continue with the physical pipeline and
[schedule](concepts.md#schedules-and-resource-estimates) the resulting P3 events:

```{literalinclude} ../../../examples/steane_bell_e2e_workflow.py
:language: python
:start-after: START_PHYSICAL_SCHEDULE
:end-before: END_PHYSICAL_SCHEDULE
```

`cql.estimate` can now report physical resources and timing under the estimate
options defined with the target:

```{literalinclude} ../../../examples/steane_bell_e2e_workflow.py
:language: python
:start-after: START_SCHEDULE_ESTIMATE
:end-before: END_SCHEDULE_ESTIMATE
```

```text
Physical qubits: 26
Scheduled events: 25
Makespan: 10.0 ns
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

## 7. Estimate the kernel through the target

The explicit calls above are useful when developing a device or inspecting
compilation stages. An application can instead select the target and call
`cudaq.estimate` on the kernel. The target then handles logical compilation,
placement, QEC realization, and physical scheduling. Recover the logical and
static counts with typed views; the schedule results are available in the
`SCHEDULE` annotation:

```{literalinclude} ../../../examples/steane_bell_e2e_workflow.py
:language: python
:start-after: START_KERNEL_ESTIMATE
:end-before: END_KERNEL_ESTIMATE
```

Both entry points estimate the same Bell pair on the same device. An
application can reuse the target for compatible kernels without repeating the
individual compiler calls.

## 8. Run the kernel and its physical build

`cudaq.run` executes the kernel directly. It also executes the P3 build, which
it first translates to Quake with `Build.to_quake()`. Both circuits are
Clifford, so the `stim` target runs them:

```{literalinclude} ../../../examples/steane_bell_e2e_workflow.py
:language: python
:start-after: START_CUDA_Q_RUN
:end-before: END_CUDA_Q_RUN
```

```text
CUDA-Q kernel:  outcomes        {(True, True): 52, (False, False): 48}
P3 Build:       outcomes        {(False, False): 46, (True, True): 54}
```

Both runs give only the two Bell outcomes, and the counts vary from run to run.
The kernel returns each shot as the tuple it declares. The P3 build returns
its two results as a `list[bool]`, in the kernel's return order, so the cell
converts each shot with `tuple(shot)` before counting.

## 9. Optional: inspect the executed Quake

The Quake that the P3 build runs is an ordinary CUDA-Q module. This cell counts
its operations next to the kernel's:

```{literalinclude} ../../../examples/steane_bell_e2e_workflow.py
:language: python
:start-after: START_QUAKE_INSPECTION
:end-before: END_QUAKE_INSPECTION
```

```text
CUDA-Q kernel: 2 qubits, 1 h, 1 cx, 2 mz
P3 Build: 28 qubits, 13 h, 39 cx, 14 reset, 2 mz
```

The Quake allocates all 13 carriers of each block, including the six check
ancillas this program never uses, plus one ancilla for each logical-Z
measurement. The schedule counts that measurement as one native Pauli-product
event, so it reports only the 26 carriers. These counts follow from how this
release translates P3 to Quake. Use the schedule estimate for physical
resources.

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
