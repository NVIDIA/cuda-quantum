# Quick start

This page takes you from a clean Python environment to a verified QEC resource
estimate and emitted Stim circuit text. Every command assumes `cudaq-logical`
is installed, and every program shown is one of the shipped, test-executed
examples under `preview/logical/examples/`.

You will:

1. install CUDA-Q Logical;
2. estimate an existing CUDA-Q kernel against a surface-code target;
3. author a portable P0 logical program and read its logical estimate;
4. define a QEC code and a verified gadget, and count their P2 operations;
5. lower a surface-code workload to physical events and schedule it;
6. emit standards-compatible Stim text from the command line.

## Install

The simplest way to install CUDA-Q Logical is with the `cudaq` base package:

```bash
pip install cudaq
```

To install CUDA-Q Logical on its own instead, run:

```bash
pip install cudaq-logical[cu13]
```

Replace `[cu13]` with `[cu12]` if you have CUDA 12 installed.

:::{admonition} Always include `[cu13]` or `[cu12]`

If you install `cudaq-logical` standalone, always suffix it with `[cu13]` or
`[cu12]`, defaulting to `[cu13]` if your machine has no CUDA installation. The
bare package does not include the required dependencies.

:::

To build from source against CUDA-Q, see
[Building against CUDA-Q](../reference/building-against-cudaq.md).

## Running the shipped examples

From a checkout of this repository, run any example with:

```bash
python3 preview/logical/examples/00_logical_resource_estimate.py
```

Every `python3` command below is the same invocation with a different example
path.

## Step 1 — Estimate an existing CUDA-Q kernel

The fastest route to a first number: take an ordinary CUDA-Q kernel, select a
distance-3 rotated-surface-code target with room for one logical qubit, and
estimate its resources through the CUDA-Q target integration.

```{eval-rst}
.. literalinclude:: ../../../examples/02_surface_code_resource_estimate.py
   :language: python
   :caption: examples/02_surface_code_resource_estimate.py
```

```bash
python3 preview/logical/examples/02_surface_code_resource_estimate.py
```

The example prints the selected backend stack, then compares physical-qubit,
event-count, and makespan estimates for distance-3 and distance-5 layouts. It
also holds the layout fixed while changing the physical error rate, failure
budget, and cycle time, making the assumptions behind the estimate explicit.

## Step 2 — Author a portable P0 program

You can also author logical programs directly through the `cudaq.logical` facade.
A P0 program names no code, no device, and no carrier — and you can already estimate
it.

```{eval-rst}
.. literalinclude:: ../../../examples/standalone/00_logical_program.py
   :language: python
   :caption: examples/standalone/00_logical_program.py
```

```bash
python3 preview/logical/examples/standalone/00_logical_program.py
```

```text
Portable Bell program: 2 logical qubits
```

Note the linear style: every operation consumes its operand and returns the
successor, so a stale value is a typed error, not a silent bug. The
`Tier.LOGICAL` estimate counts logical qubits and actions with any repetition
still folded; P2 builds add `Tier.STATIC`, which reads the realized gadget
counts directly.

## Step 3 — Choose a code and a gadget (P2)

P2 is where codes and gadgets enter. This example defines the `[[7,1,3]]` Steane
code as a CSS block, declares a terminal-memory objective, and authors a gadget
that implements it — one syndrome-extraction pass followed by data-qubit
readout.

```{eval-rst}
.. literalinclude:: ../../../examples/standalone/02_code_and_gadget.py
   :language: python
   :caption: examples/standalone/02_code_and_gadget.py
```

```bash
python3 preview/logical/examples/standalone/02_code_and_gadget.py
```

```text
Steane [[7,1,3]] terminal-memory gadget:
  authored operations: {'reset': 2, 'h': 2, 'cx': 2, 'read_syndrome_ancillas': 1, 'mz': 1, 'dealloc': 1}
```

`cudaq.logical.materialize` and `cudaq.logical.compile` lower both artifacts
into the `fabric` dialect, where `cudaq.logical.analysis.count` reports the
gadget's authored operations. These verified, named gadgets are the atoms of
every P2 static estimate — the surface-code counts in Step 1 are sums over
exactly such calls.

## Step 4 — Lower to physical events and schedule them (P3)

Everything so far stopped at the QEC realization. P3 adds the physical layer:
carriers, the binding from encoded regions onto them, and an operating point
that turns dimensionless cycles into time.

```{eval-rst}
.. literalinclude:: ../../../examples/standalone/04_physical_schedule.py
   :language: python
   :start-at: surface_architecture =
   :end-before: # Compile the logical program
   :caption: A device layered across P1, P2, and P3 (examples/standalone/04_physical_schedule.py).
```

With those facts present, the compiler lowers the placed program to a physical
event graph, schedules it, and costs the schedule:

```{eval-rst}
.. literalinclude:: ../../../examples/standalone/04_physical_schedule.py
   :language: python
   :start-at: schedule = cql.compiler.schedule
   :end-before: assert resources.physical_qubits
```

```bash
python3 preview/logical/examples/standalone/04_physical_schedule.py
```

```text
Standalone physical schedule:
  physical qubits: 17
  scheduled events: 50
  makespan: 44.0 ns
```

A distance-3 rotated surface code needs 17 carriers; the three requested
syndrome rounds become 50 scheduled events with a 44 ns makespan under a 1 ns
cycle. Change the operating point and the makespan moves; change the code
distance and the carrier count moves. The logical program is untouched by
either.

## Step 5 — Emit Stim text from the command line

A verified P2 entry gadget can be projected to standards-compatible Stim
circuit text — CUDA-Q Logical's secondary interchange path. `qlx-translate`
ships as a console script with the wheel you installed above. Point it at one
round of Steane syndrome extraction:

```bash
qlx-translate preview/logical/examples/mlir/stim_steane_memory.mlir \
  --fabric-to-stim
```

```stim
R 7 8 9
H 7 8 9
CX 7 0 7 1 7 2 7 3 8 0 8 1 8 4 8 5 9 0 9 2 9 4 9 6
H 7 8 9
R 10 11 12
CX 0 10 1 10 2 10 3 10 0 11 1 11 4 11 5 11 0 12 2 12 4 12 6 12
M 7 8 9
M 10 11 12
M 0 1 2 3 4 5 6
```

Data carriers are 0–6, ancillas 7–12, and every stabilizer coupling is written
out. The text loads directly with the reference `stim` Python package. See
[Stim emission](../use-cases/stim-emission.md) for what the projection refuses
to do.

## Where to go next

- Continue with [Build, place, and estimate a logical
  program](cudaq-logical-in-practice.md) for a step-by-step guide.
- Use the task-oriented [code](../use-cases/define-a-code.md),
  [placement](../use-cases/devices-and-placement.md), and
  [estimation](../use-cases/estimation.md) guides.
- Browse the remaining runnable studies in
  [Examples](../use-cases/examples/index.md).
- Inspect compiler internals in the
  [architecture reference](../reference/architecture.md).

To run the full conformance suite after building from source, see [Building
against CUDA-Q](../reference/building-against-cudaq.md#verifying-the-build).
