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
5. emit standards-compatible Stim text from the command line.

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

You author logical programs directly through the `cudaq.logical` facade. A P0
program names no code, no device, and no carrier — and you can already estimate
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

`ql.materialize` and `ql.compile` lower both artifacts into the `fabric`
dialect, where `ql.analysis.count` reports the gadget's authored operations.
These verified, named gadgets are the atoms of every P2 static estimate — the
surface-code counts in Step 1 are sums over exactly such calls.

## Step 4 — Emit Stim text from Python

You can emit a verified P2 entry gadget as standards-compatible Stim circuit
text, CUDA-Q Logical's secondary interchange path. Compile the Steane memory
gadget from Step 3 and request a typed emission artifact:

<!--
% invisible-code-block: python
%
% steane_memory = load_ql_example(
% "preview/logical/examples/standalone/02_code_and_gadget.py", "steane_memory")
-->

```python
import cudaq.logical as ql

build = ql.compile(steane_memory)
emission = ql.lower.emit_stim_artifact(
    build.module, root_symbol=build.root.symbol)
print(emission.text)
```

The resulting Stim circuit contains the explicit resets, Clifford operations,
and measurements of the selected encoded gadget. `emission.interface` records
the compiled boundary the text was projected from, and the text loads directly
with the reference `stim` Python package.

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

To run the full conformance suite after building from source, see
[Building against CUDA-Q](../reference/building-against-cudaq.md#verifying-the-build).
