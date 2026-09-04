# Quick start

From a clean Python environment to a verified QEC resource estimate and emitted
Stim circuit text. Every command on this page assumes `cudaq-logical` is
installed, and every embedded program is one of the shipped, test-executed
examples under `preview/logical/examples/`.

You will:

1. install CUDA-Q Logical;
2. estimate an existing CUDA-Q kernel against a surface-code target;
3. author a portable P0 logical program and read its logical estimate;
4. define a QEC code and a verified gadget, and count their P2 operations;
5. emit standards-compatible Stim text from the command line.

## Install

The simplest way to install CUDA-Q Logical is to get it with the `cudaq` base
package:

```bash
pip install cudaq
```

Alternatively, installing CUDA-Q Logical separately can be achieved with the
following command:

```bash
pip install cudaq-logical[cu13]
```

where `[cu13]` should be replaced with `[cu12]` if you have a copy of CUDA 12
installed on your machine.

:::{admonition} Always include `[cu13]` or `[cu12]`

If you opt to install `cudaq-logical` standalone, make sure to always suffix
`cudaq-logical` with either `[cu13]` or `[cu12]`, defaulting to `[cu13]` if your
machine does not have a CUDA installation. The required dependencies are not
included in the bare package.

:::

To build from source against CUDA-Q, see
[Building against CUDA-Q](../reference/building-against-cudaq.md).

## Running the shipped examples

From a checkout of this repository, run any example file with:

```bash
python3 preview/logical/examples/01_p0_bell.py
```

Every `python3` command below is the same invocation with a different example
path.

## Step 1 — Estimate an existing CUDA-Q kernel

The fastest route to a first number: take an ordinary CUDA-Q kernel, select a
distance-3 rotated-surface-code target with room for one logical qubit, and
estimate its resources through the CUDA-Q target integration.

```{eval-rst}
.. literalinclude:: ../../../examples/00_cudaq_logical_resource_estimate.py
   :language: python
   :caption: examples/00_cudaq_logical_resource_estimate.py
```

```bash
python3 preview/logical/examples/00_cudaq_logical_resource_estimate.py
```

The example prints the selected backend stack and finishes with the annotated
resource counts of the resulting P2 build:

```text
CUDA-Q logical-zero resources:
  peak encoded patches: 1
  peak protected logical qubits: 1
  CUDA-Q Logical operation counts: {'alloc': 1, 'call': 2, 'dealloc': 1, 'measure_product': 1, 'prep_z': 1}
  CUDA-Q Logical gadget calls: {'rotated_surface_3_measure_z0': 1, 'rotated_surface_3_prepare_zero': 1}
```

One encoded patch protects one logical qubit through preparation and Z readout,
and every step is a named, inspectable gadget call — CUDA-Q Logical never
invents an implementation it cannot point to.

## Step 2 — Author a portable P0 program

The `cudaq.logical` facade authors logical programs directly. A P0 program names
no code, no device, and no carrier — and it is already estimable.

```{eval-rst}
.. literalinclude:: ../../../examples/01_p0_bell.py
   :language: python
   :caption: examples/01_p0_bell.py
```

```bash
python3 preview/logical/examples/01_p0_bell.py
```

```text
P0 Bell: 2 logical qubits
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
.. literalinclude:: ../../../examples/03_code_and_gadget.py
   :language: python
   :caption: examples/03_code_and_gadget.py
```

```bash
python3 preview/logical/examples/03_code_and_gadget.py
```

```text
Steane [[7,1,3]] terminal-memory gadget:
  physical data qubits per logical block: 7
  independent X/Z stabilizer checks: 6
  logical Z support: (0, 1, 2, 3, 4, 5, 6)
  authored operations: {'reset': 2, 'h': 2, 'cx': 2, 'read_syndrome_ancillas': 1, 'mz': 1, 'dealloc': 1}
```

`ql.materialize` and `ql.compile` lower both artifacts into the `fabric`
dialect, where `ql.analysis.count` reports the gadget's authored operations.
These verified, named gadgets are the atoms of every P2 static estimate — the
surface-code counts in Step 1 are sums over exactly such calls.

## Step 4 — Emit Stim text from Python

A verified P2 entry gadget can be emitted as standards-compatible Stim circuit
text — CUDA-Q Logical's secondary interchange path. Compile the Steane memory
gadget from Step 3 and request a typed emission artifact:

<!--
% invisible-code-block: python
%
% steane_memory = load_ql_example(
% "preview/logical/examples/03_code_and_gadget.py", "steane_memory")
-->

```python
import cudaq.logical as ql

build = ql.compile(steane_memory)
emission = ql.lower.emit_stim_artifact(
    build.module, root_symbol=build.root.symbol)
print(emission.text)
```

The resulting Stim circuit contains the explicit resets, Clifford operations,
and measurements in the selected encoded gadget. `emission.interface` records
the compiled boundary from which the text was projected, and the text loads
directly with the reference `stim` Python package.

## Where to go next

- Continue with [CUDA-Q Logical in practice](cudaq-logical-in-practice.md) for
  the complete staged lowering.
- Use the task-oriented [code](../use-cases/define-a-code.md),
  [placement](../use-cases/devices-and-placement.md), and
  [estimation](../use-cases/estimation.md) guides.
- Browse the remaining runnable studies in
  [Examples](../use-cases/examples/index.md).
- Inspect compiler internals in the
  [architecture reference](../reference/architecture.md).

To run the full conformance suite after building from source, see
[Building against CUDA-Q](../reference/building-against-cudaq.md#verifying-the-build).
