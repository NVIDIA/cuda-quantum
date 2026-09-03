# Quickstart

From a clean Python environment to a verified QEC resource estimate and
emitted Stim circuit text. Every command on this page was run against the
build it produces, and every embedded program is one of the shipped,
test-executed examples under `preview/logical/examples/`.

You will:

1. install the CUDA-Q development SDK and build CUDA-Q Logical;
2. estimate an existing CUDA-Q kernel against a surface-code target;
3. author a portable P0 logical program and read its logical estimate;
4. define a QEC code and a verified gadget, and count their P2 operations;
5. emit standards-compatible Stim text from the command line.

## Install and build

CUDA-Q Logical builds against an *installed* CUDA-Q development SDK — it is
never added to the CUDA-Q build as a sub-project and never fetches or builds
a second LLVM/MLIR stack. In a fresh Python environment (Python 3.11 or
later), install the SDK wheel and the build tools:

```bash
pip install cudaq-devel nanobind lit cmake ninja
```

```{note}
While CUDA-Q Logical is in preview, `cudaq-devel` may not yet be on your
package index — see [building against CUDA-Q](reference/building-against-cudaq.md)
for the wheelhouse and source-prefix routes.
```

Then, from the repository root, configure and build:

```bash
cmake -S preview/logical -B preview/logical/build -G Ninja
cmake --build preview/logical/build
```

CMake locates the SDK through the Python interpreter it resolves, so the
wheel only has to be installed in the active environment. To build against a
CUDA-Q work tree instead of the wheel, configure with
`-DQLX_CUDAQ_INSTALL_DIR=/path/to/cudaq/install`; the full contract is in
`preview/logical/README.md`. Installing the `cudaq-logical` Python wheel is
an alternative to running from the build tree; this page uses the build tree
directly.

## Running the shipped examples

The build tree at `preview/logical/build/python` contains the complete
`cudaq.logical` package with its native MLIR bindings. Run any example file
against it with:

```bash
PYTHONPATH=preview/logical/build/python \
  python3 -c "import _cudaq_logical_devpath, runpy; runpy.run_path('preview/logical/examples/01_p0_bell.py', run_name='__main__')"
```

The leading `_cudaq_logical_devpath` import makes the build-tree
`cudaq.logical` importable next to the installed CUDA-Q runtime. Every
`python3` command below is this same invocation with a different example
path.

## Step 1 — Estimate an existing CUDA-Q kernel

The fastest route to a first number: take an ordinary CUDA-Q kernel, select a
distance-3 rotated-surface-code target with room for one logical qubit, and
estimate its resources through the CUDA-Q target integration.

```{literalinclude} ../../examples/00_cudaq_logical_resource_estimate.py
:language: python
:caption: examples/00_cudaq_logical_resource_estimate.py
```

```bash
PYTHONPATH=preview/logical/build/python \
  python3 -c "import _cudaq_logical_devpath, runpy; runpy.run_path('preview/logical/examples/00_cudaq_logical_resource_estimate.py', run_name='__main__')"
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

One encoded patch protects one logical qubit through preparation and Z
readout, and every step is a named, inspectable gadget call — CUDA-Q Logical
never invents an implementation it cannot point to.

## Step 2 — Author a portable P0 program

The `cudaq.logical` facade authors logical programs directly. A P0 program
names no code, no device, and no carrier — and it is already estimable.

```{literalinclude} ../../examples/01_p0_bell.py
:language: python
:caption: examples/01_p0_bell.py
```

```bash
PYTHONPATH=preview/logical/build/python \
  python3 -c "import _cudaq_logical_devpath, runpy; runpy.run_path('preview/logical/examples/01_p0_bell.py', run_name='__main__')"
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

P2 is where codes and gadgets enter. This example defines the
``[[7,1,3]]`` Steane code as a CSS block, declares a terminal-memory
objective, and authors a gadget that implements it — one syndrome-extraction
pass followed by data-qubit readout.

```{literalinclude} ../../examples/03_code_and_gadget.py
:language: python
:caption: examples/03_code_and_gadget.py
```

```bash
PYTHONPATH=preview/logical/build/python \
  python3 -c "import _cudaq_logical_devpath, runpy; runpy.run_path('preview/logical/examples/03_code_and_gadget.py', run_name='__main__')"
```

```text
Steane [[7,1,3]] terminal-memory gadget:
  physical data qubits per logical block: 7
  independent X/Z stabilizer checks: 6
  logical Z support: (0, 1, 2, 3, 4, 5, 6)
  authored operations: {'reset': 2, 'h': 2, 'cx': 2, 'read_syndrome_ancillas': 1, 'mz': 1, 'dealloc': 1}
```

`qlx.materialize` and `qlx.compile` lower both artifacts into the `fabric`
dialect, where `qlx.analysis.count` reports the gadget's authored operations.
These verified, named gadgets are the atoms of every P2 static estimate — the
surface-code counts in Step 1 are sums over exactly such calls.

## Step 4 — Emit Stim text from the command line

A verified P2 program can be emitted as standards-compatible Stim circuit
text — CUDA-Q Logical's secondary interchange path. The command-line tools
are built into `preview/logical/build/bin`; add them to `PATH` and translate
the shipped P2 memory fixture:

```{literalinclude} ../../examples/cli/stim_memory.mlir
:language: mlir
:caption: examples/cli/stim_memory.mlir
```

```bash
export PATH="$PWD/preview/logical/build/bin:$PATH"
qlx-translate preview/logical/examples/cli/stim_memory.mlir --fabric-to-stim
```

```stim
R 0
M 0
```

The fixture uses the trivial distance-1 `bare` code, so the circuit is a
single reset/measure pair; encoded codes expand to their full gadget bodies.
The emitted text is standard Stim — it loads directly with the reference
`stim` Python package. The companion `qlx-opt` tool runs the same compiler
pass pipelines the Python facade uses; `preview/logical/examples/cli/` shows
the logical-estimate, placement, and static-estimate pipelines.

## Where to go next

- Browse the remaining runnable studies in the
  [example gallery](example-gallery/index.md) — placement, distillation,
  Clifford+T synthesis, and the Gidney–Ekerå and Fermi–Hubbard estimates.
- Follow the guided introductions under [Start](start/index.md).
- Task-oriented guides — codes, gadgets, devices, estimation, the CLI — live
  in the [workflow guides](workflows/index.md).
- Build-system and SDK contracts are in the
  [reference section](reference/index.md).

To run the full conformance suite — the lit FileCheck tests plus the Python
suite that executes these examples:

```bash
ctest --test-dir preview/logical/build
```
