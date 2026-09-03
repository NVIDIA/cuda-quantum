# CUDA-Q Logical — from logical intent to QEC resource estimates

**CUDA-Q Logical** (`cudaq.logical`) is CUDA-Q's toolkit for estimating the
resources required by fault-tolerant quantum computations. It connects ordinary
CUDA-Q kernels and portable logical programs to logical placement, codes,
gadgets, distillation protocols, and inspectable resource estimates. As a
secondary interchange path, a realized program can be emitted as
standards-compatible Stim circuit text.

## An end-to-end estimate

You do not need to learn a new language to get a first number. Take an ordinary
`cudaq.kernel`, point it at a CUDA-Q Logical target that declares a QEC encoding
— here a distance-3 surface code with room for one logical qubit — and
`cudaq.estimate` returns the resources of the encoded computation.

```{literalinclude} ../../examples/00_cudaq_logical_resource_estimate.py
:language: python
:caption: An ordinary CUDA-Q kernel lowered through a surface-code encoding (examples/00_cudaq_logical_resource_estimate.py).
```

Behind that one call, CUDA-Q Logical takes the kernel's logical intent, realizes
it with the code and gadgets the target names, and reports the cost of that
choice: peak encoded patches, peak protected logical qubits, and the named
gadget calls that make up the estimate. Every step is an inspectable, verified
artifact — CUDA-Q Logical never invents an implementation it cannot point to.
Every embedded example in these docs is shipped under `examples/` and executed
by the test suite — nothing in this build is stale display code.

Kernels are only one entry point. You can also author portable logical programs
directly, define your own codes and gadgets, and place computations on a logical
machine. The compiler refines a program through strict semantic stages,
producing immutable evidence at each transition; see the
[architecture](architecture.md) for the full staged model.

Python is the main authoring interface. The same compiler and estimation
workflows are available from the `qlx-opt` command line through ordinary MLIR
pass-pipeline strings, and Stim circuits are emitted with `qlx-translate`.

## Choose your route

The documentation is organized by the result you want:

| If you want to…                                                            | Go to                                       |
| -------------------------------------------------------------------------- | ------------------------------------------- |
| **Install and run a first estimate**                                       | the [quickstart](quickstart.md)             |
| **Browse runnable end-to-end examples**                                    | the [example gallery](examples.md)          |
| **Understand stages, ownership, and evidence**                             | the [concepts guide](concepts.md)           |
| **Do a specific job** (define a code, place, distill, estimate, emit Stim) | the [workflow guides](workflows/index.md)   |
| **Build against CUDA-Q or contribute**                                     | the [reference section](reference/index.md) |

```{toctree}
:maxdepth: 2
:hidden:
:glob:
:caption: Start

start/*
```

```{toctree}
:maxdepth: 2
:hidden:
:glob:
:caption: Guides

*
```

```{toctree}
:maxdepth: 2
:hidden:
:glob:
:caption: Examples

example-gallery/*
```

```{toctree}
:maxdepth: 2
:hidden:
:glob:
:caption: Workflows

workflows/*
```

```{toctree}
:maxdepth: 2
:hidden:
:glob:
:caption: Reference

reference/*
```
