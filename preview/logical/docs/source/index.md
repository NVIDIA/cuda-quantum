# CUDA-Q Logical — from logical intent to QEC resource estimates

**CUDA-Q Logical** (`cudaq.logical`) is CUDA-Q's toolkit for estimating the
resources required by fault-tolerant quantum computations. It connects
ordinary CUDA-Q kernels and portable logical programs to logical placement,
codes, gadgets, distillation protocols, and inspectable resource estimates.
As a secondary interchange path, a selected P2 program can be emitted as
standards-compatible Stim circuit text.

```{literalinclude} ../../examples/01_p0_bell.py
:language: python
:caption: A portable P0 Bell program and its logical resource estimate (examples/01_p0_bell.py).
```

That program is portable P0 intent: it does not name a code or a device.
CUDA-Q Logical can then place it on a logical machine, realize it with a QEC
code and gadgets, and report the resource cost of each choice. Every embedded
example in these docs is shipped under `examples/` and executed by the test
suite — nothing in this build is stale display code.

## The model in 30 seconds

A CUDA-Q Logical program is refined through three strict semantic stages:

| Stage | Question it answers | Primary IR family |
|---|---|---|
| **P0** unplaced logical | What logical computation is requested? | `qlx` |
| **P1** placed logical | Where may each logical owner reside on a logical machine? | `lvm` |
| **P2** QEC realization | Which code, gadgets, and protocols realize it, and at what cost? | `fabric` |

Python is the main authoring interface. The same compiler and estimation
workflows are available from the `qlx-opt` command line through ordinary MLIR
pass-pipeline strings, and Stim circuits are emitted with `qlx-translate`.

## Choose your route

The documentation is organized by the result you want:

| If you want to… | Go to |
|---|---|
| **Install and run a first estimate** | the quickstart |
| **Browse runnable end-to-end examples** | the example gallery |
| **Understand stages, ownership, and evidence** | the concepts guide |
| **Do a specific job** (define a code, place, distill, estimate, emit Stim) | the workflow guides |
| **Build against CUDA-Q or contribute** | the reference section |

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
