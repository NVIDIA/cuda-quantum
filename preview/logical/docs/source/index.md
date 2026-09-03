# CUDA-Q Logical — from logical intent to QEC resource estimates

**CUDA-Q Logical** (`cudaq.logical`) is CUDA-Q's toolkit for estimating the
resources required by fault-tolerant quantum computations. It connects ordinary
CUDA-Q kernels and portable logical programs to logical placement, codes,
gadgets, distillation protocols, and inspectable resource estimates. CUDA-Q
Logical also supports emitting a realized program for further processing in
third party tools such as Stim.

## An end-to-end estimate

As a first step, define a CUDA-Q Logical target that describes the desired QEC
encoding: for instance, a distance-3 surface code with room for one logical
qubit. We can then lower any `cudaq.kernel` to a fault-tolerant program.
`cudaq.estimate` returns the required resources for the encoded computation:

```{literalinclude} ../../examples/00_cudaq_logical_resource_estimate.py
:language: python
:caption: An ordinary CUDA-Q kernel lowered through a surface-code encoding (examples/00_cudaq_logical_resource_estimate.py).
```

Behind `cudaq.estimate`, CUDA-Q Logical takes the kernel's logical intent,
realizes it with the code and gadgets the target names, and reports the cost of
that choice (peak encoded patches, peak protected logical qubits, and the named
gadget calls that make up the estimate). For more examples of CUDA-Q Logical's
features, see the [example gallery](examples.md).

CUDA-Q kernels are only one of the possible entry points. You can also author
portable logical programs directly, define your own codes and gadgets, and place
computations on a logical machine. The compiler refines a program through strict
semantic stages, producing immutable evidence at each transition; see the
[architecture](architecture.md) for the full staged model.

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
