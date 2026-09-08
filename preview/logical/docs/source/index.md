# CUDA-Q Logical — Develop and evaluate fault-tolerant quantum applications

:::{admonition} Preview release

`cudaq-logical` is in preview. Its APIs, behavior, and documentation may change
substantially in upcoming versions.

:::

**CUDA-Q Logical** (`cudaq.logical`) expands CUDA-Q with an open, extensible
logical layer for fault-tolerant quantum computing. Use it to express
fault-tolerant workloads, evaluate them across different QEC codes and system
architectures, and understand the resources they need to run.

Fault-tolerant quantum computing is a co-design problem. Choices in the
application, QEC code, logical architecture, physical hardware, decoding,
control, and classical computing can all change the qubits and runtime required
for the same computation. CUDA-Q Logical keeps the workload fixed while these
system choices change, so you can compare results directly and inspect the
assumptions behind each one.

In this preview, you can start from a CUDA-Q kernel or author a portable logical
program directly; configure codes, gadgets, placement, and distillation
protocols; generate resource estimates; and emit realized programs for
simulation and further analysis in tools such as Stim.

## Installation

CUDA-Q Logical comes pre-installed with `cudaq`, so one command is enough:

```bash
pip install cudaq
```

For more installation options or to build from source, see
[Quick start](getting-started/quickstart.md).

## An end-to-end estimate

As a first step, define a CUDA-Q Logical target that describes the desired QEC
encoding: for instance, a distance-3 surface code with room for one logical
qubit. You can then lower any `cudaq.kernel` to a fault-tolerant program.
`cudaq.estimate` returns the required resources for the encoded computation:

```{eval-rst}
.. literalinclude:: ../../examples/00_cudaq_logical_resource_estimate.py
   :language: python
   :caption: An ordinary CUDA-Q kernel lowered through a surface-code encoding (examples/00_cudaq_logical_resource_estimate.py).
```

When you call `cudaq.estimate`, CUDA-Q Logical takes the kernel's logical
intent, realizes it with the code and gadgets the target names, and reports the
cost of that choice (peak encoded patches, peak protected logical qubits, and
the named gadget calls that make up the estimate). The
[examples](use-cases/examples/index.md) page shows more of what CUDA-Q Logical
can do.

CUDA-Q kernels are only one entry point. You can also author portable logical
programs directly, define your own codes and gadgets, and place computations on
a logical machine. The compiler refines a program through strict semantic
stages, producing immutable evidence at each transition; see the
[architecture reference](reference/architecture.md) for the advanced staged
model.

## Choose your route

Start from the result you want:

| If you want to…                                                            | Go to                                                                         |
| -------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **Install and run a first estimate**                                       | [Getting started](getting-started/quickstart.md)                              |
| **Build, place, and estimate a logical program**                           | [The logical programming stack](getting-started/cudaq-logical-in-practice.md) |
| **Do a specific job** (define a code, place, distill, estimate, emit Stim) | [Use cases](use-cases/define-a-code.md)                                       |
| **Browse runnable end-to-end examples**                                    | [Examples](use-cases/examples/index.md)                                       |
| **Inspect advanced internals or build against CUDA-Q**                     | [Reference](reference/architecture.md)                                        |

```{toctree}
:maxdepth: 2
:hidden:
:caption: Getting started

getting-started/quickstart
getting-started/cudaq-logical-in-practice
getting-started/concepts
use-cases/examples/index
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Use cases

use-cases/define-a-code
use-cases/gadgets-and-verification
use-cases/devices-and-placement
use-cases/magic-states-and-protocols
use-cases/logical-synthesis
use-cases/estimation
use-cases/stim-emission
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Reference

reference/for-stim-users
reference/capabilities
reference/architecture
reference/building-against-cudaq
```
