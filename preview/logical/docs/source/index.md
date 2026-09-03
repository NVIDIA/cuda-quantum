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

```{eval-rst}
.. literalinclude:: ../../examples/00_cudaq_logical_resource_estimate.py
   :language: python
   :caption: An ordinary CUDA-Q kernel lowered through a surface-code encoding (examples/00_cudaq_logical_resource_estimate.py).
```

Behind `cudaq.estimate`, CUDA-Q Logical takes the kernel's logical intent,
realizes it with the code and gadgets the target names, and reports the cost of
that choice (peak encoded patches, peak protected logical qubits, and the named
gadget calls that make up the estimate). For more examples of CUDA-Q Logical's
features, see the [examples](use-cases/examples.md).

CUDA-Q kernels are only one of the possible entry points. You can also author
portable logical programs directly, define your own codes and gadgets, and place
computations on a logical machine. The compiler refines a program through strict
semantic stages, producing immutable evidence at each transition; see the
[architecture reference](reference/architecture.md) for the advanced staged
model.

## Choose your route

The documentation is organized by the result you want:

| If you want to…                                                            | Go to                                                                                          |
| -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **Install and run a first estimate**                                       | [Getting started](getting-started/quickstart.md)                                               |
| **Understand stages, ownership, and evidence**                             | [How CUDA-Q Logical refines a program](getting-started/how-cudaq-logical-refines-a-program.md) |
| **Do a specific job** (define a code, place, distill, estimate, emit Stim) | [Use cases](use-cases/define-a-code.md)                                                        |
| **Browse runnable end-to-end examples**                                    | [Examples](use-cases/examples.md)                                                              |
| **Inspect advanced internals or build against CUDA-Q**                     | [Reference](reference/architecture.md)                                                         |

```{toctree}
:maxdepth: 2
:hidden:
:caption: Getting started

getting-started/quickstart
getting-started/cudaq-logical-in-practice
getting-started/how-cudaq-logical-refines-a-program
getting-started/concepts
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
use-cases/examples
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
