# The logical programming stack

This guide uses a few small examples to show how CUDA-Q Logical turns an idea
into something you can place on fault-tolerant hardware and estimate. In this
short walk-through, you will write and place a Bell-pair program, define a
quantum-error-correction (QEC) realization, and inspect its outputs.

CUDA-Q Logical formalizes the layers of abstraction involved in this process as
**P0**, **P1**, and **P2**:

```text
P0: logical program  →  P1: placement  →  P2: QEC realization
       │                       │                    │
 logical estimate          region and slots         static estimate
                                                        │
                                                    Stim text
```

Each compilation stage adds detail without changing the behavior of the original
program.

## 1. Write the logical program

Start with a program that prepares and measures a Bell pair:

```{eval-rst}
.. literalinclude:: ../../../examples/01_p0_bell.py
   :language: python
   :lines: 13-27
   :caption: Define, compile, and estimate a Bell program (examples/01_p0_bell.py).
```

The program says what to compute, but not where to place the qubits or which QEC
code to use. That makes it portable.

The assignments are important. A quantum value has one live owner, so an
operation consumes the current value and returns its successor. For example,
`q[0] = ql.h(q[0])` replaces the old value of `q[0]` with the one returned by
`h`. This rule prevents stale or duplicated quantum values from reaching the
compiled program.

`ql.compile(bell)` produces an immutable P0 build. At this point, a logical
estimate can count the program's logical qubits and operations:

```text
P0 Bell: 2 logical qubits
```

It cannot yet count encoded patches or syndrome rounds because you have not
chosen a realization. Those figures become available later, when the compiler
has the facts needed to calculate them.

## 2. Place the qubits on a logical machine

Next, describe the available logical machine. This one has a `compute` region
with two slots and supports logical computation and measurement:

```{eval-rst}
.. literalinclude:: ../../../examples/02_p1_placement.py
   :language: python
   :lines: 13-21
   :caption: Define a two-slot logical machine (examples/02_p1_placement.py).
```

Keep this information out of the Bell program. The same program can then be
placed on another compatible machine, and the same machine can host other
programs.

Compile the program and ask the placement solver to keep its two `data` qubits
together:

```{eval-rst}
.. literalinclude:: ../../../examples/02_p1_placement.py
   :language: python
   :lines: 32-43
   :caption: Place the Bell program and inspect the result.
```

The resulting P1 build records the region and slot assigned to each logical
value:

```text
P1 Bell: data[0:2] placed on compute[0:2]
```

Placement refines the P0 build; it does not retrace or rewrite the Python
program. The recorded placement is also replayable, as the final assertion in
the example demonstrates.

## 3. Define a QEC realization

At P2, codes and gadgets describe how to realize logical behavior. The next
example defines the Steane code, an objective for terminal memory, and a gadget
that implements that objective:

```{eval-rst}
.. literalinclude:: ../../../examples/03_code_and_gadget.py
   :language: python
   :lines: 13-34
   :caption: Define a Steane code and terminal-memory gadget (examples/03_code_and_gadget.py).
```

The code supplies its CSS checks, logical operators, and distance. The gadget
works on a typed Steane patch, extracts its syndrome, measures its data qubits,
and ends their lifetime.

`implements=terminal_memory` is a checked claim about the gadget's behavior.
CUDA-Q Logical compares the gadget with the objective using their types and
derived actions; a matching Python name alone is not enough.

Materialize the code and compile the gadget to inspect the verified P2
definitions and their operation counts:

```{eval-rst}
.. literalinclude:: ../../../examples/03_code_and_gadget.py
   :language: python
   :lines: 37-45
```

```text
Steane [[7,1,3]] terminal-memory gadget:
  physical data qubits per logical block: 7
  independent X/Z stabilizer checks: 6
  logical Z support: (0, 1, 2, 3, 4, 5, 6)
  authored operations: {'reset': 2, 'h': 2, 'cx': 2, 'read_syndrome_ancillas': 1, 'mz': 1, 'dealloc': 1}
```

The assertions are useful beyond testing the example: they show which code and
gadget reached the P2 representation and which code properties were available to
selection.

## 4. Estimate a selected realization

CUDA-Q Logical offers two estimation tiers:

| Tier        | Available from | What it counts                                      |
| ----------- | -------------- | --------------------------------------------------- |
| **LOGICAL** | P0             | logical qubits, actions, and instruments            |
| **STATIC**  | P2             | encoded patches, gadget calls, and authored actions |

You can also enter this pipeline from a regular CUDA-Q kernel. The following
example selects a distance-3 surface-code target and asks `cudaq.estimate` for
the cost of preparing and measuring a logical zero:

```{eval-rst}
.. literalinclude:: ../../../examples/00_cudaq_logical_resource_estimate.py
   :language: python
   :lines: 13-40
   :caption: Estimate a CUDA-Q kernel with a surface-code target (examples/00_cudaq_logical_resource_estimate.py).
```

```text
CUDA-Q logical-zero resources:
  peak encoded patches: 1
  peak protected logical qubits: 1
  CUDA-Q Logical operation counts: {'alloc': 1, 'call': 2, 'dealloc': 1, 'measure_product': 1, 'prep_z': 1}
  CUDA-Q Logical gadget calls: {'rotated_surface_3_measure_z0': 1, 'rotated_surface_3_prepare_zero': 1}
```

These numbers describe the selected logical realization. They are not results
from a noise simulation or a hardware-calibrated execution.

## 5. Emit a verified gadget as Stim

A verified P2 entry gadget can also be projected to Stim circuit text:

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
assert emission.text.startswith("R ")
assert emission.interface is not None
```

The emitter accepts a verified P2 entry gadget. If the required realization is
missing, it stops at that boundary instead of filling in an implementation.

Stim emission is an interchange format, not another compilation stage. The
output contains the explicit resets, Clifford operations, and measurements in
the selected gadget; CUDA-Q Logical does not use it to sample or decode detector
events.

## What each stage owns

Each kind of information belongs to a specific stage:

| Stage                  | Adds                                          | Does not change               |
| ---------------------- | --------------------------------------------- | ----------------------------- |
| **P0** logical build   | logical actions and value ownership           | —                             |
| **P1** placed build    | regions, slot bindings, placement evidence    | requested logical behavior    |
| **P2** QEC realization | codes, patches, gadgets, and protocol details | logical behavior or placement |

Some verified facts sit alongside a stage rather than extending this sequence.
CUDA-Q Logical calls them _facets_. Code specifications, gadget realizations,
protocol networks, and patch graphs are all facets that downstream tools can
request and inspect.

Builds retain this evidence and can be serialized and replayed. When a required
fact is absent, CUDA-Q Logical reports the missing requirement rather than
choosing a machine, code, or gadget on your behalf.

When reading or writing CUDA-Q Logical code, two questions are usually enough to
orient yourself:

1. Which stage or facet owns this fact?
2. Did I supply it, or can CUDA-Q Logical derive and verify it?

## Where to go next

- Work through the task-oriented guides for
  [defining a code](../use-cases/define-a-code.md),
  [placing a program](../use-cases/devices-and-placement.md), and
  [estimating resources](../use-cases/estimation.md).
- Browse the complete set of runnable
  [examples](../use-cases/examples/index.md), including distillation, Clifford+T
  synthesis, and the Gidney–Ekerå projection.
- Read [Core concepts](concepts.md) for linear ownership, evidence, and
  implementation discovery in more detail.
- See the [architecture reference](../reference/architecture.md) for compiler
  stages, dialects, and pipelines.
