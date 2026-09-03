# CUDA-Q Logical in practice

## From logical intent to an inspectable estimate

Start with the result: one portable Bell program becomes a logical resource
estimate, a replayable placement on a logical machine, and a verified
Steane-code realization — and a selected P2 program can be emitted as
standards-compatible Stim circuit text.

The point is not that CUDA-Q Logical has one special path to Stim. The point is
that every fact enters at its owning stage, and each refinement remains
inspectable before anything downstream consumes it.

```text
portable P0 intent  →  P1 placement  →  P2 QEC realization
        │                  │                   │
 logical estimate    slot bindings      static estimate
                                            │
                                       Stim text
```

## Begin with intent

The application asks for a Bell pair. It does not choose a code, a machine, or
an estimation target.

```{eval-rst}
.. literalinclude:: ../../../examples/01_p0_bell.py
   :language: python
   :lines: 13-18
   :caption: Portable P0 intent (examples/01_p0_bell.py).
```

The linear spelling makes ownership visible: `q[0] = ql.h(q[0])` consumes one
version of the qubit and returns the only live successor.

Compiling freezes the program as an immutable P0 build, and the first estimation
tier is already available:

```{eval-rst}
.. literalinclude:: ../../../examples/01_p0_bell.py
   :language: python
   :lines: 21-27
   :caption: P0 estimate (examples/01_p0_bell.py).
```

At P0, CUDA-Q Logical counts logical qubits, actions, and instruments. It cannot
yet claim a syndrome round or an encoded-patch count, because none of those
facts has been supplied.

```text
P0 Bell: 2 logical qubits
```

## Place the program without choosing a code

Placement is a refinement of the P0 build, not a rewrite of the application. A
logical machine declares regions, capabilities, and capacity; a placement
constraint says the `data` qubits stay together.

```{eval-rst}
.. literalinclude:: ../../../examples/02_p1_placement.py
   :language: python
   :lines: 13-21
   :caption: A two-slot logical machine (examples/02_p1_placement.py).
```

```{eval-rst}
.. literalinclude:: ../../../examples/02_p1_placement.py
   :language: python
   :lines: 32-43
```

The P1 build records the placement as inspectable evidence — which region and
slot each logical owner occupies — and replays exactly from its serialization.

```text
P1 Bell: data[0:2] placed on compute[0:2]
```

## Supply a realization, not a rewritten application

The QEC realization lives in its own definitions: a code with checks and logical
operators, and gadgets that claim typed logical behavior.

```{eval-rst}
.. literalinclude:: ../../../examples/03_code_and_gadget.py
   :language: python
   :lines: 13-34
   :caption: The Steane code and a terminal-memory gadget (examples/03_code_and_gadget.py).
```

`implements=terminal_memory` tells the verifier which behavior is claimed. The
patch type tells selection which boundary the realization accepts. A matching
name without those typed facts would not be enough.

Materializing the code and compiling the gadget produces verified P2 objects
whose costs can be inspected directly:

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

| The author supplies                         | CUDA-Q Logical derives and verifies                        |
| ------------------------------------------- | ---------------------------------------------------------- |
| portable logical intent                     | typed logical ownership and stage tracking                 |
| a logical machine with regions and capacity | a replayable placement with exact slot bindings            |
| code checks and a gadget body               | verified code and gadget definitions with operation counts |

## Estimate at the honest tier

Estimation comes in exactly two tiers. `Tier.LOGICAL`, seen above at P0, counts
logical structure. `Tier.STATIC` counts the P2 realization: encoded patches,
gadget calls, and operations. The same ladder is reachable directly from an
ordinary CUDA-Q kernel by compiling through a CUDA-Q Logical target:

```{eval-rst}
.. literalinclude:: ../../../examples/00_cudaq_logical_resource_estimate.py
   :language: python
   :lines: 21-40
   :caption: A distance-3 surface-code target estimated from a CUDA-Q kernel (examples/00_cudaq_logical_resource_estimate.py).
```

```text
CUDA-Q logical-zero resources:
  peak encoded patches: 1
  peak protected logical qubits: 1
  CUDA-Q Logical operation counts: {'alloc': 1, 'call': 2, 'dealloc': 1, 'measure_product': 1, 'prep_z': 1}
  CUDA-Q Logical gadget calls: {'rotated_surface_3_measure_z0': 1, 'rotated_surface_3_prepare_zero': 1}
```

## Emit Stim text at the boundary

A verified P2 entry gadget can be projected to standards-compatible Stim circuit
text — CUDA-Q Logical's secondary interchange path.

<!--
% invisible-code-block: python
%
% steane_memory = load_ql_example(
% "preview/logical/examples/03_code_and_gadget.py", "steane_memory")
-->

```python
build = ql.compile(steane_memory)
emission = ql.lower.emit_stim_artifact(
    build.module, root_symbol=build.root.symbol)
assert emission.text.startswith("R ")
assert emission.interface is not None
```

The Python emitter accepts only a verified P2 entry gadget. Emission fails
closed at the stage boundary rather than guessing at unrealized operations.

:::{admonition} Evidence boundary :class: note

These are logical and static resource estimates of declared codes and machines,
plus a strict Stim text projection. CUDA-Q Logical does not model physical
noise, does not sample or decode detector events, and does not claim
hardware-calibrated counts. The estimates state what the declared realization
costs; they are not simulated executions. :::

## Continue from here

- [How CUDA-Q Logical refines a program](how-cudaq-logical-refines-a-program.md)
  develops stage and facet ownership in more detail.
- [Examples](../use-cases/examples.md) collects every shipped Python example,
  including distillation, Clifford+T synthesis, and the Gidney–Ekerå projection.
