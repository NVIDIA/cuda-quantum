# How CUDA-Q Logical refines a program

A CUDA-Q Logical program describes logical behavior once. Machines, codes,
gadgets, and protocols refine that behavior without being smuggled into the
application.

This walkthrough follows one small program — a Bell pair — through the entire
semantic spine. The workload is intentionally small so the changing facts
remain visible.

## The program stays portable

```{literalinclude} ../../../examples/01_p0_bell.py
:language: python
:lines: 13-18
:caption: P0 intent, unchanged from authoring to compilation (examples/01_p0_bell.py).
```

Nothing in the function says Steane, two compute slots, or distance three. At
P0 those would be guesses, so CUDA-Q Logical does not make them.

The reassignment `q[0] = qlx.h(q[0])` is part of the contract. Quantum values
are linear owners: the operation consumes one version and returns the only
live successor.

`qlx.compile(bell)` freezes the function as an immutable P0 build. Later
stages never retrace the Python program and silently change its meaning; they
refine the build.

## Placement is owned by the machine

```{literalinclude} ../../../examples/02_p1_placement.py
:language: python
:lines: 13-21
:caption: The machine owns regions, capabilities, and capacity (examples/02_p1_placement.py).
```

The machine contributes a `compute` region with two slots and declared
capabilities. None of that policy belongs in `bell`.

```{literalinclude} ../../../examples/02_p1_placement.py
:language: python
:lines: 32-37
```

`qlx.compiler.place` continues the P0 build into a P1 build whose placement
record names the region and slot of every logical owner. This separation is
what makes reuse meaningful: the same program can be placed against another
compatible machine, and the same machine can place many programs.

## A gadget claims the logical action

The QEC realization is a typed claim, not a name-based rewrite rule. The
Steane code declares its CSS checks and logical operators; the gadget's
`implements=` declares the objective it realizes.

```{literalinclude} ../../../examples/03_code_and_gadget.py
:language: python
:lines: 13-34
:caption: Code and gadget definitions carry the P2 facts (examples/03_code_and_gadget.py).
```

`qlx.materialize(Steane)` produces a verified `fabric.code` definition;
compiling the gadget produces a verified `fabric.gadget`. A matching name
without those typed facts would not be enough.

## The stages add facts in one direction

| Stage | New facts it may own | Facts it may not reinterpret |
|---|---|---|
| **P0** logical build | logical actions and value ownership | code or machine selection |
| **P1** placed build | regions, slot bindings, placement witnesses | requested logical behavior |
| **P2** QEC realization | codes, patches, gadget calls, protocol network | logical behavior or placement |

The protocol-network marker attached at P2 is a facet, not another stage.
Facets — `qec_spec`, `qec_realization`, `protocol_network`, `patch_graph` —
are orthogonal verified facts that coexist beside one immutable stage root.

## Evidence is part of the result

```{literalinclude} ../../../examples/03_code_and_gadget.py
:language: python
:lines: 41-45
```

The assertions are not incidental test scaffolding. They state the observable
contract of the example: the materialized code and the compiled gadget appear
in the P2 IR, and the declared checks and distance are exactly what selection
sees.

When a required fact is absent, CUDA-Q Logical stops at the latest honest
boundary or raises a typed diagnostic. It does not invent a gadget, assume a
placement, or estimate a tier whose facts were never supplied.

## A practical reading rule

When reading or writing `cudaq.logical` code, ask two questions at every
line:

1. Which stage or facet owns this fact?
2. Did the user supply it, or can CUDA-Q Logical derive and verify it?

That rule explains most of the public surface. Programs own intent; machines
own placement; code and gadget libraries own reusable realizations; compiler
passes derive later-stage artifacts; emitters consume only the verified stage
and facets they advertise.

The [CUDA-Q Logical in practice](cudaq-logical-in-practice.md) walkthrough
shows the same discipline ending at a static estimate and emitted Stim text.
The [example gallery](../example-gallery/index.md) covers the full shipped
set.
