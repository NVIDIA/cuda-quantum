# Command-line workflows

**Outcome.** The Python-free path through the same stories: `qlx-opt` runs
the native passes as explicit MLIR pass pipelines, and `qlx-translate` emits
Stim text from a verified P2 gadget. One command produces a typed logical
estimate and a P1 refinement of the Bell fixture while retaining the original
P0 program:

```bash
qlx-opt preview/logical/examples/cli/logical_and_placement.mlir \
  --pass-pipeline='builtin.module(qlx-estimate-logical{root=bell result=logical},qlx-to-lvm{root=bell domain=machine result=bell_placed})'
```

A second command attaches a static P2 resource estimate to the memory gadget:

```bash
qlx-opt preview/logical/examples/cli/static.mlir \
  --pass-pipeline='builtin.module(fabric-count{root=memory device=device result=static})'
```

And the terminal Stim text emitter runs from `qlx-translate`:

```bash
qlx-translate preview/logical/examples/cli/stim_memory.mlir --fabric-to-stim
```

A closed protocol is first legalized within P2
(`qlx-opt --fabric-lower-protocols='root-symbol=main'`) and then piped through
the same translator. `qlx-opt --help` lists every registered pass.

**Evidence boundary.** The native `qlx-to-lvm` pass is a deterministic
first-fit subset of the richer Python placement solver: it requires an
explicit logical-machine domain, rejects calls that have not been inlined,
and fails if control-flow joins disagree on slots — the two placers are not
claimed to be implementation-equivalent. `fabric-to-stim` accepts a verified
P2 entry gadget and rejects non-Clifford operations. Python remains
authoritative for the typed `StimEmission` result; the native translator is
the command-line text counterpart.

## Canonical sources

`logical_and_placement.mlir` — the P0 Bell fixture for the logical estimate
and P1 placement:

```{literalinclude} ../../../examples/cli/logical_and_placement.mlir
:language: mlir
```

`static.mlir` — the P2 fixture for the static resource estimate:

```{literalinclude} ../../../examples/cli/static.mlir
:language: mlir
```

`stim_memory.mlir` — the verified P2 memory gadget emitted as Stim text:

```{literalinclude} ../../../examples/cli/stim_memory.mlir
:language: mlir
```

[Back to the gallery](../examples)
