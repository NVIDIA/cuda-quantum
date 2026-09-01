# cudaq.logical command-line workflows

`qlx-opt` is the command-line compiler entry point. These commands reproduce
the corresponding Python workflow outcomes while making
the complete native MLIR pass pipeline explicit.

Build the tools, then add the build's `bin` directory to `PATH`. From the
repository root:

```bash
qlx-opt preview/logical/examples/cli/logical_and_placement.mlir \
  --pass-pipeline='builtin.module(qlx-estimate-logical{root=bell result=logical},qlx-to-lvm{root=bell domain=machine result=bell_placed})'
```

This produces a typed logical `qlx.estimate_result` and an `lvm.kernel` P1
refinement while retaining the original P0 program.

The native `qlx-to-lvm` pass is a deterministic first-fit subset of the richer
Python placement solver: it requires an explicit logical virtual machine (LVM)
domain, rejects calls that have not been inlined, and fails if control-flow
joins do not agree on slots. The example stays inside the shared supported
subset; the two placers are not claimed to be implementation-equivalent.

```bash
qlx-opt preview/logical/examples/cli/static.mlir \
  --pass-pipeline='builtin.module(fabric-count{root=memory device=device result=static})'
```

This produces a verified static P2 resource-estimate result. The
separate P2 Stim fixture exercises the terminal text emitter:

```bash
qlx-translate preview/logical/examples/cli/stim_memory.mlir --fabric-to-stim
```

The native translator accepts a verified P2 entry gadget. A closed protocol is
first legalized, still within P2, and then translated:

```bash
qlx-opt protocol.mlir --fabric-lower-protocols='root-symbol=main' \
  | qlx-translate --fabric-to-stim
```

`qlx-opt --help` is the authoritative list of registered cudaq.logical passes. Python
remains authoritative for typed `StimEmission`; the native translator is the
command-line text counterpart.
