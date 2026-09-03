# Stim emission

A selected P2 program can leave CUDA-Q Logical as standards-compatible
[Stim](https://github.com/quantumlib/Stim) circuit text. This is an interchange
path: Stim is the assembly language the realization is projected onto — explicit
physical qubits, Clifford operations, and measurements — so the result can be
inspected, diffed, archived, and consumed by any Stim-speaking tool. Emission
consumes a P2 build; it never changes the program's semantics, and it is not an
execution or sampling service.

## From the command line

`qlx-translate` performs the projection natively.
`examples/cli/stim_memory.mlir` is a one-patch memory gadget over the trivial
`[[1,1,1]]` code — prepare Z, measure Z, done:

```bash
qlx-translate preview/logical/examples/cli/stim_memory.mlir --fabric-to-stim
```

```stim
R 0
M 0
```

A closed _protocol_ is first legalized within P2 and then translated, in one
pipe. Saving a compiled protocol build to `protocol.mlir` (via
`build.to_mlir()`) and translating example 04's 15-to-1 distillation:

```bash
qlx-opt protocol.mlir --fabric-lower-protocols='root-symbol=distill_15to1' \
  | qlx-translate --fabric-to-stim
```

```text
error: fabric-to-stim: unsupported operation 'fabric.resource_request'
```

This failure is the boundary working as intended: the distillation protocol asks
for fifteen external raw T states, and Stim circuit text has no way to name an
incoming magic-state stream. Only self-contained circuits — every qubit
allocated and prepared inside the module — project to Stim.

## From Python, with types

The same projection is available as a typed artifact: `qlx.lower.emit_stim`
returns the text, while `qlx.lower.emit_stim_artifact` returns a
`qlx.lower.StimEmission` — the text plus a `CompiledInterfaceManifest` recording
the exact boundary the circuit was projected from. Emitting example 03's
compiled Steane terminal-memory gadget (`examples/03_code_and_gadget.py`):

% invisible-code-block: python % % import runpy % \_memory_mod =
runpy.run_path("preview/logical/examples/03_code_and_gadget.py") % steane_memory
= \_memory_mod["steane_memory"]

```python
import cudaq.logical as qlx

build = qlx.compile(steane_memory)
emission = qlx.lower.emit_stim_artifact(
    build.module, root_symbol=build.root.symbol)
assert emission.text.startswith("R ")
assert emission.interface is not None
```

```stim
R 7 8 9
H 7 8 9
CX 7 0 7 1 7 2 7 3 8 0 8 1 8 4 8 5 9 0 9 2 9 4 9 6
H 7 8 9
...
M 0 1 2 3 4 5 6
```

The encoded `qlx.extract_syndrome` has become the physical circuit it stands
for: ancilla resets, the Steane stabilizer CNOT pattern, ancilla and data
measurements — thirteen qubits, explicit.

## Fail-closed boundaries

Emission refuses rather than approximates:

- the module must pass native MLIR verification;
- the root must be a legalized P2 entry gadget — protocols go through
  `fabric-lower-protocols` first;
- external resource requests (`fabric.resource_request`) are unsupported — Stim
  text cannot name an incoming resource stream, as shown above;
- reset-based preparation is emitted only for a verified trivial one-carrier
  code (`n=1, k=1, r=0`); encoded preparation is rejected;
- recursive Fabric call graphs are rejected.

What Stim cannot express is out of scope for this surface entirely: noise
models, detectors, sampling, and decoding are not part of the trimmed product,
so no DEM or shot-level target exists to emit toward.
