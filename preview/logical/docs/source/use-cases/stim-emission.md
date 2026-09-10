# Stim emission

A selected P2 program can leave CUDA-Q Logical as standards-compatible
[Stim](https://github.com/quantumlib/Stim) circuit text. This is an interchange
path: Stim is the assembly language the realization is projected onto — explicit
physical qubits, Clifford operations, and measurements — so you can inspect,
diff, and archive the result, or feed it to any Stim-speaking tool. Emission
consumes a P2 build; it never changes the program's semantics, and it is not an
execution or sampling service.

## Emit from Python

The projection comes as a typed artifact: `ql.lower.emit_stim` returns the
text, while `ql.lower.emit_stim_artifact` returns a `ql.lower.StimEmission` —
the text plus a `CompiledInterfaceManifest` recording the exact boundary the
circuit was projected from. The snippet below emits the standalone compiled
Steane terminal-memory gadget (`examples/standalone/02_code_and_gadget.py`):

<!--
% invisible-code-block: python
%
% steane_memory = load_ql_example(
% "preview/logical/examples/standalone/02_code_and_gadget.py", "steane_memory")
-->

```python
import cudaq.logical as ql

build = ql.compile(steane_memory)
emission = ql.lower.emit_stim_artifact(
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

The encoded `ql.extract_syndrome` has become the physical circuit it stands for:
ancilla resets, the Steane stabilizer CNOT pattern, ancilla and data
measurements — thirteen qubits, explicit.

## Fail-closed boundaries

Emission refuses rather than approximates:

- the module must pass native MLIR verification;
- the root must be a legalized P2 entry gadget — protocols go through
  `fabric-lower-protocols` first;
- external resource requests (`fabric.resource_request`) are unsupported — Stim
  text cannot name an incoming resource stream;
- reset-based preparation is emitted only for a verified trivial one-carrier
  code (`n=1, k=1, r=0`); encoded preparation is rejected;
- recursive Fabric call graphs are rejected.

Stim cannot express noise models, detectors, sampling, or decoding. The
trimmed product does not include them, so no DEM or shot-level target exists to
emit toward.
