# Stim emission

A selected P2 program can leave CUDA-Q Logical as standards-compatible
[Stim](https://github.com/quantumlib/Stim) circuit text. This is an interchange
path: Stim is the assembly language the realization is projected onto — explicit
physical qubits, Clifford operations, and measurements — so you can inspect,
diff, and archive the result, or feed it to any Stim-speaking tool. Emission
consumes a P2 build; it never changes the program's semantics, and it is not an
execution or sampling service.

## Emit from the command line

`qlx-translate` performs the projection. It ships as a console script with the
`cudaq-logical` wheel, so an installed package is enough. This module holds one
round of CSS syndrome extraction on the Steane [[7,1,3]] code, followed by a
destructive data readout:

```bash
qlx-translate preview/logical/examples/mlir/stim_steane_memory.mlir \
  --fabric-to-stim
```

```stim
R 7 8 9
H 7 8 9
CX 7 0 7 1 7 2 7 3 8 0 8 1 8 4 8 5 9 0 9 2 9 4 9 6
H 7 8 9
R 10 11 12
CX 0 10 1 10 2 10 3 10 0 11 1 11 4 11 5 11 0 12 2 12 4 12 6 12
M 7 8 9
M 10 11 12
M 0 1 2 3 4 5 6
```

Thirteen carriers, all explicit. Data occupies 0–6; the X-type ancillas are 7–9
and the Z-type ancillas 10–12, matching the code's `partitions`. Each `CX` line
is one stabilizer row of `hx` or `hz` expanded into carrier pairs — the encoded
`cudaq.logical.extract_syndrome` has become the physical circuit it stands for.

Note what is *not* there: the data carriers are never reset. The gadget takes
its encoded patch as an entry argument, because emission projects a realization
rather than preparing an encoded state. Only the ancillas are reset, and only
because each is a single carrier.

`examples/mlir/stim_memory.mlir` is the degenerate companion — a bare
one-carrier code, whose whole projection is `R 0` then `M 0`.

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
