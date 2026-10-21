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

This export emits the ideal realization. CUDA-Q Logical does not yet attach a
noise model, derive a DEM, or provide sampling and decoding through this path.
For typed native instructions and exact boundary and measurement records, see
[Clifford projection](../reference/detector-contracts.md#project-the-realization).

## Export a profiled surface-code memory

The Python exporter adds detector and observable annotations to a closed,
sequential chain of single-block gadget profiles. It emits plain text directly;
the Stim Python package is not required.

The [surface-memory example](../../../examples/standalone/06_surface_memory.py)
authors three physical gadgets and their detector contracts:

- `prep0` resets the data and ancillas and performs one SEC round. Its Z checks
  supply initial detectors; its X-check outcomes establish virtual syndromes.
- `sec` compares new check records with incoming virtual syndromes and exports
  the updated syndrome bundle.
- `readout_z` performs a final SEC round, then measures every data carrier in Z.
  It declares the temporal and closing Z-check detectors and the logical-Z
  observable. It has no encoded output.

For `rounds >= 2`, the chain is
`prep0 -> sec ** (rounds - 2) -> readout_z`. The extraction uses the four
interleaved N/Z CNOT layers of
[Stim's rotated surface-code memory](https://github.com/quantumlib/Stim/blob/main/src/stim/gen/gen_surface_code.cc).
Each layer couples disjoint carriers. X hooks run horizontally, across the
vertical logical X string; Z hooks run vertically, across the horizontal
logical Z string. This orientation preserves the circuit distance.
Preparation and readout each contribute one extraction round.

Using `surface_memory` from the example:

<!--
% invisible-code-block: python
%
% surface_memory = load_ql_example(
% "preview/logical/examples/standalone/06_surface_memory.py", "surface_memory")
-->

```python
import cudaq.logical as cql

memory = surface_memory(distance=3, rounds=3)
p = 0.001
noise = cql.compiler.StandardDepolarizingNoise(
    after_clifford_depolarization=p,
    before_round_data_depolarization=p,
    before_measure_flip_probability=p,
    after_reset_flip_probability=p,
)
text = memory.to_stim(noise)
assert sum(line.startswith("DETECTOR") for line in text.splitlines()) == 24
assert "OBSERVABLE_INCLUDE(0)" in text
assert "DEPOLARIZE2(0.001)" in text
```

To write the example's output to a file, run from the repository root:

```bash
python preview/logical/examples/standalone/06_surface_memory.py > surface-memory.stim
```

The example can also compile the emitted text with Stim and sample detector
events and observable flips. Install the optional example dependency and pass
`--shots`:

```bash
python -m pip install 'stim>=1.16,<2'
python preview/logical/examples/standalone/06_surface_memory.py --shots 1000 --seed 7 > surface-memory.stim
```

Sampling summaries go to stderr, leaving stdout as circuit text. In Python,
the same handoff uses Stim's
[detector sampler](https://github.com/quantumlib/Stim/blob/main/doc/python_api_reference_vDev.md#stim.Circuit.compile_detector_sampler):

```python
import stim

circuit = stim.Circuit(text)
sampler = circuit.compile_detector_sampler(seed=7)
detectors, observable_flips = sampler.sample(1000, separate_observables=True)
assert detectors.shape == (1000, 24)
assert observable_flips.shape == (1000, 1)
```

Observable flips are uncorrected parity changes; a decoder has not been
applied. Stim is used by the example for simulation, while the compiler's text
exporter remains independent of it.

The surface-memory tests compare the emitted circuit with
`stim.Circuit.generated("surface_code:rotated_memory_z", ...)` at distances
3, 5, and 7. Qubits are matched by coordinates, `REPEAT` and batched gates are
expanded, and `MR` is split into measurement and reset. The comparison checks
operation order on every shared carrier, exact noise probabilities, and the
detector and observable record parities. Disjoint operations may commute;
`TICK` and annotation coordinates are excluded, so this checks circuit content
rather than identical text or timing metadata. Exact categorical fault-law
comparisons provide a separate check at distances 3 and 5.

`StandardDepolarizingNoise` applies depolarization after each physical Clifford
gate and at explicit round-start data sites. Reset and measurement flips are
separate channels. The four probabilities can be set independently.
Depolarizing channels remain categorical: one-qubit noise has three Pauli
alternatives, and two-qubit noise has fifteen. These settings do not change
gadget definitions or profiles.

**Known limitation for permutations:** `fabric.permute` specifies a noiseless
relabeling, but projection currently represents it as ordinary `SWAP`
instructions. With `after_clifford_depolarization` enabled, the exporter adds
`DEPOLARIZE2` after those SWAPs too. The surface-memory example does not use
`fabric.permute`. Preserving the distinction between relabeling and physical
operations through projection and noise policies is future work.

The example calls `cql.compiler.emit_stim(memory.profiles, noise=noise,
round_starts=memory.round_starts)`. The exporter binds physical carriers across
adjacent ports, substitutes output virtual-syndrome formulas into subsequent
detector rows, and resolves stable records to Stim measurement references.
Round-start sites are explicit native instruction IDs because an extraction
round can begin inside a preparation or readout gadget.

The exporter supports complete single-block chains with matching code
presentations. Only the final profile may declare observables; postselection is
unsupported. It checks structure, while ideal contract correctness remains a
separate question. The command-line translation above continues to emit the
selected ideal realization. The output is a circuit text file for downstream
tools.

Measurement-flip noise supports single-qubit Z and X measurements. Applying
that policy to an MPP measurement is rejected; such measurements need an
explicit measurement fault model.
