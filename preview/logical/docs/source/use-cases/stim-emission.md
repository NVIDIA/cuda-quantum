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

The [surface-memory example](../../../examples/standalone/06_surface_memory.py)
prepares one encoded logical qubit in the zero state, repeatedly measures its
stabilizers, and reads out logical Z. A **stabilizer check** measures a product
of Pauli operators on selected data qubits. Changes in check results can reveal
errors while preserving the encoded logical information.

The exported Stim circuit contains the quantum operations plus two kinds of
annotations:

- A **detector** identifies a parity of measurement results that should be zero
  without errors. A nonzero parity is a detection event.
- A **logical observable** identifies the measurement result of interest: here,
  the final logical-Z readout.

The example groups operations into **gadgets** and attaches a **detector
profile** to each gadget. A profile declares detector parities, the check
information passed to the next gadget, and any logical observables. These
annotations let Stim interpret the circuit's measurement results.

Use an environment with CUDA-Q Logical installed. Stim is needed only for the
optional sampling step. The file's `# %%` cells also let you run the example
interactively.

### Import the library

Import the compiler, the built-in surface recipe, and the lattice geometry:

```{eval-rst}
.. literalinclude:: ../../../examples/standalone/06_surface_memory.py
   :language: python
   :start-at: import sys
   :end-before: # %%
```

### Define detector profiles

The example uses three gadgets from the built-in surface recipe. They operate
on a typed **patch**, the collection of physical qubits holding the encoded
state:

- `prepare_zero` resets the data to zero and measures all stabilizers.
- `memory_round` measures the stabilizers again.
- `measure_z` measures logical Z and consumes the patch.

The `detector_profile` helper takes a gadget, its role in the experiment
(`kind`), and the code definition (`code`). `code.hx` and `code.hz` list the
data-qubit indices for each X and Z stabilizer. Preparation and memory rounds
name their measurement results
`x_check0.outcome`, `x_check1.outcome`, and so on for the rows of `code.hx`,
then `z_check0.outcome`, `z_check1.outcome`, and so on for `code.hz`.
The helper's `records` dictionary holds symbolic references to these results
in X-check then Z-check order.

A measured stabilizer has eigenvalue +1 or -1, recorded as bit 0 or 1.
Successive measurements of the same stabilizer should agree without errors.
Their XOR, written `previous ^ current`, is therefore a detector: it is zero
when the results agree and one when they differ.

Preparation needs a different comparison. Resetting the data to zero makes the
initial Z-check results zero, so each can serve as a detector by itself. Initial
X-check results can be random even without errors. They establish the baseline
for later comparisons.

A gadget's profile carries this baseline as a **virtual syndrome**: the
expected stabilizer signs inferred from the available measurement records.
The next gadget can reference it through its input interface. In this example,
`inputs.only()` and `outputs.only()` select the single encoded block at each
boundary; its `.syndrome` property names the virtual-syndrome bundle.
The helper uses `ProfileGraph` to declare both the comparisons and the outgoing
baseline:

| Expression | Purpose |
| --- | --- |
| `gadget.record("z_check0.outcome")` | Refer symbolically to a named measurement result. |
| `ProfileGraph(gadget.interface)` | Start a profile for the gadget's encoded inputs and outputs. |
| `graph.detector(parity)` | Declare a parity expected to be zero without errors. |
| `graph.bind(..., tuple(records.values()))` | Bind the outgoing syndrome to the ordered check results. |
| `graph.freeze()` | Produce the `GadgetProfile` consumed by the compiler. |

These declarations add no quantum operations. For `kind="prep"`, the helper
adds detectors for the initial Z checks. For memory rounds (`kind="sec"`), it
compares every incoming check result with the corresponding current result.
Both branches bind all current results as the outgoing syndrome, preserving
the dictionary's X-check then Z-check order. For `kind="readout"`, it freezes
the graph without adding detectors or output bindings: the gadget consumes the
patch, and its declared result automatically supplies the logical observable.

```{eval-rst}
.. literalinclude:: ../../../examples/standalone/06_surface_memory.py
   :language: python
   :start-at: def detector_profile(
   :end-before: # %%
```

### Define data-noise placements

Before every stabilizer round, the example independently applies a
one-qubit depolarizing channel to each data qubit. With probability
`data_depolarization`, the channel applies X, Y, or Z with equal probability;
otherwise it leaves the qubit unchanged. The following helper specifies where
to insert these channels.

For preparation and memory-round profiles, `data_noise_placements` projects
the gadget into physical instructions and finds its first stabilizer
measurement (`MPP` in the projection). A `Depolarize1Placement` names that
instruction's site, the data-qubit indices to affect, and the error probability.
The returned dictionary maps each occurrence's index in `profiles` to its
placements, so repeated memory rounds each receive their own noise layer.
The helper skips terminal readout, which has no output patch.

```{eval-rst}
.. literalinclude:: ../../../examples/standalone/06_surface_memory.py
   :language: python
   :start-at: def data_noise_placements(
   :end-before: # %%
```

### Configure and assemble the experiment

`distance` selects the code size, with `distance ** 2` data qubits; use an odd
integer of at least three. `rounds` counts all stabilizer-measurement rounds,
including the one in preparation. Choose at least two rounds to compare
successive measurements.

`data_depolarization = p` sets the data-error probability before each round.
The all-zero `StandardDepolarizingNoise()` policy adds no other noise, so
measurements remain ideal. `shots = 0` selects export only; `seed` controls the
optional sampler.

```{eval-rst}
.. literalinclude:: ../../../examples/standalone/06_surface_memory.py
   :language: python
   :start-at: distance = 3
   :end-before: # %%
```

Select the distance-specific code and gadgets with `surface.definitions`.
The lattice supplies data-qubit coordinates for the exported circuit:

```{eval-rst}
.. literalinclude:: ../../../examples/standalone/06_surface_memory.py
   :language: python
   :start-at: implementation = surface.definitions(distance)
   :end-before: # %%
```

Create a profile for each gadget using `detector_profile`, assemble the profile
sequence, and call `data_noise_placements` for that sequence:

```{eval-rst}
.. literalinclude:: ../../../examples/standalone/06_surface_memory.py
   :language: python
   :start-at: prep = detector_profile(
   :end-before: # %%
```

The sequence starts with `prepare_zero`, repeats `memory_round` a further
`rounds - 1` times, and ends with `measure_z`. Preparation contributes the first
round; the final memory round immediately precedes readout. With the default
distance and round count, there are four initial Z-check detectors and eight
comparisons for each of two later rounds: 20 detectors in total, plus one
logical observable.

### Export the circuit

`compiler.emit_stim` connects the patches between successive profiles and
substitutes the outgoing syndrome bindings into the next gadget's detector
expressions. It translates named measurement records into Stim `rec[...]`
references, emits `DETECTOR` annotations, and marks the logical readout with
`OBSERVABLE_INCLUDE(0)`. The example also supplies qubit coordinates:

```{eval-rst}
.. literalinclude:: ../../../examples/standalone/06_surface_memory.py
   :language: python
   :start-at: coordinates =
   :end-before: # %%
```

Run from the repository root to save the circuit. The default `shots = 0`
needs no Stim Python installation:

```bash
python preview/logical/examples/standalone/06_surface_memory.py > surface-memory.stim
```

The [gadget EDEM tutorial](gadget-edems.md#compare-circuit-and-per-gadget-dem-construction)
uses the same experiment to compare a circuit-derived detector error model
with one assembled from per-gadget models.

### Sample detection events

To simulate the exported circuit, install the optional dependency:

```bash
python -m pip install 'stim>=1.16,<2'
```

Set `shots = 1000` in the parameter cell and run the example. Stim's detector
sampler returns one row per shot, with a column for each detector and each
logical observable. At the defaults, the arrays have shapes `(1000, 20)` and
`(1000, 1)`:

```{eval-rst}
.. literalinclude:: ../../../examples/standalone/06_surface_memory.py
   :language: python
   :start-at: if shots:
```

The summary counts shots with detection events and shots with a logical-Z
flip. These are uncorrected results; estimating a decoded logical error rate
requires a decoder. Summaries go to stderr, so redirected stdout remains valid
Stim circuit text.

### Exporter constraints

The Python exporter handles a single encoded patch from preparation through
destructive readout. Adjacent profiles must agree on the code's stabilizer and
logical-operator definitions. Each gadget that outputs a patch must bind its
complete syndrome bundle, and only the final profile may declare observables.
Postselection, which accepts only selected measurement outcomes, is unsupported.
Profile construction checks the declarations' structure;
[detector-contract validation](../reference/detector-contracts.md#declare-a-contract)
is a separate analysis of their ideal correctness.

When extending the noise model, note that the standard measurement-flip policy
supports single-qubit X and Z measurements. Pauli-product measurements require
an explicit measurement fault model. Also, `fabric.permute` currently projects
to `SWAP` instructions: enabling `after_clifford_depolarization` adds noise to
those instructions even though the source operation denotes a noiseless
relabeling.
