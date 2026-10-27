# Stim emission

[Stim](https://github.com/quantumlib/Stim) is a fast stabilizer-circuit
simulator, and its `.stim` text format is a common way to exchange
quantum error-correction experiments. A `.stim` file holds the physical
operations and noise. It also holds annotations that say which measurement
parities signal errors and which result is the logical outcome. Exporting from
CUDA-Q Logical lets you analyze gadgets you built and checked here with
Stim and the tools around it, such as samplers, error-model extraction, and
decoders.

This page covers two ways to export:

- **From Python (the main tutorial).** Assemble a surface-code memory
  experiment from built-in gadgets: prepare logical zero, store it for several
  rounds, and read out logical Z. Export it with detectors, a logical
  observable, and data noise. You end with `surface-memory.stim` and,
  optionally, sampled detection events and logical-Z flips.
- **From an MLIR module.** Convert an existing gadget to its ideal circuit with
  `qlx-translate`. See [Native MLIR emission](#native-mlir-emission).

## Export a profiled surface-code memory

With CUDA-Q Logical installed, run the
[surface-memory example](../../../examples/standalone/06_stim_emission.py)
from the repository root:

```bash
python3 preview/logical/examples/standalone/06_stim_emission.py > surface-memory.stim
```

The default experiment uses a distance-three code and three rounds of
stabilizer measurements. Before each round, every data qubit receives an error
with probability `0.001`. The script prints the circuit to stdout, so the
command saves it to `surface-memory.stim`. To also sample the circuit, see
[Sample and interpret the results](#sample-and-interpret-the-results). The
sampling summary goes to stderr, so it stays out of the circuit file.

### Read the exported circuit

An abridged `surface-memory.stim` looks like this:

```stim
QUBIT_COORDS(1, 1) 0
# ... coordinates for data qubits 1-8
R 0 1 2 3 4 5 6 7 8
DEPOLARIZE1(0.001) 0
# ... the same channel on data qubits 1-8
MPP X0*X1
MPP X1*X2*X4*X5
MPP X3*X4*X6*X7
MPP X7*X8
MPP Z0*Z1*Z3*Z4
MPP Z2*Z5
MPP Z3*Z6
MPP Z4*Z5*Z7*Z8
# ... two more rounds of noise and the same eight checks
MPP Z0*Z1*Z2
DETECTOR rec[-21]
# ... rec[-20] through rec[-18]
DETECTOR rec[-25] rec[-17]
# ... 15 more comparisons between consecutive rounds
OBSERVABLE_INCLUDE(0) rec[-1]
```

Reading from the top:

- `QUBIT_COORDS` places the nine data qubits on the lattice for Stim's
  diagrams.
- `R` resets every data qubit to `|0⟩`.
- Each `DEPOLARIZE1(0.001)` is an injected data error, applied before each
  round.
- Each `MPP` is one ideal stabilizer measurement: a Pauli-product measurement
  made directly on the data qubits, with no ancilla qubits. A round measures
  four X-type and four Z-type checks, eight results in all.
- `MPP Z0*Z1*Z2` reads out logical Z.
- The `DETECTOR` and `OBSERVABLE_INCLUDE` annotations follow the last
  measurement. `rec[-k]` is the *k*-th most recent result. The circuit has 25
  results (three rounds of eight plus the readout), so `rec[-25]` is the first
  check of round one and `rec[-17]` is the first check of round two.
  `DETECTOR rec[-25] rec[-17]` therefore says that those two results should
  agree.

The experiment has 20 detectors:

- **Four from round one.** A reset to `|0…0⟩` fixes every Z-type stabilizer to
  +1, so each first-round Z-check result (`rec[-21]` through `rec[-18]`) is a
  detector by itself. X-type checks give random results on `|0…0⟩`, so in round
  one they only establish a baseline.
- **Eight from round two**, each comparing a check with its round-one result.
- **Eight from round three**, each comparing a check with its round-two result.

The single `OBSERVABLE_INCLUDE(0)` marks the logical-Z readout.

The following steps follow the example's `# %%` cells and show where each part
of the circuit comes from. You can run the full script, or work through its
cells interactively.

### Choose the experiment size and noise

Start with the parameter cell:

```{eval-rst}
.. literalinclude:: ../../../examples/standalone/06_stim_emission.py
   :language: python
   :start-at: distance = 3
   :end-before: # %%
```

- `distance` sets the code size: distance three uses nine data qubits. Choose
  an odd integer of at least three.
- `rounds` counts stabilizer-measurement rounds, including preparation's first
  round. Increasing it stores the qubit longer and exposes it to more noise.
- `p` is the probability of a data error on each qubit before each round.
  X, Y, and Z errors are equally likely. Set it to zero for a noiseless check.
- `shots` is the number of sampled repetitions. The default, zero, only exports
  the circuit; `seed` makes sampling reproducible for a fixed circuit.

The stabilizer measurements are ideal. With its defaults,
`StandardDepolarizingNoise()` adds no gate, reset, or measurement noise. The
example's explicit data errors are the only noise in the circuit.

### Build preparation, storage, and readout

Select the surface code at your chosen distance and its qubit coordinates:

```{eval-rst}
.. literalinclude:: ../../../examples/standalone/06_stim_emission.py
   :language: python
   :start-at: implementation = surface.definitions(distance)
   :end-before: # %%
```

The experiment uses three **gadgets**, each a reusable operation on the encoded
qubit:

- `prepare_zero` resets the data and measures the stabilizers once, which
  prepares logical zero.
- `memory_round` measures the stabilizers again while preserving the stored
  logical value.
- `measure_z` reads out logical Z and finishes the experiment.

Stabilizer measurements check for errors without reading out the stored
logical value. An error between two rounds changes some check results, and
comparing consecutive rounds reveals the change.

### Describe which results signal errors

A gadget specifies only quantum operations. To annotate the circuit, the
exporter also needs to know which parities of results should be zero without
errors and which result is the logical outcome. A **gadget profile** attaches
these rules to a gadget:

- **Detectors** are parities of measurement results that are zero when no
  error occurs. Each becomes a `DETECTOR`, and a nonzero value in a shot is a
  *detection event*.
- **Virtual syndromes** are the stabilizer values a gadget expects at its
  boundary. They are inferred from earlier results, not measured at the
  boundary. A gadget reads an *incoming* virtual syndrome from the gadget
  before it and binds an *outgoing* one for the gadget after it. This lets a
  detector compare results measured by two different gadgets.
- **Logical observables** are parities that form a logical measurement outcome,
  here the logical-Z readout. They come from the readout gadget's outcome map;
  see [Detector map and outcome map](../reference/detector-contracts.md#detector-map-and-outcome-map).

The memory round shows how these fit together. For each check `i`, it
declares the detector `incoming[i] ^ round[i]`, because in an ideal round the
new result equals the expected value. It then binds its own results as the
outgoing virtual syndrome, so they become the next round's baseline. In the
profile API, these steps are one `graph.detector(incoming[i] ^ record)` call
per check followed by `graph.bind(outgoing, records)`.
[Declare a contract](../reference/detector-contracts.md#declare-a-contract)
works through a complete Steane memory round.

The three surface profiles produce the detectors described earlier:

- `prepare_zero` declares the four first-round Z-check detectors and binds all
  eight results as its outgoing virtual syndrome.
- `memory_round` compares each of its eight results with the incoming virtual
  syndrome and passes its own results on.
- `measure_z` declares the logical observable.

`surface.gadget_profile(gadget)` returns these profiles. It recognizes only
gadget objects created by `surface.definitions`. Any other gadget raises
`ValueError`, even one with the same name or code. See
[Recipe-provided profiles](../reference/detector-contracts.md#recipe-provided-profiles).

The reference pages also use the term *detector contract*: a gadget's
detectors and virtual-syndrome bindings, together with the code at each
boundary. A profile is the container that holds the contract and the
observables.

### Add data errors and assemble the experiment

The script's `data_noise_placements` helper places an independent depolarizing
channel on every data qubit immediately before each round's first stabilizer
measurement. In preparation, this is after the reset, so the reset does not
erase the injected errors. The final readout receives no additional noise.

Select the profiles in execution order and apply the helper:

```{eval-rst}
.. literalinclude:: ../../../examples/standalone/06_stim_emission.py
   :language: python
   :start-at: prep = surface.gadget_profile(implementation.prepare_zero)
   :end-before: # %%
```

Preparation contributes the first round, so `rounds = 3` adds two memory rounds
before the readout.

### Save the Stim circuit

`compiler.emit_stim` combines the profiles into one annotated circuit. It
accepts a **closed single-block chain**:

- the first profile prepares one encoded block from nothing;
- each interior profile takes that block in and passes it on;
- the last profile consumes it with a measurement.

Adjacent profiles must use the same code presentation, meaning the same code
and stabilizer ordering, so that each outgoing virtual syndrome lines up with
the next gadget's incoming one. `emit_stim` substitutes the outgoing virtual
syndrome into the next gadget's detectors. That substitution turns
`incoming[i] ^ round[i]` into a comparison of two concrete results, such as
`DETECTOR rec[-25] rec[-17]`. Only the last profile may declare observables, and
postselection is unsupported.

`emit_stim` checks the chain's structure. It does not check that the declared
detectors are deterministic, so a profile you write yourself needs separate
verification (see [Customize the experiment](#customize-the-experiment)).

The example passes the profiles and noise placements to `emit_stim` and adds
qubit coordinates for visualization:

```{eval-rst}
.. literalinclude:: ../../../examples/standalone/06_stim_emission.py
   :language: python
   :start-at: coordinates =
   :end-before: assert sum(
```

### Sample and interpret the results

To enable the script's sampling step, install Stim and set `shots` to a
positive number, such as `100`:

```bash
python3 -m pip install 'stim>=1.16,<2'
```

Run the example again. Each **shot** simulates one complete repetition of the
experiment. Stim returns two arrays:

- `detectors` records which detectors fired: one row per shot and one column
  per detector.
- `observables` records whether the final logical-Z value flipped relative
  to the noiseless experiment: one row per shot and one column here.

With `shots = 100`, their shapes are `(100, 20)` and `(100, 1)`. The sampling
step is:

```{eval-rst}
.. literalinclude:: ../../../examples/standalone/06_stim_emission.py
   :language: python
   :start-at: if shots:
```

The summary counts shots with any detection event and shots with a logical-Z
flip. These results are uncorrected. To measure how well error correction
protects the stored qubit, pass the detection events to a decoder. Stim can
derive the detector error model that decoders take as input with
`circuit.detector_error_model()`.

Try setting `p = 0`: both counts should be zero. Then restore the noise and
increase `shots` to explore its effects. Individual shots are random, so small
samples need not show every kind of error.

### Customize the experiment

- **Change the parameters.** `distance`, `rounds`, `p`, and `shots` can be
  changed independently. The detector count scales as
  4 + 8 × (`rounds` − 1) at distance three. In general, it is the number of
  Z-type checks plus `distance² − 1` comparisons for each memory round.
- **Use your own gadgets.** Recipe profiles exist only for the recipe's
  gadgets, so use preparation, memory, and readout gadgets from the same
  `surface.definitions(distance)` result. For a gadget you author,
  [declare its profile](../reference/detector-contracts.md#declare-a-contract)
  before exporting. The sequence must still form a closed chain on one encoded
  block.
- **Add circuit-level noise.** The `StandardDepolarizingNoise` fields add
  noise to physical gates, resets, and measurements. Two limits apply:
  - The surface recipe measures stabilizers with `MPP`. Setting
    `before_measure_flip_probability` raises `ValueError`, because
    measurement-flip noise on a Pauli-product measurement needs an explicit
    fault model.
  - `after_clifford_depolarization` also adds noise after `SWAP`s that come
    from `fabric.permute`, even though those represent noiseless relabeling.
    This surface example has no permutations, but gadgets that use them
    receive extra noise.

## Native MLIR emission

The Python export starts from gadget profiles. It knows which results to
compare, carries virtual syndromes from one gadget to the next, and adds noise.
`qlx-translate --fabric-to-stim` starts instead from an MLIR module, written by
hand or saved from an earlier compile, and converts one of its gadgets to Stim
directly. The output is the **ideal circuit only**: resets, Clifford gates,
and measurements, with no detectors, observables, or noise. Use it to inspect
what a gadget physically does, or as the start of a Stim analysis where you
add the annotations yourself.

This section uses a few terms from the compiler's stages:

- A **P2** module is the QEC-realization stage, where codes and gadgets are
  chosen (see [Concepts](../getting-started/concepts.md)). An **entry gadget**
  is the gadget marked `{entry}`, which the command converts.
- **Carriers** are the physical qubits of a realization. They become Stim
  qubit indices.
- A code's **partitions** name groups of its carriers, such as `data`, `sx`
  (X-check ancillas), and `sz` (Z-check ancillas).

### Emit from the command line

`qlx-translate` ships as a console script with the `cudaq-logical` wheel, so an
installed package is enough. This module holds one round of CSS syndrome
extraction on the Steane [[7,1,3]] code, followed by a destructive data readout:

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

The circuit uses thirteen carriers, all of them explicit. They follow the
code's partitions:

- Data qubits are 0–6.
- X-check ancillas are 7–9.
- Z-check ancillas are 10–12.

Each `CX` line expands the three stabilizer rows of `hx` or `hz` into carrier
pairs. Unlike the surface recipe's ideal `MPP` checks, this module writes out
the syndrome-extraction circuit explicitly: the ancilla resets, the `H` and
`CX` gates, and the measurements.

The gadget takes its encoded patch as an entry argument and assumes the data
carriers are already initialized. Its circuit resets only the ancillas. To
simulate a particular encoded input state, prepend that state's preparation to
this circuit.

`examples/mlir/stim_memory.mlir` is the smallest case: a bare one-carrier
code whose whole circuit is `R 0` then `M 0`.

### Fail-closed boundaries

When the command can't convert a module faithfully, it stops with an error. It
never guesses a missing implementation. It requires that:

- the module passes native MLIR verification;
- the root is a legalized P2 entry gadget. Run `fabric-lower-protocols` on
  protocols first;
- the module has no external resource requests (`fabric.resource_request`).
  Stim text can't name an incoming resource stream;
- `fabric.prep_z` and `fabric.prep_x` appear only on a verified trivial
  one-carrier code (`n=1, k=1, r=0`). Encoded preparation must be lowered to
  explicit physical operations, as the surface recipe in the Python tutorial
  does;
- the Fabric call graph has no recursion.

The command doesn't derive an error model, sample, or decode. To get
the same instructions as typed Python objects, with exact boundary and
measurement records, see
[Project the realization](../reference/detector-contracts.md#project-the-realization).

## Where to go next

- To derive an error model directly from the gadgets and sample it in CUDA-Q
  QEC, continue with
  [Derive an error model from gadgets](gadget-edems.md).
- To write and verify profiles for your own gadgets, read
  [Detector contracts and Clifford projection](../reference/detector-contracts.md).
- To map common Stim workflows onto CUDA-Q Logical, see
  [For Stim users](../reference/for-stim-users.md).
