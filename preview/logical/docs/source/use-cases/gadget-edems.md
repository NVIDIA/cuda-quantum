# Derive an error model from gadgets

A **detector error model (DEM)** lists an experiment's possible faults. For
each fault it gives the detectors and logical observables the fault flips and
the fault's probability. Samplers and decoders take a DEM as input. Stim
derives one by analyzing a complete circuit, such as the one exported in
[Stim emission](stim-emission.md).

CUDA-Q Logical can instead build the model gadget by gadget. An **extended
detector error model (EDEM)** describes the faults of a single gadget while its
encoded input and output remain open. Besides detectors and observables, it
records how each fault changes the error carried into the next gadget. This
lets you:

- derive each gadget's model once and reuse it wherever the gadget repeats;
- compose the models in execution order;
- **project** the result to an ordinary DEM once preparation and readout close
  both ends.

No circuit simulation is involved. For the surface-memory experiment from the
Stim tutorial, the projected model is the DEM that Stim derives from the
exported circuit; the test suite checks that the two agree.

This page first shows what an EDEM records for one gadget. It then derives,
inspects, saves, and samples the surface-memory model. The final section covers
custom faults, derivation one stage at a time, and the checks that run during
derivation.

For a shorter first example with the standard noise policy and physical
ancilla extraction, start with
[Physical surface-code memory](examples/gadget-surface-memory.md). The
custom-noise example on this page uses ideal MPP measurements so the fault
factory can focus on data errors.

## What an EDEM records

Consider a one-qubit H gadget on a bare qubit with no error correction. H turns
an X error into a Z error. The example inserts an X fault at the gadget's input
and derives its EDEM:

```python
import cudaq.logical as cql
from cudaq.logical.analysis import FaultSet, edem_from_gadget


@cql.gadget(implements=cql.logical.h)
def bare_h(block: cql.patch[cql.codes.BareQubit]) -> cql.patch[cql.codes.BareQubit]:
    return cql.h(block.data)


# An empty insertion site denotes the input boundary.
faults = FaultSet().pauli_error("", 0, "x", 0.01)
model = edem_from_gadget(
    bare_h, contract=cql.gadgets.GadgetContract(bare_h), faults=faults)
assert model.apply([1, 0, 0]) == [0, 1]
```

`edem_from_gadget` accepts the gadget directly and compiles its declared
contract internally. This bare gadget has no measurements or stabilizers,
so its explicit contract has no detector or syndrome declarations. For a
custom gadget with measurements, declare those relations as described in
[Declare a contract](../reference/detector-contracts.md#declare-a-contract).

`model.apply` maps the bits entering the gadget to the bits leaving it, over
GF(2):

- **Input `[1, 0, 0]`.** The first bit is the fault indicator: the X fault
  occurred. The remaining bits describe the error already on the incoming
  encoded block. Here the incoming block carries no error.
- **Output `[0, 1]`.** The gadget has no detectors, observables, or success
  rows, so the output is only the error on the outgoing block. `[0, 1]` is a
  logical Z error.

In general, each encoded block on either side contributes three coordinates:

- `ΔV` is the change in the expected stabilizer values, the virtual syndrome.
- `S` is the syndrome of the residual error.
- `L` is the residual error's logical effect, with X and Z coordinates
  interleaved.

The bare code has no stabilizers, so `ΔV` and `S` have width zero and only
`L = (x, z)` remains. The input `L` is `(0, 0)` and the output `L` is `(0, 1)`.
[Virtual syndromes and boundary errors](../reference/detector-contracts.md#virtual-syndromes-and-boundary-errors)
defines these coordinates precisely.

Composition connects these coordinates. One gadget's output error becomes the
next gadget's input error, which is how a fault in one gadget can flip a later
gadget's detectors. A preparation has no encoded input and a destructive
readout has no encoded output. A chain that starts with one and ends with the
other is therefore **closed**: no boundary coordinates remain, and its model
can be projected to a DEM.

## Derive a surface-memory model

This tutorial uses ideal MPP stabilizer measurements and explicit data noise.
The [physical-memory introduction](examples/gadget-surface-memory.md) uses
standard gate, reset and measurement noise. The advanced
[reference comparison](examples/surface-memory-circuit.md) additionally exports
individual gadget models and compares them with Stim's generated
rotated-memory circuit.

Surface-library gadgets carry their declared contracts, so deriving one model
with standard noise needs only the gadget and noise policy:

```python
from cudaq.logical.architectures import surface

implementation = surface.interleaved_nz_definitions(3)
noise = cql.compiler.StandardDepolarizingNoise(
    after_clifford_depolarization=0.001)
memory_model = edem_from_gadget(implementation.syndrome_extraction, noise=noise)
```

For other gadgets, pass `contract=` explicitly or declare a default with
`@cql.gadget(contract=factory)`. Missing contracts raise an error; detector
relations are never inferred. `faults=` supplies an explicit fault set, while
`noise=` and `depolarizations=` generate standard faults. An explicit fault
set cannot be combined with either standard-noise argument.

The [memory experiment](../../../examples/gadgets/03_custom_noise.py)
derives the model for the surface-memory experiment in the Stim tutorial:
prepare logical zero, measure the stabilizers for several rounds, and read out
logical Z. Run it from the repository root:

```bash
python3 preview/logical/examples/gadgets/03_custom_noise.py
```

```text
Exported d=3, rounds=3: 20 detectors, 1 observable, 81 fault indicators
Model: memory-edem/memory.edemb
```

To also sample the model with Stim, see [Sample the model](#sample-the-model).
The following
steps explain the example's `# %%` cells so you can modify the experiment. Run
the full script, or work through its cells interactively.

### Import the libraries

Import CUDA-Q Logical, the surface recipe, and `cudaq_edem`, which provides
composition and DEM operations. It is installed with CUDA-Q Logical:

```{eval-rst}
.. literalinclude:: ../../../examples/gadgets/03_custom_noise.py
   :language: python
   :start-at: from pathlib import Path
   :end-before: # %%
```

### Configure the experiment

The parameter cell controls the experiment size and where noise occurs:

```{eval-rst}
.. literalinclude:: ../../../examples/gadgets/03_custom_noise.py
   :language: python
   :start-at: distance = 3
   :end-before: # %%
```

- `distance` sets the code size; distance three uses nine data qubits.
- `rounds` counts stabilizer rounds, including preparation's. For
  `rounds = 3`, the sequence is preparation, two memory rounds, and logical-Z
  readout.
- `data_noise_kind` chooses `"pauli_channel"` (the default) or
  `"independent_pauli"`.
- `data_pauli_probabilities` supplies the X, Y and Z probabilities per noisy
  data qubit and round. In channel mode they are mutually exclusive
  alternatives; in independent mode each is a separate binary event.
- `noisy_rounds` selects which rounds receive the channel. Round 0 is
  preparation.
- `noisy_data_qubits` restricts the channel to some data qubits; `None` means
  all of them.
- `shots` and `seed` control the optional Stim sampling step. The default
  `shots = 0` skips it.

Stabilizer measurements and readout are ideal; all noise comes from the data
channels. Useful changes include:

- Set `noisy_rounds = (1,)` to apply data noise only before the first memory
  round.
- Set `noisy_data_qubits = (0, 1)` to restrict noise to two data qubits.
- Change `data_pauli_probabilities` to compare symmetric and Z-biased noise.
  The model and saved file represent biased channels exactly. Keep `shots = 0`
  for categorical laws that lack a supported exact factorization. Biased
  independent errors can be sampled directly.
- Increase `distance` or `rounds` to change the experiment size.

### Select the gadget contracts

The model uses the same three recipe gadgets and contracts as the Stim tutorial:

```{eval-rst}
.. literalinclude:: ../../../examples/gadgets/03_custom_noise.py
   :language: python
   :start-at: implementation = surface.definitions(distance)
   :end-before: # %%
```

Each contract declares a gadget's detectors, its incoming and outgoing virtual
syndromes, and, for readout, the logical observable. The Stim tutorial explains
these rules for the surface gadgets in
[Describe which results signal errors](stim-emission.md#describe-which-results-signal-errors).
To use a gadget of your own, put its contract in the sequence; see
[Declare a contract](../reference/detector-contracts.md#declare-a-contract).

### Choose where faults occur

Faults are inserted at **sites** of each gadget's Clifford projection, the
list of physical instructions in its realization. A fault takes effect
immediately *after* the instruction with the given `site_id`. The empty site
`""` means the gadget's input boundary, before its first instruction.

`derive_edem_chain` calls a **fault factory** once for each occurrence, that
is, each position in the contract sequence. The factory receives the
occurrence index, the contract, and the projection, and returns that
occurrence's complete fault set:

```{eval-rst}
.. literalinclude:: ../../../examples/gadgets/03_custom_noise.py
   :language: python
   :start-at: def memory_faults(
   :end-before: # %%
```

The factory finds each gadget's first stabilizer check (`MPP`) and inserts the
noise just before it:

- In preparation, the noise goes after the reset, so the reset does not erase
  it.
- In a memory round, the check is the first instruction, so the noise goes at
  the input boundary.
- Readout receives no faults.

`pauli_channel` adds one channel per data qubit. Its X, Y, and Z alternatives
are mutually exclusive, and each one becomes a **fault indicator**, a column
of the model that is 1 when that fault occurs. With the defaults, there are
9 data qubits × 3 alternatives × 3 noisy rounds = 81 fault indicators.

With `data_noise_kind="independent_pauli"`, the factory instead uses
`pauli_error` for each supplied X, Y and Z rate. Several events can fire on
the same qubit. The same numeric rates therefore describe a different noise
model from mutually exclusive channel alternatives.

For example, the default channel has total depolarizing probability `p`:
its X, Y and Z alternatives each have probability `p / 3`. To represent
that same channel with independent mechanisms, use `factorize_dem`, which
transforms the probabilities. Merely choosing independent mode with three
rates of `p / 3` does not perform that conversion.

For the standard noise policy, you can skip the factory. Pass
`noise=StandardDepolarizingNoise(...)` and per-occurrence `depolarizations=` to
`derive_edem_chain` instead, using the same placements as `compiler.emit_stim`.
The two styles can't be combined. To insert other faults, such as measurement
flips or correlated errors, see [Specify faults](#specify-faults).

### Derive and compose the model

Pass the contracts and the factory to `derive_edem_chain`, then project the
result:

```{eval-rst}
.. literalinclude:: ../../../examples/gadgets/03_custom_noise.py
   :language: python
   :start-at: composed = cql.analysis.derive_edem_chain(
   :end-before: # %%
```

`derive_edem_chain` derives one EDEM for each distinct pair of gadget and
fault set. Here the two memory rounds receive identical faults, so they share
one derived model. Composition then gives every occurrence its own fault
indicators, so faults in different rounds stay independent. Because the chain
starts with a preparation and ends with a readout, it is closed, and
`edem.project_DEM` turns it into a DEM.

### Inspect the model

<!--
% invisible-code-block: python
%
% profiles, projected = load_ql_example(
%     "preview/logical/examples/gadgets/03_custom_noise.py", "profiles",
%     "projected")
-->

The projected model groups its detectors and fault indicators by occurrence:

```python
assert projected.inputs == projected.outputs == []
assert projected.detectors == [4, 8, 8]
assert projected.observables == [1]
assert [fault.size for fault in projected.faults] == [27, 27, 27]
```

- Empty `inputs` and `outputs` mean the model is closed.
- The 4 + 8 + 8 detectors are the same 20 described in the Stim tutorial's
  [Read the exported circuit](stim-emission.md#read-the-exported-circuit).
- Each noisy occurrence contributes 27 fault indicators. Readout has none.

`apply` shows the effect of any combination of faults. Indicator 0 is the X
alternative on data qubit 0 before the first round:

```python
x_error_on_qubit_0 = [1] + [0] * 80
flips = projected.apply(x_error_on_qubit_0)
assert [i for i, bit in enumerate(flips[:20]) if bit] == [0]
assert flips[20:] == [1]
```

This fault fires detector 0, the first-round check `Z0*Z1*Z3*Z4`, and flips the
logical-Z observable. Indicator 2, the Z alternative on the same qubit, flips
nothing, because the X-type checks only set a baseline in the first round.

### Save and reload the model

Save the projected model to `memory-edem/memory.edemb`:

```{eval-rst}
.. literalinclude:: ../../../examples/gadgets/03_custom_noise.py
   :language: python
   :start-at: output = Path(output)
   :end-before: # %%
```

The file keeps each Pauli channel exactly as specified, without approximating it
as independent errors. Loading it requires only `cudaq_edem`, with no gadget
definitions or CUDA-Q Logical installation:

```python
from pathlib import Path
from cudaq_edem import DEM

model = DEM.from_bytes(Path("memory-edem/memory.edemb").read_bytes())
assert model.to_bytes() == projected.to_bytes()
```

### Sample the model

The script's last cell converts the model to a Stim DEM and samples it. To
enable it, install Stim and set `shots` to a positive number, such as `100`:

```bash
python3 -m pip install 'stim>=1.16,<2'
```

```{eval-rst}
.. literalinclude:: ../../../examples/gadgets/03_custom_noise.py
   :language: python
   :start-at: if shots:
```

Stim samples independent errors, so the conversion takes three steps:

1. `factorize_dem` preserves independent groups and replaces supported Pauli
   channels with independent errors that
   have exactly the same distribution.
2. `coalesce_dem_columns` merges errors that flip the same detectors and
   observables.
3. `to_stim_dem` formats the result as a `stim.DetectorErrorModel`.

The exact factorization exists for uniform one- and two-qubit Pauli channels,
with total error probability up to `3/4` and `15/16`, respectively. Other
channels, such as the Z-biased example, make `factorize_dem` raise
`ValueError`. Models with postselection can't be converted.

Each shot reports which detectors fired and whether the logical-Z observable
flipped. These counts are uncorrected; to estimate a logical error rate after
correction, pass the Stim DEM to a decoder.

You can also load the saved file directly in CUDA-Q QEC and evaluate it there.
Support for EDEM files is being added in
[NVIDIA/cudaq-qec#868](https://github.com/NVIDIA/cudaq-qec/pull/868).

<!-- TODO: Replace the pull-request link with the CUDA-Q QEC documentation URL
once NVIDIA/cudaq-qec#868 is merged. -->

## Going further

### Specify faults

A `FaultSet` lists faults explicitly. Each fault is placed at a site from
`projection.instructions[i].site_id` and takes effect immediately after that
instruction. It must target a carrier that is live at that point. Site IDs
belong to a fixed realization, so obtain a new projection after rewriting the
circuit.

Besides Pauli errors, a **record flip** inverts a recorded measurement bit
without adding a physical error. This example flips a memory round's first
check result and adds an X error after its last check:

```python
round_profile = profiles[1]
round_build = cql.compile(round_profile)
round_projection = cql.compiler.project_clifford(
    round_build, gadget=round_profile.gadget.name)
last_check_site = round_projection.instructions[-1].site_id
data_carrier = round_projection.interface.inputs[0].data_carriers[0]
faults = FaultSet().record_flip(round_projection.interface.measurements[0].record, 0.01)
faults = faults.pauli_error(last_check_site, data_carrier, "x", 0.02)
model = edem_from_gadget(
    round_profile.gadget, contract=round_profile, faults=faults)
```

The two faults affect the output boundary differently. The record flip changes
the outgoing virtual syndrome `ΔV` without leaving a physical error, so `S`
does not change. The late X error changes `S` without changing `ΔV`, because no
later check in this gadget measures it.

Faults can be combined in several ways:

- **Independent faults.** Faults outside a group are independent. Each has
  its own indicator, even when two faults have identical effects.
- **Mutually exclusive alternatives.** `FaultSet.pauli_channel`, `depolarize1`,
  and `depolarize2` keep each alternative as a separate indicator, and at most
  one alternative occurs.
- **Correlated events.** An `ElementaryFault` can group several
  `PauliInsertion` values and record flips into one indicator.

Probabilities must be finite and in `[0, 1]`, and mutually exclusive
alternatives must sum to at most one. Fault column `i` corresponds to
`faults.faults[i]`. The packed EDEM doesn't store fault sources, so keep the
ordered fault set if you need to map columns back to faults.

The workflow supports stabilizer codes with `r = 0` (no gauge qubits), Clifford
operations, static calls and repeats, Pauli errors, and record flips. Erasures,
subsystem codes, dynamic feedback, and non-Clifford operations are
unsupported.

### Derive models stage by stage

`derive_edem_chain` packages a sequence of steps you can also run yourself:
derive each gadget's EDEM with its contract and noise policy, then compose the
results:

```python
import cudaq_edem as edem

prep_profile, round_profile, *_, readout_profile = profiles
noise = cql.compiler.StandardDepolarizingNoise()


def gadget_model(profile):
    return cql.analysis.edem_from_gadget(
        profile.gadget, contract=profile, noise=noise)


round_model = gadget_model(round_profile)
chain = edem.compose_chain([
    gadget_model(prep_profile), round_model, round_model,
    gadget_model(readout_profile)
])
assert sum(edem.project_DEM(chain).detectors) == 20
```

The `noise=` argument applies the same noise policy as `compiler.emit_stim`.
Use `faults=` with your own `FaultSet` to model other faults, obtaining a
Clifford projection first when you need instruction sites or record indices.
`compose_chain` supports serial chains in which adjacent boundaries use the
same code presentation, meaning the same code and stabilizer ordering. Call
`project_DEM` only once every encoded boundary is closed.

### Contract checks during derivation

`edem_from_gadget` checks the contract against the realization before deriving a
model. Declared detectors must be zero without faults, and outgoing virtual
syndromes must match the measured stabilizers. The contract must also account
for every deterministic measurement relation. An inconsistent or incomplete
contract is rejected.

Compilation materializes the authored declarations; it does not establish
their correctness. EDEM derivation checks the detector contract, but does not
prove the gadget's intended logical action. Use
[`cql.gadgets.verify_gadget`](../reference/detector-contracts.md#verify-a-contract-and-the-logical-action)
to check both the contract and supported logical behavior.
