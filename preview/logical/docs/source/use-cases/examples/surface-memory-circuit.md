# Surface-Memory Reference Comparison

[`gadgets/04_surface_memory_reference.py`](../../../../examples/gadgets/04_surface_memory_reference.py)
extends the [physical-memory introduction](gadget-surface-memory.md) with
individual gadget-model exports, explicit data depolarization, and a comparison
against Stim's generated rotated-memory reference. It also offers physical
circuit and independent-DEM sampling.

Start with gadget examples 00 and 01 for the basic verification and model
workflow. Optional examples 02 and 03, covered by the
[Stim-emission](../stim-emission.md) and [custom-noise](../gadget-edems.md)
tutorials, instead use ideal MPP stabilizer measurements with explicit data
noise. This advanced example uses physical ancillas, four interleaved N/Z
CNOT layers, reset and measurement noise, and final data-parity detectors.

## Run the example

With CUDA-Q Logical and `cudaq-edem` installed, install the optional reference
simulator and run from the repository root:

```bash
python3 -m pip install 'stim>=1.16,<2'
python3 preview/logical/examples/gadgets/04_surface_memory_reference.py
```

The default output is:

```text
Verified prep, SEC, readout: ideal logical actions and detector contracts
d=3, rounds=3: 24 detectors, 1 observable, 1307 original fault indicators
Artifacts: surface-memory-circuit
219 DEM error terms match the circuit and Stim reference (atol=1e-12)
```

Edit the parameter cell, or run the `# %%` cells interactively:

```{eval-rst}
.. literalinclude:: ../../../../examples/gadgets/04_surface_memory_reference.py
   :language: python
   :start-at: distance = 3
   :end-before: # %%
   :caption: Configure the physical-memory reference comparison.
```

Set `compare_with_stim = False` and `shots = 0` to export the circuit and packed
models without importing Stim. Change `rounds` to `10` to exercise model reuse
across a longer memory experiment.

## Select the concrete gadgets

Select preparation, syndrome extraction (SEC), and destructive Z readout,
then connect their code boundaries:

```{eval-rst}
.. literalinclude:: ../../../../examples/gadgets/04_surface_memory_reference.py
   :language: python
   :start-at: implementation = surface.interleaved_nz_definitions(distance)
   :end-before: # %%
   :caption: Construct the gadget sequence before verifying its contracts.
```

The interleaved definition set contains `prepare_zero`, `syndrome_extraction`,
and `measure_z_with_syndrome`, together with their code and physical-qubit
coordinates. `surface.gadget_contract(gadget)` supplies each detector contract.
The separate `surface.definitions(distance)` collection uses ideal-MPP
syndrome rounds.

Each physical gadget includes one extraction. At `rounds = 3`, the sequence is:

```text
prepare_zero[SEC] -> syndrome_extraction -> measure_z_with_syndrome[SEC + data MZ]
```

This experiment requires at least two rounds because its preparation and
readout both contain an SEC. The ideal-MPP examples allow one round because
their direct logical readout contains no SEC. Interleaved readout measures all
data qubits, compares their Z-check parities with its extraction records, and
forms logical Z from the data-measurement parity. At distance three, the circuit
uses nine data qubits and eight check ancillas. With `r` extraction rounds, it
declares `r * (distance**2 - 1)` detectors and one logical observable.

## Verify the ideal behavior

Before adding noise, verify each reusable gadget once:

```{eval-rst}
.. literalinclude:: ../../../../examples/gadgets/04_surface_memory_reference.py
   :language: python
   :start-at: verification = {
   :end-before: # %%
```

The receipts establish logical +Z preparation, an identity syndrome round, and
destructive logical Z readout, including its returned measurement parity.
They also establish deterministic and complete detector contracts. These are
ideal stabilizer checks; noisy circuit distance is checked separately. See
[Verify a contract and the logical action](../../reference/detector-contracts.md#verify-a-contract-and-the-logical-action)
for the supported claims and failure receipts.

## Project the realization and choose physical noise

Compile and project each reusable gadget to choose physical fault sites:

```{eval-rst}
.. literalinclude:: ../../../../examples/gadgets/04_surface_memory_reference.py
   :language: python
   :start-at: builds = {
   :end-before: # %%
```

`StandardDepolarizingNoise` applies depolarization after Clifford gates,
measurement-flip errors before measurements, and reset errors after resets.
This example also places data depolarization at the recipe's declared
`before_syndrome` site:

```{eval-rst}
.. literalinclude:: ../../../../examples/gadgets/04_surface_memory_reference.py
   :language: python
   :start-at: noise = cql.compiler.StandardDepolarizingNoise(
   :end-before: # %%
```

`surface.data_depol(contract, projection, probability)` returns ordinary
`Depolarize1Placement` objects. The helper checks the supplied projection
against the registered realization without compiling it again. It rejects
unknown sites, mismatched projections, and gadgets without the requested site.
The example chooses the probabilities and noisy occurrences.

These data placements follow initialization resets and precede syndrome
extraction. Readout's data noise precedes its SEC, rather than its final
measurements. A direct readout without SEC has no `before_syndrome` site; the
compiler does not infer extraction boundaries from gadget names.

## Emit the circuit and reuse gadget models

The same contracts and physical noise placements feed circuit emission and
EDEM derivation:

```{eval-rst}
.. literalinclude:: ../../../../examples/gadgets/04_surface_memory_reference.py
   :language: python
   :start-at: coordinates =
   :end-before: # %%
   :caption: Derive each gadget EDEM once and compose its occurrences.
```

`edem_from_gadget` accepts each gadget and its explicit contract directly. The
projections above are used to choose faults; callers do not need a compiled
Build for EDEM derivation.

An EDEM retains fault effects on an open gadget's code boundaries, allowing a
fault to influence later gadgets' detectors. The example derives preparation,
SEC, and readout models once, then reuses the SEC model at every intermediate
round. `compose_chain` preserves distinct fault indicators and exact
categorical laws for each occurrence. Preparation and destructive readout
close the chain's code interfaces, allowing `project_DEM` to produce a DEM.

At `rounds = 10`, the eight intermediate SEC occurrences still share one
derived SEC model. The composed model gains each occurrence's detectors and
fault indicators, without repeating the gadget derivation. The
[EDEM tutorial](../gadget-edems.md) explains these boundary coordinates and the
shorter `derive_edem_chain` API used in example 01.

## Export the original models

Save the circuit, projected DEM, and reusable gadget EDEMs before any noise
conversion:

```{eval-rst}
.. literalinclude:: ../../../../examples/gadgets/04_surface_memory_reference.py
   :language: python
   :start-at: output = Path(output)
   :end-before: # %%
```

The `output` directory contains:

| File | Content |
| --- | --- |
| `memory.stim` | Emitted physical circuit with noise, detectors, and logical Z |
| `memory.edemb` | Closed projected DEM with original categorical fault laws |
| `prep.edemb`, `sec.edemb`, `mz.edemb` | Reusable gadget EDEMs with their code interfaces |
| `reference.stim` | Stim's generated rotated-memory circuit |
| `circuit.dem` | DEM obtained from the emitted circuit |
| `composed.dem` | Formatted, exactly converted and coalesced gadget-derived DEM |

The last three files are written when `compare_with_stim = True`. The packed
models retain their original fault laws and response columns. The Surface
recipe supplies the gadgets and contracts directly, so another composer can
reuse them without importing this example.

## Compare with Stim and choose a sampler

The reference comparison explicitly converts the categorical DEM to supported
independent mechanisms, coalesces equal complete responses, and formats the
result for Stim:

```{eval-rst}
.. literalinclude:: ../../../../examples/gadgets/04_surface_memory_reference.py
   :language: python
   :start-at: if compare_with_stim or
   :end-before: # %%
```

`factorize_dem` converts the uniform Pauli channels exactly, with transformed
mechanism probabilities. `coalesce_dem_columns` merges equal complete
responses. `to_stim_dem` formats that transformed model; it performs no
probability conversion or column merging.

In the original EDEM, a depolarizing channel's X/Y/Z or two-qubit alternatives
are mutually exclusive, with the total probability split among three or
fifteen nonidentity Paulis. Reusing those alternative probabilities as
independent rates would change the model. The exact conversion above preserves
the model's distribution, and the saved packed files precede that conversion.
The [custom-noise tutorial](../gadget-edems.md#choose-where-faults-occur) explains
independently specified Pauli errors as a separate modeling choice.

To sample, set `shots` to a positive number and choose `sampling_mode`:

- `"circuit"` samples the physical circuit with its depolarizing channels.
- `"dem"` samples the exactly factorized independent DEM derived from the
  composed EDEMs. Factorization runs even when the reference comparison is
  disabled; unsupported exact conversions raise an error.

```{eval-rst}
.. literalinclude:: ../../../../examples/gadgets/04_surface_memory_reference.py
   :language: python
   :start-at: if shots:
```

Both samplers use the same physical noise model. The reported detector-event
and observable-flip counts are uncorrected; the example does not run a decoder.

## Physical schedule and reference equivalence

The standard-library calls correspond to Stim's generated reference as follows:

| Stim `surface_code:rotated_memory_z` option | Example configuration |
| --- | --- |
| `distance` | `surface.interleaved_nz_definitions(distance)` |
| `rounds` | Preparation, `rounds - 2` SEC occurrences, SEC-plus-readout |
| `after_clifford_depolarization` | Same field of `StandardDepolarizingNoise` |
| `before_measure_flip_probability` | Same field of `StandardDepolarizingNoise` |
| `after_reset_flip_probability` | Same field of `StandardDepolarizingNoise` |
| `before_round_data_depolarization` | `surface.data_depol` at each occurrence's `before_syndrome` site |

The final two CNOT interactions put X hooks horizontally and Z hooks
vertically, transverse to the same-Pauli logical strings. Regression tests
check the circuit's graphlike fault distance at distances three, five, and
seven. This evidence is separate from the ideal verification receipts.

Readout resets the check ancillas after their final measurements. These
terminal resets are part of this concrete schedule and match the reference
circuit's reset fault sites. Their errors have zero response because the
ancillas no longer interact with data.

Physical operations, noise probabilities, and detector/observable parities
match the reference. Text layout, qubit numbering, ticks, and annotation
coordinates can differ. Tests allow disjoint operations to commute and
normalize these representation choices before comparing per-qubit operation
order and parity rows. They also compare exact categorical fault-response laws
before independent conversion.

[Back to examples](index.md)
