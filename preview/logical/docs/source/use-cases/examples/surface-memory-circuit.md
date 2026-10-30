# Circuit-Level Surface-Code Memory

[`standalone/08_surface_memory_circuit.py`](../../../../examples/standalone/08_surface_memory_circuit.py)
constructs a rotated surface-code Z-memory experiment from three reusable
gadgets: preparation, syndrome extraction (SEC), and destructive Z readout.
It emits a noisy physical circuit and independently derives and composes the
gadgets' EDEMs. The resulting DEM is compared with both the emitted circuit's
DEM and Stim's generated rotated-memory reference.

This recipe uses physical ancillas, four interleaved N/Z CNOT layers, and
ancilla measurements. The final two interactions put X hooks horizontally
and Z hooks vertically, transverse to the same-Pauli logical strings. The
regression tests check the circuit's graphlike fault distance at distances
three, five, and seven.

The [Stim-emission tutorial](../stim-emission.md) and
[EDEM tutorial](../gadget-edems.md) retain their simpler ideal-MPP realization
with explicit data noise. This example additionally exercises the physical
extraction schedule, gate noise, reset noise, measurement noise, and final
data-parity detectors.

## Run the comparison

With cudaq-logical and cudaq-edem installed, install the optional reference
simulator and run from the repository root:

```bash
python3 -m pip install 'stim>=1.16,<2'
python3 preview/logical/examples/standalone/08_surface_memory_circuit.py
```

The default output is:

```text
d=3, rounds=3: 24 detectors, 1 observable, 1307 original fault indicators
Artifacts: surface-memory-circuit
219 DEM error terms match the circuit and Stim reference (atol=1e-12)
```

Edit the parameter cell or run the cells interactively to change `distance`,
`rounds`, the three standard noise probabilities, `data_depolarization`,
`output`, `shots`, `seed`, and `sampling_mode`. Set
`compare_with_stim=False` and `shots=0` to export the circuit and packed models
without importing Stim. The recipe itself has no Stim dependency.

## Reuse the concrete gadgets

Select the gadgets with the interleaved N/Z gate schedule, then assemble the
experiment explicitly:

```python
from cudaq.logical.architectures import surface
from cudaq.logical.compiler import (
    StandardDepolarizingNoise, compile, emit_stim, project_clifford,
)

implementation = surface.interleaved_nz_definitions(3)
prep = surface.gadget_contract(implementation.prepare_zero)
sec = surface.gadget_contract(implementation.syndrome_extraction)
readout = surface.gadget_contract(implementation.measure_z_with_syndrome)
# Three syndrome rounds: one in each gadget.
contracts = (prep, sec, readout)
projections = {
    contract.gadget: project_clifford(compile(contract))
    for contract in (prep, sec, readout)
}
noise = StandardDepolarizingNoise(.001, .001, .001)
placements = {
    index: surface.data_depol(contract, projections[contract.gadget], .001,
                             site="before_syndrome")
    for index, contract in enumerate(contracts)
}
coordinates = "\n".join(
    f"QUBIT_COORDS({x}, {y}) {q}"
    for q, (x, y) in enumerate(implementation.coordinates))
text = coordinates + "\n" + emit_stim(
    contracts, noise=noise, depolarizations=placements)
```

The interleaved definition set contains `prepare_zero`, `syndrome_extraction`,
and `measure_z_with_syndrome`, together with their code and physical qubit
coordinates. Their detector contracts are available individually through
`surface.gadget_contract(gadget)`. The existing `surface.definitions(distance)`
returns the full Surface collection with ideal-MPP syndrome rounds.

The standard-library calls correspond to Stim's generated reference as follows:

| Stim `surface_code:rotated_memory_z` option | Example configuration |
| --- | --- |
| `distance` | `surface.interleaved_nz_definitions(distance)` |
| `rounds` | Preparation, `rounds - 2` SEC occurrences, SEC-plus-readout |
| `after_clifford_depolarization` | Same field of `StandardDepolarizingNoise` |
| `before_measure_flip_probability` | Same field of `StandardDepolarizingNoise` |
| `after_reset_flip_probability` | Same field of `StandardDepolarizingNoise` |
| `before_round_data_depolarization` | `surface.data_depol` at each occurrence's `before_syndrome` site |

Physical operations, noise probabilities and detector/observable parities match
the reference. Text layout, qubit numbering, ticks and annotation coordinates
can differ; the tests normalize these representation choices before comparing
per-qubit operation order and parity rows.

Each circuit gadget includes one extraction. Therefore `rounds=3` means:

```text
prepare_zero[SEC] -> syndrome_extraction -> measure_z_with_syndrome[SEC + data MZ]
```

The example requires at least two rounds because its chosen preparation and
readout each contain an SEC. The ideal-MPP examples allow one round because
their direct logical readout contains no SEC. Interleaved readout
measures all data qubits, compares their Z-check parities with its extraction
records, and forms logical Z from the data-measurement parity. At distance
three, the circuit uses nine data qubits and eight check ancillas. With `r`
extraction rounds, it declares `r * (distance**2 - 1)` detectors and one
logical observable.

The readout also resets the check ancillas after their final measurements.
These terminal resets are part of this concrete physical schedule and match
the reference circuit's reset fault sites; their errors have zero response
because the ancillas no longer interact with data.

The gadgets and contracts are shared by the example and its tests. Consumers
such as a window composer can import the three gadgets and contracts directly
from the Surface recipe, without importing this example.

## Physical noise and fault columns

`StandardDepolarizingNoise` applies depolarization after Clifford gates,
measurement-flip errors before measurements, and reset errors after resets.
The example exposes all three probabilities independently and uses `0.001`
for each by default. The same policy and data placements feed circuit emission
and EDEM derivation.

`surface.data_depol(contract, projection, probability)` returns ordinary
`Depolarize1Placement` objects for the gadget's declared `before_syndrome`
site. The helper checks the supplied projection against the registered physical
realization and does not compile it again. It rejects unknown sites, mismatched
projections and gadgets without the requested site. The example chooses the
probability and the noisy occurrences.

The data placements precede syndrome extraction and follow initialization
resets. Readout's extra data noise precedes its SEC, rather than its final
destructive measurements. A direct readout without SEC has no
`before_syndrome` site. The compiler never infers rounds from gadget names.

The example derives each of the three gadget EDEMs once, then reuses the SEC
model at repeated occurrences. Native `compose_chain` preserves distinct fault
indicators and exact categorical laws for each occurrence. `project_DEM`
requires the chain's code interfaces to be closed.

For comparison, native `factorize_dem` converts the uniform Pauli channels
exactly into independent mechanisms. Native `coalesce_dem_columns` merges
equal complete responses. `to_stim_dem` only formats that transformed model;
it performs no probability conversion or column merging.

Circuit simulation retains the physical `DEPOLARIZE1` and `DEPOLARIZE2`
channels. In the original EDEM, their alternatives remain mutually exclusive:
the total channel probability is split among three or fifteen nonidentity
Paulis. The independent DEM is an explicit, exactly converted representation
of the same noise model, with transformed mechanism probabilities.

The [EDEM tutorial](../gadget-edems.md#choose-where-faults-occur) also shows
independently specified Pauli noise as a separate model choice. Reusing the
categorical alternative probabilities as independent rates would change the
model; it is not the conversion performed by `factorize_dem`.

The tests additionally compare physical per-qubit operation order and detector
parities against Stim's generated circuit, allowing disjoint operations to
commute and ignoring ticks and annotation coordinates. Exact categorical
fault-response laws are checked before independent conversion.

## Exported artifacts

The `output` directory contains:

| File | Content |
| --- | --- |
| `memory.stim` | Emitted physical circuit with noise, detectors and logical Z |
| `memory.edemb` | Closed projected DEM with original categorical fault laws |
| `prep.edemb`, `sec.edemb`, `mz.edemb` | Reusable gadget EDEMs with their code interfaces |
| `reference.stim` | Stim's generated rotated-memory circuit |
| `circuit.dem` | DEM obtained from the emitted circuit |
| `composed.dem` | Formatted, exactly converted and coalesced gadget-derived DEM |

The last three files are written when `compare_with_stim=True`. The packed
models are saved before factorization and coalescing, so they retain their
original fault laws and response columns.

Set `shots` to a positive number and choose `sampling_mode`:

- `"circuit"` samples the physical circuit with its depolarizing channels.
- `"dem"` samples the exactly factorized independent DEM derived from the
  composed gadget EDEMs. Factorization runs even when the reference comparison
  is disabled. Unsupported exact conversions raise an error.

Both modes use the same physical noise model and leave the original packed
fault laws unchanged. The reported detector-event and observable-flip counts
are uncorrected; the example does not run a decoder.

```{literalinclude} ../../../../examples/standalone/08_surface_memory_circuit.py
:language: python
```

[Back to examples](index.md)
