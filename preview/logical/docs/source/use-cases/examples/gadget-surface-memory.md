# Physical Surface-Code Memory

[`gadgets/01_surface_memory.py`](../../../../examples/gadgets/01_surface_memory.py)
stores logical zero using physical ancillas, CNOT gates, and measurements.
It verifies three reusable gadgets, applies standard physical noise, exports
a Stim circuit, and derives the same experiment's detector error model.

This is the second gadget example. Start with
[Verify an authored gadget](gadget-verification.md) to see how a custom gadget's
contract is declared. Here, the Surface recipe supplies the gadgets and their
contracts.

Run from the repository root with CUDA-Q Logical and `cudaq-edem` installed:

```bash
python3 preview/logical/examples/gadgets/01_surface_memory.py
```

The default run writes `surface-memory/memory.stim` and
`surface-memory/memory.edemb` without requiring Stim. Each `# %%` cell can also
be run interactively.

## Select preparation, extraction, and readout

The parameter cell chooses a distance-three code, three syndrome-extraction
rounds, and physical error probability `p = 0.001`:

```{eval-rst}
.. literalinclude:: ../../../../examples/gadgets/01_surface_memory.py
   :language: python
   :start-at: distance = 3
   :end-before: # %%
   :caption: Configure the physical-memory experiment.
```

Select the recipe's ancilla-based interleaved N/Z realization and connect its
contracts in execution order:

```{eval-rst}
.. literalinclude:: ../../../../examples/gadgets/01_surface_memory.py
   :language: python
   :start-at: implementation = surface.interleaved_nz_definitions(distance)
   :end-before: # %%
   :caption: Connect the reusable physical gadgets (examples/gadgets/01_surface_memory.py).
```

`prepare_zero` prepares logical +Z and extracts a syndrome.
`syndrome_extraction` measures the checks while preserving the logical state.
`measure_z_with_syndrome` extracts a syndrome and then destructively measures
the data, returning logical Z. Preparation has no encoded input; readout has
no encoded output. Together they close the experiment's code boundaries.

Both endpoints include an extraction, so `rounds` must be at least two.
At `rounds = 3`, the sequence is:

```text
prepare_zero[SEC] -> syndrome_extraction -> measure_z_with_syndrome[SEC + data MZ]
```

## Verify the ideal behavior

Check each reusable gadget before adding noise:

```{eval-rst}
.. literalinclude:: ../../../../examples/gadgets/01_surface_memory.py
   :language: python
   :start-at: verification = {
   :end-before: # %%
   :caption: Verify preparation, identity extraction, and destructive logical-Z readout.
```

The receipts establish the declared ideal logical actions and deterministic,
complete detector contracts. The readout check includes the parity returned
as its logical measurement result. These stabilizer checks do not establish
fault tolerance or noisy circuit distance; see
[Detector contracts](../../reference/detector-contracts.md#verify-a-contract-and-the-logical-action)
for the supported claims.

## Emit a circuit and derive its model

Use one policy for depolarization after Clifford gates, flips before
measurements, and errors after resets:

```{eval-rst}
.. literalinclude:: ../../../../examples/gadgets/01_surface_memory.py
   :language: python
   :start-at: noise = cql.compiler.StandardDepolarizingNoise(
   :end-before: # %%
```

The same contracts and noise feed both circuit emission and model derivation:

```{eval-rst}
.. literalinclude:: ../../../../examples/gadgets/01_surface_memory.py
   :language: python
   :start-at: text = cql.compiler.emit_stim(
   :end-before: # %%
   :caption: Emit the noisy circuit and derive the composed model.
```

An **extended detector error model (EDEM)** records each gadget's fault effects
on detectors, logical outcomes, and its open code boundaries. This boundary
information lets one gadget's residual error propagate into later gadgets.
`derive_edem_chain` derives and composes these models, reusing a derived model
where the same gadget and fault set repeat. Once preparation and readout close
the boundaries, `project_DEM` produces the experiment's **detector error model
(DEM)**: fault probabilities and the detectors and observables each fault flips.

Try `rounds = 10`. The circuit and composed model grow to cover the additional
extractions, while repeated extraction occurrences reuse the same derived
gadget model. The [EDEM tutorial](../gadget-edems.md) explains the boundary
coordinates and custom fault sets in detail.

## Save and optionally sample

The export cell writes both artifacts:

```{eval-rst}
.. literalinclude:: ../../../../examples/gadgets/01_surface_memory.py
   :language: python
   :start-at: output = Path(output)
   :end-before: # %%
```

`memory.stim` contains the physical circuit, its noise, detectors, and logical
observable. `memory.edemb` contains the projected model with the original
categorical fault laws. Individual depolarizing-channel alternatives remain
mutually exclusive in the saved model.

To sample the physical circuit, install `stim>=1.16,<2` and change `shots` to
a positive number. The final cell uses Stim's detector sampler:

```{eval-rst}
.. literalinclude:: ../../../../examples/gadgets/01_surface_memory.py
   :language: python
   :start-at: if shots:
```

The counts report raw detector events and uncorrected logical-observable
flips. The example does not run a decoder.

For individual gadget-model exports, explicit data depolarization, a Stim
reference comparison, and circuit or DEM sampling, continue to
[Surface-memory reference comparison](surface-memory-circuit.md). The optional
[Stim-emission](../stim-emission.md) and [custom-noise](../gadget-edems.md)
tutorials instead use direct ideal MPP stabilizer measurements.

[Back to examples](index.md)
