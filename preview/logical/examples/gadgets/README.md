# Gadget verification and error-model examples

These examples use `cudaq.logical` directly to check reusable gadgets and
analyze their physical realizations. Start with 00 and 01. They introduce
gadget verification and the standard circuit-to-error-model workflow without
requiring the standalone compiler sequence.

1. `00_verify_gadget.py` authors an idle syndrome-extraction gadget on the
   Repetition code, declares its detector contract, and verifies both the
   contract and the ideal logical action. It then inspects a failed receipt
   for the same realization with an incorrect logical claim. See
   [Verify an authored gadget](../../docs/source/use-cases/examples/gadget-verification.md).
2. `01_surface_memory.py` selects physical ancilla/CNOT surface-code gadgets,
   verifies preparation, syndrome extraction, and readout, then emits their
   noisy circuit and derives their composed detector error model. It exports
   `surface-memory/memory.stim` and `surface-memory/memory.edemb`. See
   [Physical surface-code memory](../../docs/source/use-cases/examples/gadget-surface-memory.md).
3. `02_stim_emission.py` is an optional ideal-MPP experiment: it exports
   direct stabilizer measurements with explicit data noise. See
   [Stim emission](../../docs/source/use-cases/stim-emission.md).
4. `03_custom_noise.py` is an optional custom-noise experiment on the same
   ideal-MPP realization. Its `FaultSet` factory selects noisy rounds and
   data qubits and specifies categorical Pauli channels or independent Pauli
   errors. See [Derive an error model from gadgets](../../docs/source/use-cases/gadget-edems.md).
5. `04_surface_memory_reference.py` is the advanced physical-memory example.
   It derives and saves individual gadget EDEMs, reuses the extraction model,
   adds explicit data depolarization, and compares the composed DEM with
   Stim's rotated-memory reference. It also offers circuit and DEM samplers.
   See [Surface-memory reference comparison](../../docs/source/use-cases/examples/surface-memory-circuit.md).

Run from the repository root:

```bash
python3 preview/logical/examples/gadgets/00_verify_gadget.py
python3 preview/logical/examples/gadgets/01_surface_memory.py
```

The general standalone introduction remains
[`standalone/00_logical_program.py`](../standalone/00_logical_program.py): it
authors and estimates a logical Bell program. Gadget example 00 instead
authors one encoded realization and checks its declared meaning.

Every example uses `# %%` cells. The memory experiments include editable
parameters. In physical-memory examples 01 and 04, preparation and readout
each include an extraction, so `rounds` must be at least two. Try `rounds = 10`
to repeat the extraction gadget while reusing its derived model. Ideal-MPP
examples 02 and 03 allow one
round because their direct logical readout contains no extraction.

Example 00 needs no simulator dependency. Stim is optional for memory examples
01–03 with `shots = 0`. To enable sampling, install `stim>=1.16,<2` and set
`shots` to a positive number. Example 01 samples the physical circuit; the
counts are raw detector events and uncorrected logical-observable flips.
Example 02 prints the circuit to stdout and sampling
summaries to stderr, so it can be redirected:

```bash
python3 preview/logical/examples/gadgets/02_stim_emission.py > surface-memory.stim
```

Example 03 samples its model through an exact independent-error conversion;
categorical channels require a supported exact factorization. Example 04
requires Stim for its default reference comparison. Set
`compare_with_stim = False` and `shots = 0` to export without it. Its
`sampling_mode = "circuit"` samples physical depolarizing channels;
`sampling_mode = "dem"` samples the exactly factorized independent DEM.
The saved `.edemb` files retain their original categorical fault laws.
