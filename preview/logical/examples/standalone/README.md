# Standalone CUDA-Q Logical examples

These examples use `cudaq.logical` directly. They do not define
`@cudaq.kernel` functions and do not call `cudaq.estimate`. Instead, they make
the logical authoring, placement, QEC selection, physical lowering, scheduling,
direct estimation, and `cudaq.run` APIs visible.

Examples 00–05 progress from a portable logical program to a complete physical
schedule. The Stim-export and EDEM examples analyze a surface-code memory
experiment. Two execution examples demonstrate ideal computation and noisy
T-state distillation.

1. `00_logical_program.py` authors a portable Bell program without a CUDA-Q
   kernel and counts its logical resources.
2. `01_logical_placement.py` places a Bell program on a two-slot logical
   machine.
3. `02_code_and_gadget.py` defines the Steane code and a syndrome-extraction
   gadget, then inspects the gadget's static operations.
4. `03_magic_state_distillation.py` defines a 15-to-1 T-state distillation
   protocol and estimates its static resources.
5. `04_physical_schedule.py` compiles a logical memory experiment onto a
   surface-code device and lowers it to physical events.
6. `05_gidney_ekera_lookup_addition.py` combines these steps on one four-bit
   Gidney--Ekerå lookup-addition circuit, from logical program to physical
   schedule.
7. `06_stim_emission.py` exports a surface-code memory experiment as an
   annotated Stim circuit. See
   [Stim emission](../../docs/source/use-cases/stim-emission.md).
8. `07_memory_edem.py` derives the same experiment's detector error model
   gadget by gadget and saves it to `memory-edem/memory.edemb`. See
   [Derive an error model from gadgets](../../docs/source/use-cases/gadget-edems.md).
9. `08_surface_memory_circuit.py` selects `surface.interleaved_nz_definitions`
   for gadgets with the ancilla-based interleaved N/Z gate schedule
   with circuit-level noise. It exports the circuit and reusable gadget EDEMs,
   then compares their DEM with Stim's generated rotated-memory reference. See
   [Circuit-level surface memory](../../docs/source/use-cases/examples/surface-memory-circuit.md).

Run any example from the repository root:

```bash
python3 preview/logical/examples/standalone/00_logical_program.py
```

`06_stim_emission.py` prints the circuit to stdout and any sampling summary to
stderr, so you can redirect the circuit to a file:

```bash
python3 preview/logical/examples/standalone/06_stim_emission.py > surface-memory.stim
```

The Stim-export and EDEM examples can also sample their results with Stim.
Sampling is off by default; to enable it, install `stim>=1.16,<2` and set `shots`
to a positive number in the parameter cell. `07_memory_edem.py` offers
categorical Pauli channels and independently specified Pauli errors. Sampling
categorical channels requires a supported exact factorization.

Examples 06 and 07 use ideal MPP stabilizer measurements and explicit data
noise. Example 08 includes the physical ancilla circuits, reset and measurement
noise, and final data-parity detectors. Its reference comparison requires
`stim>=1.16,<2`; set `compare_with_stim=False` and `shots=0` to export without
Stim.

In example 08, `sampling_mode="circuit"` samples the physical depolarizing
channels, while `sampling_mode="dem"` samples an exactly factorized independent
DEM of the same noise model. The saved `.edemb` values retain their original
categorical laws in either mode.

The execution examples provide two additional paths:

- `06_steane_bernstein_vazirani.py` recovers a hidden bit string with Steane
  encoded qubits on Stim.
- `07_t_factory_evaluation.py` evaluates T-state distillation with non-Clifford
  gates, measurement feedback, and raw-state noise on `BareQubit` patches.

The encoded example uses an explicit Clifford preparation circuit. It does not
claim fault-tolerant preparation or decoded error correction.
