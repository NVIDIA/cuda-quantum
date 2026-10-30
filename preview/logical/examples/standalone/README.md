# Standalone CUDA-Q Logical examples

These examples use `cudaq.logical` directly. They do not define
`@cudaq.kernel` functions and do not call `cudaq.estimate`. Instead, they make
the logical authoring, placement, QEC selection, physical lowering, scheduling,
direct estimation, and `cudaq.run` APIs visible.

Examples 00–03 introduce the compiler from a portable logical program to a
complete physical schedule. Continue with surface-memory analysis (04–05),
encoded execution (06), or factory protocols and evaluation (07–08).
Example 09 is an optional capstone combining synthesis, factories, estimation,
and scheduling.

1. `00_logical_program.py` authors a portable Bell program without a CUDA-Q
   kernel and counts its logical resources.
2. `01_logical_placement.py` places a Bell program on a two-slot logical
   machine.
3. `02_code_and_gadget.py` defines the Steane code and a syndrome-extraction
   gadget, then inspects the gadget's static operations.
4. `03_physical_schedule.py` compiles a logical memory experiment onto a
   surface-code device and lowers it to physical events.
5. `04_stim_emission.py` exports a surface-code memory experiment as an
   annotated Stim circuit. See
   [Stim emission](../../docs/source/use-cases/stim-emission.md).
6. `05_memory_edem.py` derives the same experiment's detector error model
   gadget by gadget and saves it to `memory-edem/memory.edemb`. See
   [Derive an error model from gadgets](../../docs/source/use-cases/gadget-edems.md).
7. `06_steane_bernstein_vazirani.py` recovers a hidden bit string with Steane
   encoded qubits using `cudaq.run` on Stim. Its explicit Clifford encoder
   prepares ideal encoded states; it does not perform noisy error correction
   or decoding.
8. `07_magic_state_distillation.py` defines a 15-to-1 T-state distillation
   protocol with resource requests and postselection, then estimates its
   static resources.
9. `08_t_factory_evaluation.py` evaluates T-state distillation with
   non-Clifford gates, measurement feedback, and raw-state noise on `BareQubit`
   patches. It supplies explicit preparation and gate teleportation, selecting
   accepted shots in Python, rather than executing example 07 unchanged.
10. `09_gidney_ekera_lookup_addition.py` compiles a four-bit Gidney--Ekerå
    lookup-addition circuit through synthesis, PBC normalization, placement,
    QEC selection, and physical scheduling, including factory provisioning
    and analytical estimates.

Run any example from the repository root:

```bash
python3 preview/logical/examples/standalone/00_logical_program.py
```

`04_stim_emission.py` prints the circuit to stdout and any sampling summary to
stderr, so you can redirect the circuit to a file:

```bash
python3 preview/logical/examples/standalone/04_stim_emission.py > surface-memory.stim
```

The Stim-export and EDEM examples can also sample their results with Stim.
Sampling is off by default; to enable it, install `stim>=1.16,<2` and set `shots`
to a positive number in the parameter cell. `05_memory_edem.py` samples only
uniform Pauli channels.
