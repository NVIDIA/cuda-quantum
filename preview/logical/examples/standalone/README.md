# Standalone CUDA-Q Logical examples

These examples use `cudaq.logical` directly. They do not define
`@cudaq.kernel` functions and do not call `cudaq.estimate`. Instead, they make
the logical authoring, placement, QEC selection, physical lowering, scheduling,
and direct estimation APIs visible.

Examples 00–05 progress from a portable logical program to a complete physical
schedule. Examples 06 and 07 export a surface-code memory experiment for
analysis outside CUDA-Q Logical.

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

Run any example from the repository root:

```bash
python3 preview/logical/examples/standalone/00_logical_program.py
```

Example 06 prints the circuit to stdout and any sampling summary to stderr, so
you can redirect the circuit to a file:

```bash
python3 preview/logical/examples/standalone/06_stim_emission.py > surface-memory.stim
```

Examples 06 and 07 can also sample their results with Stim. Sampling is off by
default; to enable it, install `stim>=1.16,<2` and set `shots` to a positive
number in the parameter cell. Example 07 samples only uniform Pauli channels.
