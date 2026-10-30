# Standalone CUDA-Q Logical examples

These examples use `cudaq.logical` directly. They do not define
`@cudaq.kernel` functions and do not call `cudaq.estimate`. Instead, they make
the logical authoring, placement, QEC selection, physical lowering, scheduling,
direct estimation, and `cudaq.run` APIs visible.

Examples 00–05 progress from a portable logical program to a complete physical
schedule. Examples 06 and 07 demonstrate ideal encoded computation and noisy
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
7. `06_steane_bernstein_vazirani.py` recovers a hidden bit string with Steane
   encoded qubits on Stim.
8. `07_t_factory_evaluation.py` evaluates T-state distillation with
   non-Clifford gates, measurement feedback, and raw-state noise on `BareQubit`
   patches.

Run any example from the repository root:

```bash
python3 preview/logical/examples/standalone/00_logical_program.py
```

The encoded example uses an explicit Clifford preparation circuit. It does not
claim fault-tolerant preparation or decoded error correction.

For gadget verification, annotated Stim export, and gadget-by-gadget detector
error models, follow the separate [gadget sequence](../gadgets/README.md).
Its `00_verify_gadget.py` introduces verification of one authored gadget;
`01_surface_memory.py` applies the workflow to physical surface-code memory.
