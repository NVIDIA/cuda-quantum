# Standalone MLIR command-line examples

These examples expose the compiler pipelines used underneath the Python APIs.
Build CUDA-Q Logical, add the build's `bin` directory to `PATH`, and run the
commands from the repository root.

Estimate logical resources and place the program:

```bash
qlx-opt preview/logical/examples/mlir/logical_estimate_and_placement.mlir \
  --pass-pipeline='builtin.module(qlx-estimate-logical{root=bell result=logical},qlx-to-lvm{root=bell domain=machine result=bell_placed})'
```

Count the static resources in an encoded gadget:

```bash
qlx-opt preview/logical/examples/mlir/static_resource_estimate.mlir \
  --pass-pipeline='builtin.module(fabric-count{root=memory device=device result=static})'
```

Schedule physical events and calculate schedule-level resources:

```bash
qlx-opt preview/logical/examples/mlir/physical_schedule_estimate.mlir \
  --phys-schedule='graph=events result=events_schedule' \
  --phys-estimate-schedule='schedule=events_schedule lower-tier=analytical result=estimate'
```

Translate an explicitly authored encoded memory gadget to Stim:

```bash
qlx-translate preview/logical/examples/mlir/stim_memory.mlir --fabric-to-stim
```

`qlx-opt --help` and `qlx-translate --help` list the registered compiler passes
and translations.
