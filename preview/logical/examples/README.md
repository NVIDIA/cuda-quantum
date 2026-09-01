# cudaq.logical examples

These examples introduce the `cudaq.logical` P0-to-P2 resource-estimation
workflow.

1. `00_cudaq_logical_resource_estimate.py` estimates a simple CUDA-Q kernel
   with a one-logical-qubit, distance-3 surface-code target.
2. `01_p0_bell.py` defines and estimates a machine-independent logical Bell
   program.
3. `02_p1_placement.py` refines the program onto a logical machine.
4. `03_code_and_gadget.py` defines the Steane code and inspects a concrete
   syndrome-extraction gadget.
5. `04_distillation.py` defines a concrete 15-to-1 distillation protocol with
   the root authoring facade and reports its static P2 resource estimate.
6. `05_clifford_t.py` estimates an arbitrary CUDA-Q rotation in the `cudaq.logical`
   Clifford+T target.
7. `06_gidney_ekera.py` estimates the folded Gidney--Ekerå RSA-2048 logical
   resource envelope and projects its physical costs.

The examples are resource studies, not factor-producing executions. The
physical-qubit, runtime, and retry-risk values in `06_gidney_ekera.py` are
analytical projections, not further compilation stages. CUDA-Q and its `cudaq.logical`
ingress are required to run the CUDA-Q-authored examples.

For equivalent workflows expressed as MLIR pass pipelines, continue with the
[`cli/`](cli/README.md) examples.
