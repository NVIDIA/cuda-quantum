# CUDA-Q Logical by Example

The numbered Python programs and `steane_bell_e2e_workflow.py` under
`preview/logical/examples/` run in the test suite. The top-level sequence uses
ordinary CUDA-Q kernels and target stacks; the standalone sequence exposes
each CUDA-Q Logical compiler stage directly.

For one computation across both routes, follow the
[Steane Bell-pair walkthrough](../../getting-started/cudaq-logical-in-practice.md).

## CUDA-Q kernel examples

```{toctree}
:maxdepth: 1

logical-resource-estimate
clifford-t
cudaq-resource-estimate
fermi-hubbard
carbon-code
gidney-ekera
```

## Standalone compiler examples

Examples 00–03 introduce authoring, placement, code and gadget definitions,
and physical scheduling. Continue with surface-memory analysis (04–05),
encoded execution (06), or factory protocols and evaluation (07–08).
Example 09 combines synthesis, factory provisioning, estimation, and scheduling
in a lookup-addition capstone.

```{toctree}
:maxdepth: 1

p0-bell
p1-placement
code-and-gadget
physical-schedule
stim-emission
memory-edem
steane-bernstein-vazirani
distillation
t-factory-evaluation
lookup-addition
```
