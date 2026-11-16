# CUDA-Q Logical by Example

The numbered Python programs and `steane_bell_e2e_workflow.py` under
`preview/logical/examples/` run in the test suite. The top-level sequence uses
ordinary CUDA-Q kernels and target stacks; the standalone sequence exposes
each CUDA-Q Logical compiler stage directly. The gadget sequence introduces
verification and detector error models separately: start with one authored
gadget, then build a physical surface-code memory experiment.

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
and physical scheduling. Continue with encoded execution (06), or factory
protocols and evaluation (07–08).
Example 09 combines synthesis, factory provisioning, estimation, and scheduling
in a lookup-addition capstone.

```{toctree}
:maxdepth: 1

p0-bell
p1-placement
code-and-gadget
physical-schedule
steane-bernstein-vazirani
distillation
t-factory-evaluation
lookup-addition
```

## Gadget verification and error models

Start with [Verify an authored gadget](gadget-verification.md) and
[Physical surface-code memory](gadget-surface-memory.md). The optional
[Stim-emission tutorial](../stim-emission.md) uses ideal MPP measurements, and
the [custom-noise tutorial](../gadget-edems.md) derives models with an explicit
fault factory. The reference comparison below adds individual model exports
and two sampling paths to the physical-memory workflow.

```{toctree}
:maxdepth: 1

gadget-verification
gadget-surface-memory
stim-emission
memory-edem
surface-memory-circuit
```
