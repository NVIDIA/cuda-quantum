# Examples

These executable Python programs ship under `preview/logical/examples/` and run
in the test suite. Start with an ordinary CUDA-Q kernel and then use the
remaining examples to inspect each stage of the fault-tolerant lowering.

## Estimate a CUDA-Q kernel

[`00_cudaq_logical_resource_estimate.py`](../../../examples/00_cudaq_logical_resource_estimate.py)
lowers a CUDA-Q kernel through a distance-3 surface-code target and reports the
resulting resources:

```{literalinclude} ../../../examples/00_cudaq_logical_resource_estimate.py
:language: python
```

## Follow the lowering stages

- [`01_p0_bell.py`](../../../examples/01_p0_bell.py) — author a portable
  machine-independent logical program and obtain a logical estimate.
- [`02_p1_placement.py`](../../../examples/02_p1_placement.py) — place that
  program on a declared logical machine.
- [`03_code_and_gadget.py`](../../../examples/03_code_and_gadget.py) — define a
  Steane code and a verified terminal-memory gadget.
- [`04_distillation.py`](../../../examples/04_distillation.py) — build a 15-to-1
  T-state distillation protocol with postselection.

## Study larger computations

- [`05_clifford_t.py`](../../../examples/05_clifford_t.py) — synthesize a CUDA-Q
  rotation into Clifford+T operations before estimation.
- [`06_gidney_ekera.py`](../../../examples/06_gidney_ekera.py) — estimate the
  folded lookup-addition workload used in an RSA-2048 resource envelope.
- [`07_fermi_hubbard.py`](../../../examples/07_fermi_hubbard.py) — compare a
  Trotterized Fermi–Hubbard kernel in the logical action model and after
  Clifford+T synthesis.
