# Gidney–Ekerå RSA-2048 resource envelope

**Outcome.** The historical Table-3, `c=5` lookup-addition workload — 503,808
folded lookup additions over a 2,086-qubit accumulator — is expressed as
ordinary `cudaq.kernel`s and estimated through the estimator target. CUDA-Q
Logical reports the logical profile (2.63 billion Toffoli demand, 4,187 peak
logical qubits), and the Gidney–Ekerå model projects the physical envelope:
physical qubits, lookup cycles, total cycles, runtime, and a proxy retry
risk.

**Evidence boundary.** This is a resource study, not a factor-producing
execution: the lookup data are placeholders. The physical-qubit, runtime, and
retry-risk values are analytical projections from the logical profile, not
further compilation stages — no code or machine is selected along the way.

## Canonical source

```{literalinclude} ../../../examples/06_gidney_ekera.py
:language: python
:linenos:
```

[Back to the gallery](../examples)
