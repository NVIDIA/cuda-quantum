# A first CUDA-Q kernel estimate

**Outcome.** A stock `cudaq.kernel` — allocate one qubit, measure it — is
compiled through a CUDA-Q Logical target declaring a one-logical-qubit,
distance-3 surface code. `cudaq.estimate` returns annotations that
`FabricCounts` reads back as typed P2 evidence: peak encoded patches, peak
protected logical qubits, per-operation counts, and gadget calls. No CUDA-Q
Logical program authoring is required — the target does the work.

**Evidence boundary.** The counts describe *this* target choice — distance 3,
capacity 1 — which is an input to the estimate, not an optimized result. They
are fabric-level resource counts only: no noise, decoding, or runtime claims
are attached.

## Canonical source

```{literalinclude} ../../../examples/00_cudaq_logical_resource_estimate.py
:language: python
:linenos:
```

[Back to the gallery](../examples)
