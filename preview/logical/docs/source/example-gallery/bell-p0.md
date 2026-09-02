# A portable P0 Bell program

**Outcome.** A Bell program stays entirely machine-independent — it names no
code, device, or simulator — while CUDA-Q Logical still produces a typed P0
build and a logical resource estimate: two logical qubits at peak, one H and
one CX action. The machine-independence is checkable, not aspirational: the
compiled MLIR contains no `lvm` placement constructs.

**Evidence boundary.** P0 evidence is logical only: action counts, instrument
counts, and peak logical qubits. There is no code, physical carrier, or noise
claim at this stage — that absence is the lesson, not missing detail. Codes
and their costs enter at P2.

## Canonical source

```{literalinclude} ../../../examples/01_p0_bell.py
:language: python
:linenos:
```

[Back to the gallery](../examples)
