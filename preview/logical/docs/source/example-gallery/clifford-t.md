# Clifford+T synthesis of an arbitrary rotation

**Outcome.** An ordinary CUDA-Q kernel applying `rz(0.1234)` — a rotation off
the Clifford+T lattice — is estimated through CUDA-Q Logical's Clifford+T
target. The synthesis is visible in the result: the logical estimate contains
more than one T action per rotation, alongside an action-depth upper bound.
Selecting the target is two lines: import `clifford_t` and pass it to
`cudaq.set_target`.

**Evidence boundary.** These are *logical* Clifford+T counts after synthesis —
how many H and T actions the rotation approximates to — not encoded or
physical costs. Attaching a code and reading fabric-level counts is what the
surface-code target adds in the [first CUDA-Q kernel estimate](cudaq-resource-estimate).

## Canonical source

```{literalinclude} ../../../examples/05_clifford_t.py
:language: python
:linenos:
```

[Back to the gallery](../examples)
