# Placing a program on a logical machine

**Outcome.** A portable P0 build is refined onto a declared logical machine —
a region with `logical_compute` and `logical_measurement` capabilities and
capacity for two slots — guided by an explicit `colocate` constraint on the
program's `data` values. The result is a code-agnostic P1 build whose
placement bindings put both data qubits on the two compute slots, and which
round-trips through serialization: `Build.replay` reproduces the placement
exactly.

**Evidence boundary.** P1 fixes *where* each logical value may live, not *how*
it is protected: no code, gadget, or physical cost is chosen yet. Placement
decisions are recorded as inspectable, replayable evidence rather than folded
silently into the program.

## Canonical source

```{literalinclude} ../../../examples/02_p1_placement.py
:language: python
:linenos:
```

[Back to the gallery](../examples)
