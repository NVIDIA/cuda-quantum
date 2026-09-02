# Defining a code and a gadget

**Outcome.** The self-dual [[7,1,3]] Steane code is authored as data — CSS
block shape, X/Z stabilizer checks, logical operators, distance — and
materialized into the `fabric` dialect. A terminal-memory gadget is then
authored against a declared objective (`implements=terminal_memory`) over a
typed `qlx.patch[Steane]` boundary, compiled, and inspected: the example reads
back the code's parameters and the gadget's authored operation counts.

**Evidence boundary.** The typed boundary is the contract: the gadget claims
to implement the objective, and that claim is what the compiler checks. The
reported counts are authored operations in the gadget body — they say nothing
about noise, decoding, or failure rates, none of which exist at P2.

## Canonical source

```{literalinclude} ../../../examples/03_code_and_gadget.py
:language: python
:linenos:
```

[Back to the gallery](../examples)
