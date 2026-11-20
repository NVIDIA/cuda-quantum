# Steane Preparation Check with Noise on Stim

[`standalone/04_steane_preparation_check.py`](../../../../examples/standalone/04_steane_preparation_check.py)
prepares one Steane encoded zero, extracts its six stabilizers, and measures
the seven data carriers in Z. It compares noiseless execution with
depolarizing gate noise on Stim.

Each shot returns 14 Booleans: three X checks, three Z checks, seven physical
data results, and logical Z (their XOR). The encoder and syndrome circuits are
not fault-tolerant protocols, and no error correction, decoding, or
postselection is applied. For decoding, see the
[detector error model tutorial](../gadget-edems.md) and the
[memory EDEM example](memory-edem.md). The next example,
[Steane-encoded Bernstein–Vazirani](steane-bernstein-vazirani.md), uses the same
encoder in an ideal computation.

```{literalinclude} ../../../../examples/standalone/04_steane_preparation_check.py
:language: python
```

[Back to examples](index.md)
