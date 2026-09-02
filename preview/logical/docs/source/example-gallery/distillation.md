# A 15-to-1 distillation protocol

**Outcome.** A concrete 15-to-1 T-state protocol is authored with the root
facade — fifteen raw T-state requests, the eleven triorthogonal product
rotations, a correcting S gate, and four even-row X measurements whose +1
outcomes are explicitly postselected — and statically estimated at P2. The
estimate reports exactly what the protocol consumes: 15 resource requests, 11
resource rotations, 4 selection checks, and the peak live patch count.

**Evidence boundary.** The static estimate counts protocol structure —
requests, rotations, postselections, patches — not acceptance rates or error
suppression. Postselection makes the accept condition explicit in the program;
the physics of how often it accepts is noise-model territory, and no noise
model exists in the trimmed product.

## Canonical source

```{literalinclude} ../../../examples/04_distillation.py
:language: python
:linenos:
```

[Back to the gallery](../examples)
