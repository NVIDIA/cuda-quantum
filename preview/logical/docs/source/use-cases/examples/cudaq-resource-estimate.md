# Surface-code resource estimate

[`02_surface_code_resource_estimate.py`](../../../../examples/02_surface_code_resource_estimate.py)
prepares and measures one logical zero using a distance-three surface code.
The target handles encoding, physical lowering, and scheduling; the example
reports physical qubits and scheduled runtime from the `SCHEDULE` annotation.

```{literalinclude} ../../../../examples/02_surface_code_resource_estimate.py
:language: python
:start-at: import cudaq
```

For the result layers and model assumptions, see
[Resource estimation](../estimation.md).

[Back to examples](index.md)
