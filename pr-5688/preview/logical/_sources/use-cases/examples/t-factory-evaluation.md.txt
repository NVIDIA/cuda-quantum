# T-State Distillation with `cudaq.run`

[`standalone/08_t_factory_evaluation.py`](../../../../examples/standalone/08_t_factory_evaluation.py)
runs a 15-to-1 T-state distillation attempt per shot with `cudaq.run`, with
and without noise on its raw T states.
It complements the [static protocol example](distillation.md): execution uses
explicit raw-state preparation and gate teleportation, with accepted shots
selected in Python, rather than running the resource-backed protocol unchanged.

```{literalinclude} ../../../../examples/standalone/08_t_factory_evaluation.py
:language: python
```

[Back to examples](index.md)
