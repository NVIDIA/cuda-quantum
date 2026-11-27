# Gidney–Ekerå RSA

[`05_gidney_ekera.py`](../../../../examples/05_gidney_ekera.py) estimates the
folded lookup-addition workload for RSA-2048, RSA-3072, and RSA-4096. Pass
`--bits` to select the size. The companion
[`gidney_ekera_factory.py`](../../../../examples/gidney_ekera_factory.py)
defines each size's operating point and the factory.

```{literalinclude} ../../../../examples/05_gidney_ekera.py
:language: python
```

[Back to examples](index.md)
