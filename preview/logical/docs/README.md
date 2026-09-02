# CUDA-Q Logical documentation

Sphinx sources for the [CUDA-Q Logical](../README.md) (`cudaq.logical`)
documentation site. The theme is `sphinx_rtd_theme`, in parity with the main
CUDA-Q documentation; content is authored in MyST Markdown.

## Build

From the repository root:

```bash
uv venv
uv pip install ./preview/logical/docs
sphinx-build -W -b html preview/logical/docs/source preview/logical/docs/_build/html
```

Then open `preview/logical/docs/_build/html/index.html`.

From this directory, `make html` and `make clean` wrap the same `sphinx-build`
call (`-W` is on by default; override with `make html SPHINXOPTS=`).

The build is hermetic: it never imports or executes `cudaq.logical`. Shipped
examples are embedded with `literalinclude` (they are executed by the pytest
example suite and CI, not by the docs build).
