# CUDA-Q Logical documentation

Sphinx sources for the [CUDA-Q Logical](../README.md) (`cudaq.logical`)
documentation site. The theme is `sphinx_rtd_theme`, in parity with the main
CUDA-Q documentation; content is authored in MyST Markdown.

## Build

From the repository root:

```bash
uv run --directory preview/logical/docs build-docs
```

Then open `preview/logical/docs/_build/html/index.html`.

Extra arguments are forwarded to `sphinx-build` (for example,
`uvx --from ./preview/logical/docs build-docs -- -n` for nit-picky mode).

From this directory, `make html` and `make clean` wrap the same `sphinx-build`
call (`-W` is on by default; override with `make html SPHINXOPTS=`).

The build is hermetic: it never imports or executes `cudaq.logical`. Shipped
examples are embedded with `literalinclude` (they are executed by the pytest
example suite and CI, not by the docs build).
