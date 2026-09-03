# Developing CUDA-Q Logical

Contributor mechanics for the `preview/logical/` tree: build, test, docs, and
where things live.

## Prerequisites

CUDA-Q Logical requires CMake 3.30+, a C++20 compiler, Python 3.11+, and a
CUDA-Q development installation. That installation is CUDA-Q Logical's sole
source of LLVM, MLIR, and the shared `libcudaqMLIR` compiler library — the
project does not discover or build LLVM/MLIR itself. Python-side build
requirements are `nanobind` (extension modules) and `lit` (the FileCheck suite);
Ninja and ccache are recommended.

## Configure and build

From the repository root:

```bash
pip install cudaq-devel nanobind lit cmake ninja
cmake -S preview/logical -B preview/logical/build -G Ninja
cmake --build preview/logical/build
```

CUDA-Q is discovered from the `cudaq-devel` wheel in the Python environment
CMake resolves (pass `-DPython3_EXECUTABLE=...` to pin one); configuration fails
with a diagnostic if that wheel is absent. The full development-installation
contract, the source-build alternative (`CUDAQ_INSTALL_PREFIX`), and the failure
diagnostics are in
[Building against CUDA-Q](reference/building-against-cudaq.md). A normal CUDA-Q
runtime installation is not sufficient.

## Output layout

```text
preview/logical/build/
├── bin/                    qlx-opt, qlx-translate
├── lib/                    dialect, pass, and binding libraries
├── python/cudaq/logical/   staged Python package
│   ├── dialects/           generated MLIR Python bindings
│   └── _mlir_libs/         native extensions (resolve libcudaqMLIR)
└── test/                   configured lit suites
```

The embedded MLIR bindings live under `cudaq.logical._mlir_libs` and
`cudaq.logical.dialects`, nested inside the `cudaq.logical` package so the
extension RPATHs resolve against CUDA-Q's own bindings.

## The Python edit–test loop

`build/python/cudaq/logical` is a configure-time **copy** of
`python/cudaq/logical` (CMake `configure_file` COPYONLY), not a symlink. Edit
under `python/cudaq/logical/`, re-run the configure step to re-stage, then test
against the staged package:

```bash
cmake -S preview/logical -B preview/logical/build   # re-stages the package
PYTHONPATH=preview/logical/build/python \
  python3 -m pytest preview/logical/python/tests/cudaq/logical/ -q
```

The staged tree has no `cudaq/__init__.py` of its own; importing
`_cudaq_logical_devpath` first extends `cudaq.__path__` so
`import cudaq.logical` resolves from the build tree next to the installed CUDA-Q
runtime (the test suite's `conftest.py` does this automatically). The same
mechanism runs examples directly:

```bash
PYTHONPATH=preview/logical/build/python \
  python3 -c "import _cudaq_logical_devpath, runpy; runpy.run_path('preview/logical/examples/01_p0_bell.py', run_name='__main__')"
```

## Tests

```bash
ctest --test-dir preview/logical/build
ctest --test-dir preview/logical/build -L conformance
```

CTest registers three entries: `qlx-lit-build-tools` (a build fixture that
produces the command-line tools), `qlx-filecheck-tests` (the lit/FileCheck
suites), and `qlx-python-tests` (the pytest suite, which also executes every
shipped example). The latter two carry the `conformance` label.

For focused work:

```bash
# One lit directory.
lit -sv preview/logical/build/test/Model

# One conversion directly.
preview/logical/build/bin/qlx-opt preview/logical/examples/cli/static.mlir \
  --pass-pipeline='builtin.module(fabric-count{root=memory device=device result=static})'
```

Test placement:

| Kind                                                             | Directory                                                                        |
| ---------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Dialect, verifier, conversion, and CLI-workflow FileCheck tests  | `preview/logical/test/` (`Conversion/`, `Dialect/`, `Model/`, `QLX/`, `Target/`) |
| Python API/compiler/target tests and the example-execution suite | `preview/logical/python/tests/cudaq/logical/`                                    |

Tests import the same `cudaq.logical` package users receive; there is no second
legacy test mode.

## Documentation

The docs build is plain Sphinx — hermetic, wheel-free, and independent of the
CMake build (there is no CMake `docs` target):

```bash
uv run --directory preview/logical/docs --extra build-deps build-docs
```

`make -C preview/logical/docs html` wraps the same command. Shipped examples are
embedded with `literalinclude` and executed by the test suite, never by the docs
build; broken cross-references fail the build (`-W`).

## Repository map

| Path                                          | Content                                                                                                  |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `preview/logical/python/cudaq/logical/`       | Canonical Python model: programs, codes, gadgets, protocols, devices, compiler, estimation, targets      |
| `preview/logical/ir/`                         | MLIR dialects (`qlx`, `lvm`, `fabric`), passes, Python bindings, and the `qlx-opt`/`qlx-translate` tools |
| `preview/logical/test/`                       | All lit/FileCheck tests                                                                                  |
| `preview/logical/python/tests/cudaq/logical/` | The pytest suite                                                                                         |
| `preview/logical/examples/`                   | The shipped, test-executed examples (`00`–`07` plus `cli/`)                                              |
| `preview/logical/docs/`                       | This documentation (Sphinx sources, `pyproject.toml`, `Makefile`)                                        |
| `preview/logical/cmake/`                      | The CUDA-Q discovery module and build helpers                                                            |
| `preview/logical/scripts/`                    | Packaging helper scripts                                                                                 |

## Adding an MLIR operation

1. Add or update the dialect TableGen record under
   `preview/logical/ir/include/...`.
2. Implement custom parsing, printing, folding, or verification in the matching
   `preview/logical/ir/lib/` source.
3. Regenerate or extend the Python bindings and any typed Python builder if the
   operation is user-facing.
4. Add positive and negative FileCheck cases under `preview/logical/test/`.
5. Add Python semantic evidence when the operation is reachable from the
   programming model.

## Adding Python functionality

The decorator programming model is convenience syntax over typed builder and
model objects. New APIs should normalize to those types and lower through the
same MLIR backbone. Avoid global catalogs, raw string identities where a typed
value exists, and text scraping when an operation binding is available.

## Diagnostics

Set `QLX_MLIR_PRINT_IR` to inspect the embedded pass pipelines: `before`,
`after`, `all`, `changed`, or `failure` select the mode (any other truthy value
prints after each pass). IR printing forces the pass manager single-threaded,
exactly as `--mlir-print-ir-*` does for the command-line tools.

If an import appears stale after moving Python sources, reconfigure before
rebuilding so CMake refreshes the staged package.
