# cudaq.logical

`cudaq.logical` is CUDA-Q's toolkit for estimating the resources required by
fault-tolerant quantum computations. It connects ordinary CUDA-Q kernels and
portable logical programs to logical placement, codes, gadgets, distillation
protocols, and inspectable resource estimates. As a secondary interchange
path, a selected P2 program can be emitted as standards-compatible Stim
circuit text.

`cudaq.logical` organizes compilation into explicit semantic stages:

- **P0** describes a machine-independent logical program.
- **P1** places that program on a logical machine without choosing a QEC code.
- **P2** selects codes, gadgets, and protocols and exposes their resource cost.

Python is the main authoring interface. The same compiler and estimation
workflows are also available from `qlx-opt` through ordinary MLIR pass-pipeline
strings, and Stim circuits can be emitted from `qlx-translate`.

## Start with the examples

The executable walkthroughs in [`examples/`](examples/README.md) cover:

- CUDA-Q-to-P2 compilation with logical and static resource estimates;
- logical program authoring and code-agnostic placement;
- Steane-code gadgets and concrete resource counts;
- 15-to-1 magic-state distillation;
- explicit Steane-encoding Stim text emission; and
- a P0-backed Gidney--Ekerå logical resource estimate.

Command-line MLIR versions of the compiler and estimation workflows are in
[`examples/cli/`](examples/cli/README.md).

The full documentation (quickstart, concepts, workflow guides, and the
example gallery) lives in [`docs/`](docs/README.md) and builds with
`sphinx-build -W -b html docs/source docs/_build/html`.

## Build

`cudaq.logical` builds against an *installed* CUDA-Q development SDK -- it is
never added to the CUDA-Q build as a sub-project, and it never fetches or builds
a second LLVM/MLIR stack. The expected setup is a Python environment with the
matching `cudaq-devel` wheel installed:

```bash
pip install cudaq-devel nanobind lit
cmake -S preview/logical -B preview/logical/build -G Ninja
cmake --build preview/logical/build
```

CMake finds the SDK through the Python interpreter it resolves, so the wheel
just has to be installed in the active environment. To build against a CUDA-Q
work tree instead, install it with `-DCUDAQ_BUNDLE_MLIR_INSTALL=ON` and point
the build at that prefix:

```bash
cmake -S preview/logical -B preview/logical/build -G Ninja \
  -DQLX_CUDAQ_INSTALL_DIR=/path/to/cudaq/install
```

Run an example against the build tree with:

```bash
PYTHONPATH=preview/logical/build/python \
  python3 -c "import _cudaq_logical_devpath, runpy; runpy.run_path('preview/logical/examples/01_p0_bell.py', run_name='__main__')"
```

Or install the `cudaq-logical` wheel and run the example files directly.

Run the test suites with `ctest --test-dir preview/logical/build`.
