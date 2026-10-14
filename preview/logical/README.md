# cudaq.logical

`cudaq.logical` is the CUDA-Q toolkit for estimating the resources required by
fault-tolerant quantum computations. It connects ordinary CUDA-Q kernels and
portable logical programs to logical placement, codes, gadgets, distillation
protocols, and inspectable resource estimates. As a secondary interchange path,
a selected P2 program can be emitted as standards-compatible Stim circuit text.

`cudaq.logical` organizes compilation into explicit semantic stages:

- **P0** describes a machine-independent logical program.
- **P1** places that program on a logical machine without choosing a QEC code.
- **P2** selects codes, gadgets, and protocols and exposes their resource cost.
- **P3** materializes physical resources and events, schedules them against a
  typed device, and supports schedule-derived resource estimates.

Python is the main authoring interface. The same compiler and estimation
workflows are also available from `qlx-opt` through ordinary MLIR pass-pipeline
strings, and Stim circuits can be emitted from `qlx-translate`.

## Start with the examples

The executable walkthroughs in [`examples/`](examples/README.md) cover:

- CUDA-Q kernel estimation with logical and Clifford+T target stacks;
- complete reference-physical estimation with a configurable built-in
  surface-code target;
- complete custom target construction with the Carbon code, kernel-backed
  gadgets, and an explicit physical machine;
- Fermi--Hubbard and Gidney--Ekerå application studies;
- standalone Python authoring with each compiler step exposed; and
- standalone MLIR workflows through `qlx-opt` and `qlx-translate`.

The full documentation (quick start, concepts, workflow guides, and the example
gallery) lives in [`docs/`](docs/README.md) and builds with
`sphinx-build -W -b html docs/source docs/_build/html`.

## Build

`cudaq.logical` builds against an _installed_ CUDA-Q development SDK -- it is
never added to the CUDA-Q build as a sub-project, and it never fetches or builds
a second LLVM/MLIR stack. The expected setup is a Python environment with the
matching `cudaq-devel` wheel installed:

```bash
pip install cudaq-devel "nanobind>=2.12,<3" "lit<23" pytest stim cmake ninja
cmake -S preview/logical -B build/preview/logical -G Ninja
cmake --build build/preview/logical
```

CMake finds the SDK through the Python interpreter it resolves, so the wheel
just has to be installed in the active environment. To build against a CUDA-Q
work tree instead, install it with `-DCUDAQ_BUNDLE_MLIR_INSTALL=ON` and point
the build at that prefix:

```bash
cmake -S preview/logical -B build/preview/logical -G Ninja \
  -DCUDAQ_INSTALL_PREFIX=/path/to/cudaq/install
```

Run an example against the build tree with:

```bash
PYTHONPATH=build/preview/logical/python \
  python3 -c "import _cudaq_logical_devpath, runpy; runpy.run_path('preview/logical/examples/00_logical_resource_estimate.py', run_name='__main__')"
```

Or install the `cudaq-logical` wheel and run the example files directly.

Run the test suites with `ctest --test-dir build/preview/logical`.
