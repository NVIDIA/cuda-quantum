# cudaq.logical

`cudaq.logical` is the CUDA-Q toolkit for estimating the resources required by
fault-tolerant quantum computations. It connects ordinary CUDA-Q kernels and
portable logical programs to logical placement, codes, gadgets, distillation
protocols, and inspectable resource estimates.
Supported P3 physical builds can also execute through `cudaq.run` on CUDA-Q
simulators, returning the program's explicit Boolean results per shot.

`cudaq.logical` organizes compilation into explicit semantic stages:

- **P0** describes a machine-independent logical program.
- **P1** places that program on a logical machine without choosing a QEC code.
- **P2** selects codes, gadgets, and protocols and exposes their resource cost.
- **P3** materializes physical resources and events, schedules them against a
  typed device, and supports schedule-derived resource estimates.

Python is the main authoring interface. The same compiler and estimation
workflows are also available from `qlx-opt` through ordinary MLIR pass-pipeline
strings, and Stim circuits can be emitted from `qlx-translate`.

As a secondary interchange path, a selected P2 program can be emitted as
standards-compatible Stim circuit text.

## Start with the examples

The executable walkthroughs in [`examples/`](examples/README.md) cover:

- CUDA-Q kernel estimation with logical and Clifford+T target stacks;
- complete reference-physical estimation with a configurable built-in
  surface-code target;
- complete custom target construction with the Carbon code, kernel-backed
  gadgets, and an explicit physical machine;
- Fermi--Hubbard and Gidney--Ekerå application studies;
- standalone Python authoring with each compiler step exposed;
- encoded circuit execution and T-state distillation through `cudaq.run`
  (standalone examples 06 and 07); and
- standalone MLIR workflows through `qlx-opt` and `qlx-translate`.

The full documentation (quick start, concepts, workflow guides, and the example
gallery) lives in [`docs/`](docs/README.md) and builds with
`sphinx-build -W -b html docs/source docs/_build/html`.

## Build

There are two ways to build `cudaq.logical`. Both use CUDA-Q's single shared
LLVM/MLIR image (`libcudaqMLIR`), so the CUDA-Q you build against must come from
the same sources as this directory.

### In-tree (recommended for development)

Enable it in the CUDA-Q build, together with the Python bindings:

```bash
cmake -S . -B build -DCUDAQ_ENABLE_LOGICAL=ON -DCUDAQ_ENABLE_PROJECTS="cudaq;runtime;python"  # plus your usual CUDA-Q options
cmake --build build
ctest --test-dir build/external/logical
```

`CUDAQ_ENABLE_LOGICAL` requires CUDA-Q's Python bindings: include `python` in
`CUDAQ_ENABLE_PROJECTS` (for example `-DCUDAQ_ENABLE_PROJECTS="cudaq;runtime;python"`;
the deprecated `-DCUDAQ_ENABLE_PYTHON=ON` is equivalent). `scripts/build_cudaq.sh`
enables Python by default. `CUDAQ_ENABLE_LOGICAL` is shorthand for
`-DCUDAQ_EXTERNAL_PROJECTS=logical -DCUDAQ_EXTERNAL_LOGICAL_SOURCE_DIR=$PWD/preview/logical`.
The tools (`qlx-opt`, `qlx-translate`) land in the build's `bin/` next to the
CUDA-Q tools.

### Out-of-tree

Build against an *installation* of CUDA-Q, i.e. the prefix given to
`cmake --install` on a CUDA-Q build of the same sources (configure CUDA-Q with
`-DCUDAQ_BUNDLE_MLIR_INSTALL=ON` to co-locate LLVM/MLIR there, or point
`LLVM_DIR`/`MLIR_DIR` at the LLVM it was built with):

```bash
cmake -S preview/logical -B build/preview/logical -G Ninja \
  -DCUDAQ_INSTALL_PREFIX=/path/to/cudaq/install
cmake --build build/preview/logical
```

`CUDAQ_DIR` or `CMAKE_PREFIX_PATH` work in place of `CUDAQ_INSTALL_PREFIX`.
You also need `nanobind` (`-Dnanobind_DIR=...` or `NANOBIND_INSTALL_PREFIX`),
GMP/MPFR headers and `lit`. The build engineer is trusted to supply a matching
installation; to have CMake audit that the LLVM version and the compiler family
agree with those CUDA-Q was built with, add `-DQLX_VERIFY_TOOLCHAIN=ON`.

Build with the **same compiler family** (clang or GCC) as CUDA-Q itself, e.g.
`CC=clang CXX=clang++`: CMake otherwise silently picks the system default, and a
mismatch compiles fine but fails at runtime with verifier errors
(`parent must have the SymbolTable trait`). `QLX_VERIFY_TOOLCHAIN` checks this
too. See [`docs/source/reference/building-against-cudaq.md`](docs/source/reference/building-against-cudaq.md).

If no prefix is given, CMake falls back to the `cudaq-devel` wheel in the active
Python environment. That is how the `cudaq-logical` wheel is built in CI, and
it is not the way to build from a source checkout.

Run an example against the build tree. In-tree, Logical's package lives in
CUDA-Q's own build-tree Python package, so no shim is needed:

```bash
PYTHONPATH=build/python python3 preview/logical/examples/00_logical_resource_estimate.py
```

Out-of-tree, the build writes a `_cudaq_logical_devpath` shim that puts the
build tree and the CUDA-Q installation on the import path:

```bash
PYTHONPATH=build/preview/logical/python \
  python3 -c "import _cudaq_logical_devpath, runpy; runpy.run_path('preview/logical/examples/00_logical_resource_estimate.py', run_name='__main__')"
```

Or install the `cudaq-logical` wheel and run the example files directly.

Run the test suites with `ctest` in the Logical build directory
(`build/external/logical` for an in-tree build).

Python tests exclude the `slow` group by default, including in PR CI. This
keeps the RSA-2048 Gidney–Ekerå checks in routine runs while the RSA-3072/4096
factory-characterization and resource-kernel checks are opt-in. With the
package installed, run just the slow group or the full Python suite with:

```bash
python3 -m pytest preview/logical/python/tests/cudaq/logical -m slow
python3 -m pytest preview/logical/python/tests/cudaq/logical -m ""
```

For a build-tree CTest run, use `PYTEST_ADDOPTS='-m slow' ctest -R
qlx-python-tests` from the Logical build directory, or `PYTEST_ADDOPTS='-m ""'`
to include all Python tests.
