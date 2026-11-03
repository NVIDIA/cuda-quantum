# cudaq.logical

`cudaq.logical` is the CUDA-Q toolkit for estimating the resources required by
fault-tolerant quantum computations. It connects ordinary CUDA-Q kernels and
portable logical programs to logical placement, codes, gadgets, distillation
protocols, and inspectable resource estimates.

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
standards-compatible Stim circuit text. Supported P3 physical builds can also
execute through `cudaq.run` on CUDA-Q simulators, returning the program's
explicit Boolean results per shot.

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

Follow the repository's [development environment setup](../../Dev_Setup.md)
and [CUDA-Q build instructions](../../Building.md) first. The commands below
run from the repository root and assume those native prerequisites are available.

### Build and install with the CUDA-Q script

Add `-L` to the usual CUDA-Q build command:

```bash
bash scripts/build_cudaq.sh -L
ctest --test-dir build/preview/logical --output-on-failure
```

The script builds and installs CUDA-Q, then EDEM and Logical against that
installation. It reuses CUDA-Q's configured compilers, Python interpreter, and
nanobind. Logical's build tree is `build/preview/logical` with the default
build directory; use `-i -L` for subsequent incremental builds.

### Manual build with CMake

To drive CMake yourself instead of using the script, enable Logical in the
CUDA-Q build, together with the Python bindings:

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
CUDA-Q tools. Enabling Logical also builds EDEM and its Python bindings,
placing `cudaq_edem` alongside `cudaq` in `build/python`.

Run an example against the build tree. Logical's package lives in CUDA-Q's own
build-tree Python package, so no shim is needed:

```bash
PYTHONPATH=build/python python3 preview/logical/examples/00_logical_resource_estimate.py
```

Or install the `cudaq-logical` wheel and run the example files directly.

### Test dependencies

The tests and reference examples need `pytest`, `pytest-xdist`, `lit`, and
`Stim`, which [`requirements-test.txt`](requirements-test.txt) supplies (both
Docker test stages use it too). These are not build requirements, but install
them from the repository root *before* configuring: CMake registers the
`FileCheck` tests only if lit is already available.

```bash
python3 -m pip install -r preview/logical/requirements-test.txt
```

Run the test suites with `ctest` in the Logical build directory:
`build/preview/logical` for the script build, or `build/external/logical` for
the manual CMake build.

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
