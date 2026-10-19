# CUDA-Q EDEM

This component establishes the build and packaging structure for the EDEM
library and its GF(2) dependency. The headers and compiled libraries are stubs;
functional interfaces will follow in subsequent changes.

We are building toward a stable C ABI. The API and ABI are currently work in
progress and may change between releases.

GF2 sources live in `math/gf2` so other math components can be added alongside
them. EDEM owns their combined installation and CMake package. GF2 has no
separate package or release version.

## Standalone build

The libraries require CMake 3.23 or newer and a C11 compiler. Packaging tests
also require a C++17 compiler.

From the CUDA-Q repository root:

```sh
cmake -S edem -B build-edem -DBUILD_SHARED_LIBS=ON
cmake --build build-edem
ctest --test-dir build-edem --output-on-failure
cmake --install build-edem --prefix /path/to/install
```

Set `BUILD_SHARED_LIBS=OFF` for static libraries. Set
`CUDAQ_EDEM_BUILD_TESTS=OFF` to build with only a C compiler. The standalone
build brings GF2 in from the sibling source directory.

## Python bindings and wheel

Python bindings use nanobind and require Python 3.11 or newer and a C++17
compiler. Enable them in a native build with `CUDAQ_EDEM_BUILD_PYTHON=ON` after
installing `nanobind>=2.12,<3` into the chosen Python environment. Set
`Python_EXECUTABLE` to select an interpreter. Native builds default to
`CUDAQ_EDEM_BUILD_PYTHON=OFF`.

The `cudaq-edem` wheel includes the `cudaq_edem` Python package, both native
libraries, headers, and CMake exports. The compiled Python module is initially
a stub; type and function bindings will follow their native implementations.

From the repository root, build the source distribution and its wheel:

```sh
python3 -m pip install "build[virtualenv]"
python3 edem/scripts/build_wheel.py --outdir dist/edem
```

The script stages the EDEM and GF2 sources with `pyproject.toml.edem` as the
standard `pyproject.toml`. It leaves the repository's main wheel configuration
unchanged. Scikit-build-core builds the standalone EDEM project and packages
its CMake installation. The wheel version comes from `CUDAQ_EDEM_VERSION`.

Install the resulting wheel with pip. A downstream CMake build can locate its
package configuration through the same Python environment:

```sh
cmake -S /path/to/consumer -B build-consumer \
  -Dcudaq-edem_DIR="$(python3 -c 'import cudaq_edem; print(cudaq_edem.get_cmake_dir())')"
```

Wheels use CPython and platform compatibility tags. CI produces wheel and
source-distribution artifacts; public release publishing is a separate step.

## CUDA-Q build

Append `edem` to `CUDAQ_ENABLE_PROJECTS`, for example:

```sh
cmake -S . -B build -DCUDAQ_ENABLE_PROJECTS="cudaq;runtime;edem"
```

The top-level CUDA-Q configuration retains its usual toolchain and dependency
requirements. EDEM tests default to off in a parent build; enable them with
`CUDAQ_EDEM_BUILD_TESTS=ON`.

## Installed package

Use the same targets in source-tree and installed builds:

```cmake
find_package(cudaq-edem CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE cudaq::edem)
```

Add the installation prefix to `CMAKE_PREFIX_PATH`. The package also exports
`cudaq::gf2`; linking EDEM supplies that dependency transitively. Both targets
share the EDEM distribution version.

Headers, static archives, and package configuration files belong to the
development install component; shared libraries belong to the runtime component.
Parent CUDA-Q builds supply their component names. Standalone builds use
`Development` and `Runtime`. Python bindings and their third-party notices
belong to `EDEMPython`.

## Packaging checks

The tests compile and link C and C++ consumers using the build-tree targets,
then install, relocate, and consume the package from a separate build. They
also exercise a parent that defines GF2 before adding EDEM and installs both
libraries through one `Core` component, and a standalone build with custom
library and header installation directories.

These checks verify build and packaging contracts. The stubs provide no
mathematical operations or functional ABI to test.

When Python bindings are enabled, CTest also checks the build-tree import.
`edem/test/check_wheel.py` installs a wheel into a fresh virtual environment,
imports its extension, loads both native libraries, relocates the installed
package, and compiles C and C++ consumers against its CMake exports. Run it with
the path to the wheel as its argument.
