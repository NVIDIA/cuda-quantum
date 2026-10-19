# CUDA-Q EDEM

This component establishes the build and packaging structure for the EDEM
library and its GF(2) dependency. The headers and compiled libraries are stubs;
functional interfaces will follow in subsequent changes.

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
`Development` and `Runtime`.

## Packaging checks

The tests compile and link C and C++ consumers using the build-tree targets,
then install, relocate, and consume the package from a separate build. They
also exercise a parent that defines GF2 before adding EDEM and installs both
libraries through one `Core` component, and a standalone build with custom
library and header installation directories.

These checks verify build and packaging contracts. The stubs provide no
mathematical operations or functional ABI to test.
