# Building against CUDA-Q

CUDA-Q Logical is part of CUDA-Q: it shares CUDA-Q's single `libcudaqMLIR`
image, its LLVM/MLIR, its Python bindings, and its nanobind domain. Every MLIR
symbol in Logical's libraries, tools, and Python extensions must resolve from
that one image. Otherwise the dialect registry, the pass registry, and MLIR
`TypeID`s are duplicated between Logical and CUDA-Q, and in-process `quake`
import breaks in confusing ways.

There are two ways to build it. Both assume you build everything -- CUDA-Q,
LLVM/MLIR, and Logical -- with the same toolchain.

## Use one compiler for everything

```{important}
Build CUDA-Q, its LLVM/MLIR, and CUDA-Q Logical with **the same C++ compiler
family** (clang or GCC), and ideally the same version.
```

This is a requirement, not a preference. MLIR derives some `TypeID`s -- for
example the `SymbolTable` op trait -- from compiler-specific type-name strings.
Objects built by a different compiler family disagree with `libcudaqMLIR` about
those IDs. The result compiles and links cleanly and then fails at runtime, for
example with `'qlx.program' op symbol's parent must have the SymbolTable trait`
on almost every Python test.

- In-tree builds get this for free: there is one configure and one compiler.
- For an out-of-tree build, configure with the compiler CUDA-Q was built with,
  e.g. `CC=clang CXX=clang++` (or `-DCMAKE_C_COMPILER`/`-DCMAKE_CXX_COMPILER`).
  CMake otherwise silently picks the system default, which is often GCC.
- To have CMake check this for you, configure with
  `-DQLX_VERIFY_TOOLCHAIN=ON`. It fails the configure when the compiler family
  differs from the one recorded in `libcudaqMLIR`, and when the LLVM version
  differs from the one CUDA-Q was built with.

## Option 1 (recommended): in-tree

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
`-DCUDAQ_EXTERNAL_PROJECTS=logical -DCUDAQ_EXTERNAL_LOGICAL_SOURCE_DIR=<repo>/preview/logical`.
Logical's Python package, extensions, and tools land in CUDA-Q's own build
tree (`python/cudaq/...`, `bin/`) and install into CUDA-Q's install prefix.

## Option 2: out-of-tree, against a CUDA-Q installation

Build against an *installation* of CUDA-Q -- the prefix given to
`cmake --install` on a CUDA-Q build **of the same sources**. Use this route to
build Logical separately, for instance when iterating on Logical alone.

We recommend installing CUDA-Q with `CUDAQ_BUNDLE_MLIR_INSTALL=ON` so the
matching LLVM/MLIR development tree is colocated in the prefix:

```text
$CUDAQ_INSTALL_PREFIX/
├── bin/
├── include/
├── cudaq/mlir/_mlir_libs/        Python bindings and nanobind
└── lib/
    ├── cmake/
    │   ├── cudaq/CUDAQConfig.cmake
    │   ├── llvm/LLVMConfig.cmake
    │   └── mlir/MLIRConfig.cmake
    └── libcudaqMLIR.so
```

Then configure against it, with the same compiler:

```bash
CC=clang CXX=clang++ cmake -S preview/logical -B build/preview/logical -G Ninja \
  -DCUDAQ_INSTALL_PREFIX="$CUDAQ_INSTALL_PREFIX" \
  -DQLX_VERIFY_TOOLCHAIN=ON
cmake --build build/preview/logical
cmake --install build/preview/logical   # installs into the CUDA-Q prefix
```

`CUDAQ_DIR` or `CMAKE_PREFIX_PATH` work in place of `CUDAQ_INSTALL_PREFIX`.
Unless you set `CMAKE_INSTALL_PREFIX`, installation goes into the same prefix,
so Logical and CUDA-Q end up in one tree. You also need `nanobind`
(`-Dnanobind_DIR=...` or `NANOBIND_INSTALL_PREFIX`), the GMP/MPFR headers, and
`lit`.

`LLVM_DIR` and `MLIR_DIR` still work as overrides -- needed when CUDA-Q was
installed without `CUDAQ_BUNDLE_MLIR_INSTALL=ON` -- but they must name the exact
LLVM/MLIR installation CUDA-Q itself was built against.

```{note}
Make sure the installation is clean. `cmake --install` never removes files left
by an older CUDA-Q build, and stale libraries from an earlier packaging (such as
a second copy of MLIR embedded in an old Python binding) can make `import cudaq`
itself abort with `Trying to register different dialects for the same
namespace`.
```

## Fallback: the `cudaq-devel` wheel

If no CUDA-Q installation is given, CMake falls back to the `cudaq-devel` wheel
in the active Python environment. This is how the `cudaq-logical` wheel is built
in CI. It is not the way to build from a source checkout, and the wheel may not
be available on your package index while Logical is in preview.

```bash
pip install <path-to-cudaq-devel-wheel> "nanobind<3" "lit<23" cmake ninja
cmake -S preview/logical -B preview/logical/build -G Ninja
cmake --build preview/logical/build
```

## Build options

| Option                   | Default | Purpose                                                                                           |
| ------------------------ | ------- | ------------------------------------------------------------------------------------------------- |
| `CUDAQ_ENABLE_LOGICAL`   | `OFF`   | In-tree builds: build Logical as part of the CUDA-Q build.                                        |
| `CUDAQ_INSTALL_PREFIX`   | empty   | Out-of-tree builds: prefix of the CUDA-Q installation to build against.                           |
| `QLX_VERIFY_TOOLCHAIN`   | `OFF`   | Out-of-tree builds: fail the configure if the compiler family or LLVM version differs from CUDA-Q's. |

## Verifying the build

```bash
# In-tree: C++ (lit) and Python tests
ctest --test-dir build/external/logical
# Out-of-tree
ctest --test-dir build/preview/logical
```

The `FileCheck` suite needs `lit` plus the LLVM utilities `FileCheck`, `not`,
and `count`, found under the CUDA-Q prefix or on `PATH`; CMake reports which
suites it had to disable when one is missing.

## Troubleshooting

| Symptom                                                                          | Cause                                                                                                                                  |
| -------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `No CUDA-Q installation found`                                                   | `CUDAQ_INSTALL_PREFIX` does not name a CUDA-Q installation (it must contain `lib/cmake/cudaq/CUDAQConfig.cmake`).                      |
| `Could NOT find MLIR` after CUDA-Q was found                                     | The CUDA-Q installation was built without `CUDAQ_BUNDLE_MLIR_INSTALL=ON`, so LLVM/MLIR are not colocated; pass `LLVM_DIR`/`MLIR_DIR`.  |
| Hundreds of `...parent must have the SymbolTable trait` errors; links fine       | Logical was built with a different compiler family than CUDA-Q. Rebuild with the same one (see above).                                   |
| `error: ... no member named 'SSIType' in mlir::TypeTrait` while compiling        | The `LLVM_DIR`/`MLIR_DIR` found is not the LLVM CUDA-Q was built against.                                                              |
| `import cudaq` aborts with `Trying to register different dialects ... namespace` | Stale files from an older CUDA-Q packaging in the install prefix; install into a clean prefix.                                         |
| Duplicate dialect/pass registration or a `TypeID` mismatch at import             | Two MLIR images in one process -- usually an `LLVM_DIR`/`MLIR_DIR` override pointing outside the CUDA-Q prefix.                         |
