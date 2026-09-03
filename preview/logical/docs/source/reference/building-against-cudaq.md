# Building against CUDA-Q

CUDA-Q Logical is built *against* CUDA-Q, not merely alongside it. One CUDA-Q
installation supplies:

- the LLVM and MLIR CMake packages (`LLVMConfig.cmake`, `MLIRConfig.cmake`)
  and headers CUDA-Q Logical compiles against;
- `CUDAQConfig.cmake` and the `AddCUDAQ` helpers;
- the `cudaq::MLIR` target — the shared `libcudaqMLIR` library; and
- the CUDA-Q headers for the Quake dialect.

CUDA-Q Logical never discovers or builds LLVM/MLIR on its own, and it is never
added to the CUDA-Q build as a sub-project (configuring it with
`add_subdirectory()` is a hard error). Every MLIR symbol in CUDA-Q Logical's
libraries, tools, and Python extensions has to resolve from the single
`libcudaqMLIR` image — otherwise the dialect registry, the pass registry, and
MLIR TypeIDs are duplicated between CUDA-Q Logical and CUDA-Q, and in-process
Quake import breaks in confusing ways. Taking LLVM/MLIR from the same prefix
as CUDA-Q keeps that invariant true by construction.

The normal `cudaq` runtime packages do **not** provide that development
surface; the nvq++ toolchain is not required either, since CUDA-Q Logical
compiles no CUDA-Q kernels.

## Option 1 (default): the `cudaq-devel` wheel

Install the CUDA-Q development wheel into the Python environment you build
with, then configure with no CUDA-Q-specific flags at all:

```bash
pip install cudaq-devel nanobind lit cmake ninja
cmake -S preview/logical -B preview/logical/build -G Ninja
cmake --build preview/logical/build
```

CMake resolves the wheel through the Python interpreter it selects (pass
`-DPython3_EXECUTABLE=/path/to/python` to pin a specific environment): the
interpreter's `site-packages` is prepended to `CMAKE_PREFIX_PATH`, so CUDA-Q,
LLVM, and MLIR all come from that one prefix.

If the development installation is missing, configuration stops with:

```text
No CUDA-Q development installation under <site-packages>
(expected lib/cmake/cudaq/CUDAQConfig.cmake).
```

```{note}
While CUDA-Q Logical is in preview, `cudaq-devel` may not yet be on your
package index. Obtain it from a CUDA-Q source build — CUDA-Q's packaging
scripts produce the wheel, and `pip install -f <wheelhouse> cudaq-devel`
installs it together with the matching `cudaq` runtime — or point the build
at a CUDA-Q install prefix directly (Option 2).
```

## Option 2: a CUDA-Q installation built from source

Set `QLX_CUDAQ_INSTALL_DIR` to the install prefix of a CUDA-Q build. This is
the route for validating a CUDA-Q Logical change against an unreleased CUDA-Q
work tree.

CUDA-Q must be built with the LLVM/MLIR revision it expects — prefer the LLVM
tree from CUDA-Q's pinned submodule. An arbitrary system LLVM is not ABI
evidence. Install CUDA-Q with `CUDAQ_BUNDLE_MLIR_INSTALL=ON` so the matching
LLVM/MLIR development tree is colocated in the install prefix:

```text
$CUDAQ_INSTALL_PREFIX/
├── bin/                          FileCheck, not, count, mlir-tblgen, …
├── include/
└── lib/
    ├── cmake/
    │   ├── cudaq/CUDAQConfig.cmake
    │   ├── llvm/LLVMConfig.cmake
    │   └── mlir/MLIRConfig.cmake
    └── libcudaqMLIR.so
```

Then configure against it:

```bash
cmake -S preview/logical -B preview/logical/build -G Ninja \
  -DQLX_CUDAQ_INSTALL_DIR="$CUDAQ_INSTALL_PREFIX"
cmake --build preview/logical/build
```

`LLVM_DIR` and `MLIR_DIR` still work as overrides — needed when CUDA-Q was
installed without `CUDAQ_BUNDLE_MLIR_INSTALL=ON` — but they must name the
exact LLVM/MLIR installation CUDA-Q itself was built against. Pointing them
at an unrelated LLVM reintroduces exactly the duplicate-MLIR problem this
contract exists to prevent.

## Build options

| Option | Default | Purpose |
|---|---|---|
| `QLX_CUDAQ_INSTALL_DIR` | empty (use the `cudaq-devel` wheel) | Prefix of a CUDA-Q installation built from source. |
| `QLX_BUILD_USE_CCACHE` | `OFF` | Route C/C++/CUDA compilation through `ccache`. |

## Verifying the build

```bash
ctest --test-dir preview/logical/build
```

The FileCheck suite needs `lit` plus the LLVM utilities `FileCheck`, `not`,
and `count`, found under the CUDA-Q prefix or on `PATH`; CMake reports which
suites it had to disable when one is missing.

Confirm the staged Python package imports, and that its native extension
resolves CUDA-Q's shared MLIR library rather than carrying its own copy —
`libcudaqMLIR.so` must appear as a `NEEDED` entry:

```bash
readelf -d preview/logical/build/python/cudaq/logical/_mlir_libs/libQLXPythonCAPI.so \
  | grep libcudaqMLIR
```

## Troubleshooting

| Symptom | Cause |
|---|---|
| `No CUDA-Q development installation under <prefix>` | `cudaq-devel` is not installed in the resolved Python environment, or `QLX_CUDAQ_INSTALL_DIR` does not name a CUDA-Q install prefix. |
| `Could NOT find MLIR` after CUDA-Q was found | The CUDA-Q installation was built without `CUDAQ_BUNDLE_MLIR_INSTALL=ON`, so LLVM/MLIR are not colocated. |
| `does not export cudaq::MLIR` | The CUDA-Q build predates the shared-`libcudaqMLIR` packaging, or only the runtime packages are installed. |
| Duplicate dialect/pass registration or a TypeID mismatch at import | Two MLIR images in one process — usually an `LLVM_DIR`/`MLIR_DIR` override pointing outside the CUDA-Q prefix. |
