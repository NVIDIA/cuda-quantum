# Building against CUDA-Q

CUDA-Q Logical is built against CUDA-Q. The CUDA-Q installation supplies:

- the LLVM and MLIR CMake packages (`LLVMConfig.cmake`, `MLIRConfig.cmake`) and
  headers CUDA-Q Logical compiles against;
- `CUDAQConfig.cmake` and the `AddCUDAQ` helpers;
- the `cudaq::MLIR` target — the shared `libcudaqMLIR` library; and
- the CUDA-Q headers for the Quake dialect.

Every MLIR symbol in CUDA-Q Logical's libraries, tools, and Python extensions
has to resolve from the single `libcudaqMLIR` image provided by CUDA-Q.
Otherwise the dialect registry, the pass registry, and MLIR `TypeID`s are
duplicated between CUDA-Q Logical and CUDA-Q, and in-process `quake` import
breaks in confusing ways.

There are two supported ways to build CUDA-Q Logical against CUDA-Q:

## Option 1 (default): the `cudaq-devel` wheel

Install the CUDA-Q development wheel into the Python environment you build with,
then configure with no CUDA-Q-specific flags at all:

```bash
pip install cudaq-devel "nanobind<3" "lit<23" cmake ninja
cmake -S preview/logical -B preview/logical/build -G Ninja
cmake --build preview/logical/build
```

CMake resolves the wheel through the Python interpreter it selects (pass
`-DPython3_EXECUTABLE=/path/to/python` to pin a specific environment): the
interpreter's `site-packages` is prepended to `CMAKE_PREFIX_PATH`, so CUDA-Q,
LLVM, and MLIR all come from that one prefix.

```{note}
While CUDA-Q Logical is in preview, `cudaq-devel` may not yet be on your
package index. Obtain it from a CUDA-Q source build — CUDA-Q's packaging
scripts produce the wheel, and `pip install -f <wheelhouse> cudaq-devel`
installs it together with the matching `cudaq` runtime — or point the build
at a CUDA-Q install prefix directly (Option 2).
```

## Option 2: a CUDA-Q installation built from source

Set `CUDAQ_INSTALL_PREFIX` to the install prefix of a CUDA-Q build. This is the
route for validating a CUDA-Q Logical change against an unreleased CUDA-Q work
tree.

CUDA-Q must be built with the LLVM/MLIR revision it expects. It is therefore
recommended to install CUDA-Q with `CUDAQ_BUNDLE_MLIR_INSTALL=ON` so the
matching LLVM/MLIR development tree is colocated in the install prefix:

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
  -DCUDAQ_INSTALL_PREFIX="$CUDAQ_INSTALL_PREFIX"
cmake --build preview/logical/build
```

`LLVM_DIR` and `MLIR_DIR` still work as overrides — needed when CUDA-Q was
installed without `CUDAQ_BUNDLE_MLIR_INSTALL=ON` — but they must name the exact
LLVM/MLIR installation CUDA-Q itself was built against.

## Build options

| Option                 | Default                             | Purpose                                            |
| ---------------------- | ----------------------------------- | -------------------------------------------------- |
| `CUDAQ_INSTALL_PREFIX` | empty (use the `cudaq-devel` wheel) | Prefix of a CUDA-Q installation built from source. |

## Verifying the build

```bash
# Run C++ tests
ctest --test-dir preview/logical/build
# Run Python tests
pytest preview/logical/python
```

The `FileCheck` suite needs `lit` plus the LLVM utilities `FileCheck`, `not`,
and `count`, found under the CUDA-Q prefix or on `PATH`; CMake reports which
suites it had to disable when one is missing.

## Troubleshooting

| Symptom                                                              | Cause                                                                                                                               |
| -------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `No CUDA-Q development installation under <prefix>`                  | `cudaq-devel` is not installed in the resolved Python environment, or `CUDAQ_INSTALL_PREFIX` does not name a CUDA-Q install prefix. |
| `Could NOT find MLIR` after CUDA-Q was found                         | The CUDA-Q installation was built without `CUDAQ_BUNDLE_MLIR_INSTALL=ON`, so LLVM/MLIR are not colocated.                           |
| Duplicate dialect/pass registration or a `TypeID` mismatch at import | Two MLIR images in one process — usually an `LLVM_DIR`/`MLIR_DIR` override pointing outside the CUDA-Q prefix.                      |
