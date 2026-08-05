# libcudaqMLIR.so

We ship all of upstream MLIR as well as the dialects, passes etc that
CUDAQ defines as one shared `mondo` library. The goal is twofold:

1. By providing a shared library that all CUDAQ libraries (runtime, compiler,
   python extension, QPUs etc) can depend on, we ensure that all components
   share a single copy of MLIR. This ensures that there is always a unique
   instance of global variables such as pass registries.
2. Downstream extensions of CUDAQ that use MLIR can in turn depend on
   `libcudaqMLIR`, thus sharing the single MLIR instance with CUDAQ rather than
   shipping their own (which would lead to the same duplication issues we are
   trying to avoid internally).

## Build strategy

The `libcudaqMLIR` library is built from the single list in
[`mlir-bundled-libs.txt`](mlir-bundled-libs.txt) (plus per-arch native
`codegen` resolved at configure time), available as the CMake variable
`CUDAQ_MLIR_BUNDLED_LIBS`. Every entry in that list is bundled into
the shared library:

1. CUDA-Q dialect/transform libraries are bundled as object files (`obj.<lib>`
   targets built via `add_cudaq_library`).
2. MLIR/LLVM libraries are whole-archived so their full symbol set is
   re-exported for CUDAQ libraries and downstream extensions.

## C API

The C-APIs are shipped as a separate thin wrapper shared library that links to
`libcudaqMLIR` and contains the C API object files. This keeps the C ABI separate
while ensuring that C API calls use the same MLIR `TypeID`s, registries etc. as
the rest of CUDA-Q.

## Adding a new library

In the library's `CMakeLists.txt`:

```cmake
add_cudaq_library(MyNewLib ...)

# Check that the library is listed in mlir-bundled-libs.txt.
check_registered_mlir_lib(MyNewLib)
```

Add `MyNewLib` to `mlir-bundled-libs.txt` under the CUDA-Q section.

If the library needs additional upstream MLIR symbols, add the corresponding
`MLIR*` target to the MLIR/LLVM section of the same file.
