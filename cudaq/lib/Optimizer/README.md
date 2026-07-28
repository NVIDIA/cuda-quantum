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

The `libcudaqMLIR` library is built in two layers:

1. All object files from CUDAQ MLIR libraries (registered with `register_cudaq_mlir_lib`)
   are bundled together in the shared library.
2. All MLIR targets listed in [`mlir-libs-allowlist.txt`](mlir-libs-allowlist.txt)
   are added as static dependencies. By using CMake's `WHOLE_ARCHIVE` flag, we
   ensure that all symbols from these libraries are re-exported, so that CUDAQ libraries
   as well as downstream extensions can use them.

## C API

The C-APIs are shipped as a separate thin wrapper shared library that links to
`libcudaqMLIR` and contains the C API object files. This keeps the C ABI separate
while ensuring that C API calls use the same MLIR `TypeID`s, registries etc. as
the rest of CUDA-Q.

## Adding a new library

In the library's `CMakeLists.txt`:

```cmake
add_cudaq_library(MyNewLib ...)
register_cudaq_mlir_lib(MyNewLib)
```

If the library needs additional upstream MLIR symbols, add the corresponding `MLIR*`
target to `mlir-libs-allowlist.txt`.
