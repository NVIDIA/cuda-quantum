# libcudaqMLIR.so

CUDA-Q ships its dialects, passes and builders as one shared library,
`libcudaqMLIR`, alongside the upstream runtime providers `libLLVM`, `libMLIR`,
`libMLIRExecutionEngineShared` and `libMLIR-C`. The goal is twofold:

1. By providing shared libraries that all CUDAQ libraries (runtime, compiler,
   python extension, QPUs etc) can depend on, we ensure that all components
   share a single copy of the MLIR implementation. This ensures that there is
   always a unique instance of global variables such as pass registries.
2. Downstream extensions of CUDAQ that use MLIR can in turn depend on the same
   provider set (`cudaq::MLIR` resolves it), thus sharing the single MLIR
   instance with CUDAQ rather than shipping their own (which would lead to the
   same duplication issues we are trying to avoid internally).

`libcudaqMLIR` itself carries only CUDA-Q's own code; the upstream MLIR/LLVM
implementation lives in the providers and is never re-embedded in CUDA-Q
libraries. The C API is split the same way: upstream's C API is in `libMLIR-C`,
CUDA-Q's additions are in `libcudaqMLIRCAPI`.

## Build strategy

The `libcudaqMLIR` library is built from the single list in
[`mlir-bundled-libs.txt`](../../../cmake/modules/mlir-bundled-libs.txt) (plus
per-arch native `codegen` resolved at configure time), available as the CMake
variable `CUDAQ_MLIR_BUNDLED_LIBS`. The CUDA-Q section of the list is always
bundled into the shared library as object files (`obj.<lib>` targets built via
`add_cudaq_library`). The upstream MLIR/LLVM sections name what the bundled
objects depend on:

1. With shared providers (the SDK default, `CUDAQ_MLIR_SHARED_PROVIDERS`), they
   are resolved from `libMLIR`/`libMLIRExecutionEngineShared`/`libLLVM`, linked
   publicly so consumers resolve them transitively.
2. Otherwise (legacy static configuration), they are whole-archived into
   `libcudaqMLIR` so their full symbol set is re-exported.

A post-build audit (scripts/check_mlir_symbols.sh) fails the build if a CUDA-Q
library references an MLIR/LLVM symbol the provider set does not export, or
strongly redefines one.

## C API

The C-APIs are shipped as a separate thin wrapper shared library
(`libcudaqMLIRCAPI`) that links to `libcudaqMLIR` and `libMLIR-C` and contains
CUDA-Q's own C API object files. Upstream C API objects are never embedded a
second time; they come from `libMLIR-C`. This keeps the C ABI separate while
ensuring that C API calls use the same MLIR `TypeID`s, registries etc. as the
rest of CUDA-Q.

## Adding a new library

In the library's `CMakeLists.txt`:

```cmake
add_cudaq_library(MyNewLib ...)

# Check that the library is listed in cmake/modules/mlir-bundled-libs.txt.
check_registered_mlir_lib(MyNewLib)
```

Add `MyNewLib` to `cmake/modules/mlir-bundled-libs.txt` under the CUDA-Q section.

If the library needs additional upstream MLIR symbols, add the corresponding
`MLIR*` target to the MLIR/LLVM section of that file.

### Usage requirements and object libraries

`add_cudaq_library` produces `obj.<lib>` targets, used for aggregation into
`libcudaqMLIR.so`. LLVM forwards only `INCLUDE_DIRECTORIES` to the object
library.

To ensure dependencies are correctly reflected in the object library, use
`cudaq_target_link_libraries()` instead of `target_link_libraries()` when
adding dependencies after having used `add_cudaq_library`:

```cmake
add_cudaq_library(MyNewLib ...)
cudaq_target_link_libraries(MyNewLib PRIVATE SomeOtherLib)
```
