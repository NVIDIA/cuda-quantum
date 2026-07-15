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

The `libcudaqMLIR` library can be thought of as having three layers:

1. First, all object files from CUDAQ MLIR libraries (registered with `register_cudaq_mlir_lib`)
   are bundled together in the shared library.
2. Next, all MLIR targets listed in [`mlir-libs-allowlist.txt`](mlir-libs-allowlist.txt)
   are added as static dependencies. By using CMake's `WHOLE_ARCHIVE` flag, we
   ensure that all symbols from these libraries are re-exported, so that CUDAQ libraries
   as well as downstream extensions can use them.
3. Finally, we reduce the surface area of the shared library by hiding symbols
   from the block-list [`mlir-symbols-blocklist.txt`](mlir-symbols-blocklist.txt).
   This list contains symbols that are both unused by CUDAQ and (we estimate)
   are unlikely to become useful in the future for either CUDAQ or downstream extensions.

This somewhat complex setup allows us to ship as much as needed but as little as
possible of MLIR. If at any point, symbols from MLIR are needed that are either
not within the allow-listed MLIR components or are hidden by the block-list, we
can adjust these lists accordingly.

## Adding a new library

In the library's `CMakeLists.txt`:

```cmake
add_cudaq_library(MyNewLib ...)
register_cudaq_mlir_lib(MyNewLib)
```

If the library needs additional upstream MLIR symbols, add the corresponding `MLIR*`
target to `mlir-libs-allowlist.txt` and ensure the required symbols are not hidden
by the block-list.
