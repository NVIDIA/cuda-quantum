# LLVM patches

This folder contains changes that deviate from the content of the LLVM
repository. This file contains some comments about why these patches were
created.

## `mlir-c-shared-providers.diff` — single MLIR implementation in `libMLIR-C`

Upstream's `MLIR-C` aggregate embeds the CAPI object libraries and then links
their static MLIR dependencies, so `libMLIR-C.so` carries a second, hidden
copy of the MLIR implementation next to `libMLIR.so`. When the shared MLIR
providers are built (`MLIR_LINK_MLIR_DYLIB`/`LLVM_BUILD_LLVM_DYLIB`), the
patch instead excludes the upstream `MLIR_STATIC_LIBS` and
`MLIRExecutionEngine` from the aggregate's dependency embedding and links the
shared `MLIR` and `MLIRExecutionEngineShared` providers. Without this, CUDA-Q
consumers would load two MLIR implementations in the same process. The patch
lives in `mlir/tools/mlir-shlib/CMakeLists.txt` (not `lib/CAPI`): the `MLIR`
target only exists after the shlib subdirectory is processed.

## Building `compiler-rt`

Building `compiler-rt` as part of the runtimes, rather than as a project, is
necessary to properly link all runtimes against each other. However, when doing
so, some of the libraries that are successfully installed when building it as a
project are not built/installed when building it as part of the runtimes. The
issue seems to be that the `CAN_TARGET_*` variables are only computed once,
which causes them to be incorrectly set to `FALSE` when building `compiler-rt`
as part of the runtimes. The patch in `CompilerRTUtils.cmake.diff` forces the
variables to be rechecked during the build. This patch may no longer be
necessary after updating to LLVM 19+.
