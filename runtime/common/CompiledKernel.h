/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "common/JIT.h"
#include "common/ThunkInterface.h"
#include <string>
#include <vector>

namespace cudaq {

/// @brief A compiled, ready-to-execute kernel.
///
/// This type does not have a dependency on MLIR (or LLVM) as it only keeps
/// type-erased pointers to JIT-related types.
///
/// The constructor is private; use the factory function in
/// `runtime/common/JIT.h` to construct instances.
class CompiledKernel {
public:
  /// Pre-computed result metadata, set at build time. Used at execution time
  /// to allocate a result buffer and convert the raw result to a Python object.
  struct ResultInfo {
    /// Opaque pointer to the mlir::Type of the result. Obtained via
    /// mlir::Type::getAsOpaquePointer(). The underlying storage is owned by
    /// the MLIRContext, not by us — no allocation or cleanup needed.
    const void *typeOpaquePtr = nullptr;

    /// Size (in bytes) of the buffer needed to hold the result value.
    /// Pre-computed from the MLIR type at build time.
    std::size_t bufferSize = 0;

    /// Pre-computed struct field offsets (from getTargetLayout). Only non-empty
    /// for struct return types.
    std::vector<std::size_t> fieldOffsets;

    /// Whether this kernel has a result that must be marshaled.
    bool hasResult() const { return typeOpaquePtr != nullptr; }
  };

  /// @brief Execute a fully specialized kernel (no external args needed).
  ///
  /// Allocates a result buffer on-the-fly if the kernel has a return type.
  /// Throws if the kernel has unspecialized parameters (argsCreator is set);
  /// use execute(rawArgs) instead for partially specialized kernels.
  KernelThunkResultType execute() const;

  /// @brief Execute the JIT-ed kernel with caller-provided arguments.
  ///
  /// Use this for partially specialized kernels (e.g. variational args) where
  /// the caller provides the remaining parameter values. If the kernel has a
  /// return type, the caller must have appended a result buffer as the last
  /// element of \p rawArgs.
  KernelThunkResultType execute(const std::vector<void *> &rawArgs) const;

  /// Whether the kernel is fully specialized (all args inlined, no
  /// argsCreator).
  bool isFullySpecialized() const { return argsCreator == nullptr; }

  /// Access the pre-computed result metadata.
  const ResultInfo &getResultInfo() const { return resultInfo; }

  // TODO: remove these two methods once the CompiledKernel is returned to
  // Python.
  void (*getEntryPoint() const)();
  JitEngine getEngine() const;

private:
  CompiledKernel(JitEngine engine, std::string kernelName, void (*entryPoint)(),
                 int64_t (*argsCreator)(const void *, void **),
                 ResultInfo resultInfo);

  // Use the following factory function (compiled into cudaq-mlir-runtime) to
  // construct CompiledKernels.
  friend CompiledKernel createCompiledKernel(JitEngine engine,
                                             std::string kernelName,
                                             bool hasVariationalArgs,
                                             ResultInfo resultInfo);

  JitEngine engine;
  std::string name;

  // Function pointers into JITEngine
  void (*entryPoint)();
  int64_t (*argsCreator)(const void *, void **);

  ResultInfo resultInfo;
};

CompiledKernel createCompiledKernel(JitEngine engine, std::string kernelName,
                                    bool hasVariationalArgs,
                                    CompiledKernel::ResultInfo resultInfo);
} // namespace cudaq
