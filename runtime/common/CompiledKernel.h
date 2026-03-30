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
  /// @brief Execute the JIT-ed kernel.
  ///
  /// If the kernel has a return type, the caller must have appended a result
  /// buffer as the last element of \p rawArgs.
  KernelThunkResultType execute(const std::vector<void *> &rawArgs) const;

  // TODO: remove the following two methods once the CompiledKernel is returned
  // to Python.
  /// @brief Get the entry point of the kernel as a function pointer.
  ///
  /// The returned function pointer will expect different arguments depending
  /// on the kernel:
  ///  - if the kernel returns a value and/or is not fully specialized, the
  ///    entry point will expect a pointer to a buffer storing the packed
  ///    arguments and result.
  ///  - otherwise, the entry point will not expect any arguments.
  ///
  /// Prefer using `CompiledKernel::execute` instead of calling this function as
  /// it will handle the buffer and argument packing automatically.
  void (*getEntryPoint() const)();
  JitEngine getEngine() const;

private:
  CompiledKernel(JitEngine engine, std::string kernelName, void (*entryPoint)(),
                 int64_t (*argsCreator)(const void *, void **), bool hasResult);

  // Use the following factory function (compiled into cudaq-mlir-runtime) to
  // construct CompiledKernels.
  friend CompiledKernel createCompiledKernel(JitEngine engine,
                                             std::string kernelName,
                                             bool hasResult,
                                             bool isFullySpecialized);

  JitEngine engine;
  std::string name;

  // Function pointers into JITEngine
  void (*entryPoint)();
  int64_t (*argsCreator)(const void *, void **);

  bool hasResult;
};

/// @brief Create a CompiledKernel from JIT-compiled code.
///
/// `hasResult` and `isFullySpecialized` affect how the mangled kernel name
/// and the arguments buffer passed to the compiled kernel are constructed.
/// See `CompiledKernel::getEntryPoint` for more details.
CompiledKernel createCompiledKernel(JitEngine engine, std::string kernelName,
                                    bool hasResult, bool isFullySpecialized);
} // namespace cudaq
