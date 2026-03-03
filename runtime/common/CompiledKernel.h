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

  // TODO: remove these two methods once the CompiledKernel is returned to
  // Python.
  void (*getEntryPoint() const)();
  const JitEngine getEngine() const;

private:
  CompiledKernel(JitEngine engine, std::string kernelName, void (*entryPoint)(),
                 bool hasResult);

  // Use the following factory function (compiled into cudaq-mlir-runtime) to
  // construct CompiledKernels.
  friend CompiledKernel createCompiledKernel(JitEngine engine,
                                             std::string kernelName,
                                             bool hasResult);

  JitEngine engine;
  std::string name;
  void (*entryPoint)();
  bool hasResult;
};

CompiledKernel createCompiledKernel(JitEngine engine, std::string kernelName,
                                    bool hasResult);
} // namespace cudaq
