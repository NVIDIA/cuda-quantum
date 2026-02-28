/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "common/ThunkInterface.h"
#include <memory>
#include <string>
#include <vector>

namespace cudaq {

class JitEngine;

/// A unique_ptr with a plain function pointer as destructor, allowing
/// type-erased ownership of forward-declared (incomplete) types.
template <typename T>
using OpaquePtr = std::unique_ptr<T, void (*)(T *)>;

template <typename T, typename... Args>
OpaquePtr<T> makeOpaquePtr(Args &&...args) {
  return OpaquePtr<T>(new T(std::forward<Args>(args)...),
                      [](T *p) { delete p; });
}

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

  void (*getEntryPoint() const)();

  const JitEngine &getEngine() const;

private:
  CompiledKernel(OpaquePtr<JitEngine> engine, std::string kernelName,
                 void (*entryPoint)(), bool hasResult);

  // Use the following factory function in JIT.h to construct CompiledKernels.
  friend CompiledKernel createCompiledKernel(JitEngine engine,
                                             std::string kernelName,
                                             bool hasResult);

  OpaquePtr<JitEngine> engine;
  std::string name;
  void (*entryPoint)();
  bool hasResult;
};

} // namespace cudaq
