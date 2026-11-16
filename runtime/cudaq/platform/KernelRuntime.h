/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/ThunkInterface.h"
#include <vector>

namespace cudaq {
class ExecutionManager;

namespace detail {
/// Host entry points needed by nvq++ output. The Python loader captures these
/// from its local runtime before exposing the facade globally. Function
/// pointers preserve the active runtime instance without linking its LLVM
/// dependencies.
struct KernelRuntimeCallbacks {
  KernelThunkResultType (*alt)(const char *, KernelThunkType, void *,
                               std::uint64_t, std::uint64_t);
  KernelThunkResultType (*streamlined)(const char *,
                                       const std::vector<void *> &);
  KernelThunkResultType (*hybrid)(const char *, KernelThunkType, void *,
                                  std::uint64_t, std::uint64_t,
                                  const std::vector<void *> &);
  ExecutionManager *(*executionManager)();
  ExecutionManager *(*registeredExecutionManager)();
  ExecutionManager *(*contextExecutionManager)();
  void (*disableYamlTargetConfig)();
  void (*setTargetBackend)(const char *);
};

using InitializeKernelRuntime = void (*)(const KernelRuntimeCallbacks &);
} // namespace detail
} // namespace cudaq
