/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// This facade is loaded after libcudaq and its execution manager have been
// bound locally. Only the facade and the LLVM-free QIR runtime become global.
#include "KernelRuntime.h"
#include "cudaq/host_config.h"
#include <stdexcept>

namespace {
cudaq::detail::KernelRuntimeCallbacks callbacks{};

template <typename Callback>
Callback requireCallback(Callback callback) {
  if (!callback)
    throw std::runtime_error(
        "Import cudaq before invoking a compiled kernel library.");
  return callback;
}
} // namespace

extern "C" void cudaqInitializeKernelRuntime(
    const cudaq::detail::KernelRuntimeCallbacks &runtimeCallbacks) {
  callbacks = runtimeCallbacks;
}

namespace cudaq {
extern "C" {
KernelThunkResultType altLaunchKernel(const char *name, KernelThunkType kernel,
                                      void *args, std::uint64_t size,
                                      std::uint64_t resultOffset) {
  return requireCallback(callbacks.alt)(name, kernel, args, size, resultOffset);
}

KernelThunkResultType streamlinedLaunchKernel(const char *name,
                                              const std::vector<void *> &args) {
  return requireCallback(callbacks.streamlined)(name, args);
}

KernelThunkResultType hybridLaunchKernel(const char *name,
                                         KernelThunkType kernel, void *args,
                                         std::uint64_t size,
                                         std::uint64_t resultOffset,
                                         const std::vector<void *> &rawArgs) {
  return requireCallback(callbacks.hybrid)(name, kernel, args, size,
                                           resultOffset, rawArgs);
}
} // extern "C"

// nvq++'s host object can retain inline QIS definitions even though the kernel
// itself is lowered to QIR. Resolve their execution-manager references too.
ExecutionManager *getExecutionManagerInternal() {
  return requireCallback(callbacks.executionManager)();
}
ExecutionManager *getRegisteredExecutionManager() {
  return requireCallback(callbacks.registeredExecutionManager)();
}

namespace detail {
ExecutionManager *getExecutionManagerFromContext() {
  return requireCallback(callbacks.contextExecutionManager)();
}

YamlTargetConfigDisabler::YamlTargetConfigDisabler() {
  requireCallback(callbacks.disableYamlTargetConfig)();
}
TargetSetter::TargetSetter(const char *backend) {
  requireCallback(callbacks.setTargetBackend)(backend);
}
} // namespace detail
} // namespace cudaq
