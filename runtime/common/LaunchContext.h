/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/CompiledModule.h"
#include "common/KernelArgs.h"
#include "common/LaunchCompileConfig.h"
#include <cstddef>
#include <functional>
#include <utility>

namespace cudaq {

/// What a kernel launch needs from the algorithm that started it.
///
/// An algorithm such as `sample` or `observe` resolves this from its launch
/// policy before it calls the user's kernel. The kernel launch entry points
/// receive it through `detail::claimLaunchContext` and pass it on by
/// reference. It is the only state that crosses from the algorithm to the
/// kernel launch: the algorithm's `ExecutionContext` is only installed by
/// `launch`, while the kernel runs.
struct LaunchContext {
  /// The QPU to launch on.
  std::size_t qpuId = 0;

  /// The target and options resolved from the launch policy.
  LaunchCompileConfig compileConfig;

  /// Launch a kernel under the launch policy, within the algorithm's
  /// execution context. When empty, the kernel is launched on the platform
  /// without a policy.
  std::function<void(const AnyModule &module, const KernelArgs &args)> launch;
};

namespace detail {

/// The launch context handed from an algorithm to the next kernel launch on
/// this thread.
///
/// It only exists to cross the call into the user's kernel, which has the
/// user's signature and cannot take the context as an argument. Use
/// `LaunchContextHandoff` and `claimLaunchContext` rather than accessing it
/// directly.
extern thread_local LaunchContext *pendingLaunchContext;

/// Hand \p ctx to the next kernel launch on this thread.
///
/// Only one launch context can be pending at a time: handing over a second one
/// before the first is claimed throws. On scope exit, normal or exceptional, a
/// launch context nobody claimed is withdrawn.
class [[nodiscard]] LaunchContextHandoff {
public:
  explicit LaunchContextHandoff(LaunchContext &ctx);
  ~LaunchContextHandoff() { pendingLaunchContext = nullptr; }

  LaunchContextHandoff(const LaunchContextHandoff &) = delete;
  LaunchContextHandoff &operator=(const LaunchContextHandoff &) = delete;
};

/// Claim the launch context handed over by the algorithm, leaving none.
///
/// Returns null when no algorithm handed one over, for example when the kernel
/// is called directly or under `quantum_platform::with_execution_context`.
[[nodiscard]] inline LaunchContext *claimLaunchContext() noexcept {
  return std::exchange(pendingLaunchContext, nullptr);
}

} // namespace detail

extern "C" {
/// `detail::claimLaunchContext` for the kernel launch stubs that nvq++
/// generates. A stub claims the launch context right before it calls a kernel
/// launch entry point, and passes it on as an argument.
[[nodiscard]] LaunchContext *__cudaq_claimLaunchContext();
} // extern "C"
} // namespace cudaq
