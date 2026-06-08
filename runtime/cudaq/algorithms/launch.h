
/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/ExecutionContext.h"
#include "common/KernelArgs.h"
#include "cudaq/platform.h"
#include "cudaq/platform/qpu.h"
#include "cudaq/qis/execution_manager.h"
#include "cudaq/runtime/logger/logger.h"
#include "cudaq/utils/cudaq_utils.h"
#include <stdexcept>

namespace cudaq {
namespace detail {

/// @brief Execute the given function within the given execution context.
template <typename Policy, typename Callable, typename... Args>
auto launch(Policy &policy, std::size_t qpu_id, ExecutionContext &ctx,
            quantum_platform &platform, Callable &&f, Args &&...args)
    -> Policy::result_type {

  // Python builds without CUDAQ_LIBRARY_MODE defined, so we need to check for
  // it at runtime
  bool library_mode = platform.is_library_mode();
#ifdef CUDAQ_LIBRARY_MODE
  library_mode = true;
#endif
  if (library_mode) {
    // async_policy (async_policy_wrapper) + library mode is unreachable:
    if constexpr (requires { policy.inner; })
      throw std::logic_error("async policy must not reach library-mode path");
    else {
      CUDAQ_INFO("Launching kernel in library mode with policy {}",
                 policy.name);
      return detail::with_policy_and_ctx(policy, ctx, [&]() {
        return cudaq::ExecutionManager::with_default_em(
            policy, [&]() { f(std::forward<Args>(args)...); });
      });
    }
  }

  typename Policy::result_type result;
  auto &qpu = platform.getQPU(qpu_id);
  ctx.executeKernelApi = [&qpu, &result, &policy](const AnyModule &module,
                                                  const KernelArgs &args) {
    result = qpu.launchKernel(policy, module, args);
  };

  if constexpr (requires { policy.inner; })
    CUDAQ_INFO("Launching kernel in async mode with policy {}",
               policy.inner.name);
  else
    CUDAQ_INFO("Launching kernel in sync mode with policy {}", policy.name);

  detail::try_finally(
      [&] {
        detail::with_policy_and_ctx(policy, ctx, std::forward<Callable>(f),
                                    std::forward<Args>(args)...);
      },
      [&] { ctx.executeKernelApi = nullptr; });
  return result;
}
} // namespace detail
} // namespace cudaq
