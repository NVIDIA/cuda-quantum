
/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/CompiledModule.h"
#include "common/ExecutionContext.h"
#include "common/KernelArgs.h"
#include "cudaq/algorithms/msm/policy.h"
#include "cudaq/platform.h"
#include "cudaq/platform/qpu.h"
#include "cudaq/qis/execution_manager.h"
#include "cudaq/runtime/logger/logger.h"
#include "cudaq/utils/cudaq_utils.h"
#include <stdexcept>
#include <utility>
#include <variant>

#ifndef CUDAQ_DISABLE_JIT_COMPILER
namespace cudaq_internal::compiler {
cudaq::CompiledModule
compileModule(std::unique_ptr<cudaq::CompileTarget> target,
              const cudaq::SourceModule &src, cudaq::KernelArgs args,
              bool isEntryPoint = true);
} // namespace cudaq_internal::compiler
#endif

namespace cudaq {
namespace detail {

/// @brief Execute the given function within the given execution context.
template <typename Policy, typename Callable, typename... Args>
auto launch(const Policy &policy, std::size_t qpu_id, ExecutionContext &ctx,
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
    CompiledModule compiled;
    if (const auto *source = std::get_if<SourceModule>(&module)) {
#ifdef CUDAQ_DISABLE_JIT_COMPILER
      // If JIT compilation is disabled, compilation is a no-op. QPUs may throw
      // an error if they expect a JIT-compiled module.
      CUDAQ_INFO("JIT compilation is disabled. Compilation is a no-op.");
      compiled = CompiledModule{*source};
#else
      CUDAQ_INFO("No compiled module found. Compiling.");
      std::unique_ptr<cudaq::CompileTarget> target;
      if constexpr (requires { policy.inner; }) {
        target = cudaq::get_compile_target(policy.inner);
      } else {
        target = cudaq::get_compile_target(policy);
      }
      compiled = cudaq_internal::compiler::compileModule(std::move(target),
                                                         *source, args);
#endif
    } else {
      CUDAQ_INFO("Found compiled module. Skipping compilation.");
      compiled = std::get<CompiledModule>(module);
    }
    CUDAQ_INFO("Launching kernel.");
    result = qpu.launchKernel(policy, compiled, args);
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

/// @brief Execute the given function with an MSM size policy.
template <typename Callable, typename... Args>
msm_dimensions launch(msm_size_policy policy, Callable &&f, Args &&...args) {
  auto &platform = get_platform();
  ExecutionContext ctx(msm_size_policy::name);
  const size_t qpu_id = 0;
  ctx.qpuId = qpu_id;
  policy.kernelName = cudaq::getKernelName(f);
  ctx.kernelName = policy.kernelName;
  policy.noiseModel = platform.get_noise(qpu_id);
  ctx.noiseModel = policy.noiseModel;
  return detail::launch(policy, qpu_id, ctx, platform,
                        std::forward<Callable>(f), std::forward<Args>(args)...);
}

/// @brief Execute the given function with an MSM policy.
template <typename Callable, typename... Args>
msm_result launch(msm_policy policy, Callable &&f, Args &&...args) {
  auto &platform = get_platform();
  ExecutionContext ctx(msm_policy::name);
  const size_t qpu_id = 0;
  ctx.qpuId = qpu_id;
  policy.kernelName = cudaq::getKernelName(f);
  ctx.kernelName = policy.kernelName;
  policy.noiseModel = platform.get_noise(qpu_id);
  ctx.noiseModel = policy.noiseModel;
  ctx.msm_dimensions = policy.dimensions;
  return detail::launch(policy, qpu_id, ctx, platform,
                        std::forward<Callable>(f), std::forward<Args>(args)...);
}

} // namespace cudaq
