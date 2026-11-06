/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/CompiledModule.h"
#include "common/ExecutionContext.h"
#include "common/KernelArgs.h"
#include "common/Resources.h"
#include "nvqir/resourcecounter/ResourceCounterScope.h"
#include "cudaq/algorithms/policies.h"
#include "cudaq/qis/execution_manager.h"
#include "cudaq/utils/cudaq_utils.h"
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace nvqir {

/// @brief Launch-specific facts that are not part of the policy, typically
/// derived from the compiled module.
struct LaunchConfig {
  /// @brief The ID of the QPU running the simulation.
  std::size_t qpuId = 0;

  /// @brief Whether the kernel branches on measurement results.
  bool hasConditionalsOnMeasureResults = false;

  /// @brief Qubit reorder indices that undo the qubit mapping in the sample
  /// result. Empty means no reordering. Only used by `sample_policy`.
  std::vector<std::size_t> reorderIdx = {};

  /// @brief Resource counts pre-computed by the compiler. Only used by
  /// `estimate_policy`.
  const cudaq::Resources *resources = nullptr;
};

namespace detail {
/// @brief Fill \p ctx with what the simulator needs to know about the launch.
template <typename Policy>
void configureExecutionContext(cudaq::ExecutionContext &ctx,
                               const Policy &policy,
                               const LaunchConfig &config) {
  ctx.qpuId = config.qpuId;
  ctx.hasConditionalsOnMeasureResults = config.hasConditionalsOnMeasureResults;
  if constexpr (requires { policy.kernelName; })
    ctx.kernelName = policy.kernelName;
  // PTSBE resolves noise into the trace before execution: the simulator must
  // run noiseless.
  if constexpr (requires { policy.noiseModel; } &&
                !std::is_same_v<Policy, cudaq::ptsbe::sample_policy>)
    ctx.noiseModel = policy.noiseModel;

  if constexpr (std::is_same_v<Policy, cudaq::sample_policy>) {
    ctx.shots = policy.options.shots;
    ctx.explicitMeasurements = policy.options.explicit_measurements;
    ctx.batchIteration = policy.batchIteration;
    ctx.totalIterations = policy.totalIterations;
  } else if constexpr (std::is_same_v<Policy, cudaq::observe_policy>) {
    // Simulators read `static_cast<std::size_t>(-1)` as "no shots requested".
    ctx.shots = static_cast<std::size_t>(policy.options.shots);
    ctx.spin = policy.spin;
    ctx.numberTrajectories = policy.options.num_trajectories;
    ctx.batchIteration = policy.batchIteration;
    ctx.totalIterations = policy.totalIterations;
  } else if constexpr (std::is_same_v<Policy, cudaq::run_policy> ||
                       std::is_same_v<Policy, cudaq::ptsbe::sample_policy>) {
    ctx.shots = policy.shots;
  } else if constexpr (std::is_same_v<Policy, cudaq::estimate_policy>) {
    ctx.shots = 1;
  }
}

template <typename Policy, typename F>
typename Policy::result_type
launchWithContext(const Policy &policy, const LaunchConfig &config, F &&f) {
  cudaq::ExecutionContext ctx(Policy::name);
  configureExecutionContext(ctx, policy, config);

  auto *outerContext = cudaq::getExecutionContext();
  cudaq::detail::setExecutionContext(&ctx);
  auto result = cudaq::detail::try_finally(
      [&] {
        return cudaq::ExecutionManager::with_default_em(policy, [&] {
          if constexpr (std::is_same_v<Policy, cudaq::estimate_policy>)
            if (config.resources)
              nvqir::resource_counter::prepopulate(*config.resources);
          f();
        });
      },
      [outerContext] { cudaq::detail::setExecutionContext(outerContext); });

  if constexpr (std::is_same_v<Policy, cudaq::sample_policy>)
    if (!config.reorderIdx.empty())
      result.reorder(config.reorderIdx);
  return result;
}
} // namespace detail

/// @brief Run \p f, which executes a kernel, on the NVQIR simulator under
/// \p policy.
///
/// Installs the execution context the simulator reads for the duration of the
/// launch and drives the default execution manager through its configure /
/// finalize life cycle.
template <typename Policy, typename F>
typename Policy::result_type launch(const Policy &policy, LaunchConfig config,
                                    F &&f) {
  if constexpr (std::is_same_v<Policy, cudaq::estimate_policy>) {
    // RAII: the scope is released (and the resource-counter state cleared) on
    // every exit path, including exceptions thrown from the kernel.
    auto rcScope = nvqir::resource_counter::make_scope(policy.choice);
    return detail::launchWithContext(policy, config, std::forward<F>(f));
  } else {
    return detail::launchWithContext(policy, config, std::forward<F>(f));
  }
}

/// @brief Run the compiled binary stored in \p module on the NVQIR simulator
/// under \p policy.
///
/// Dispatches to `executeFunctionPtrBinary` when an AOT function-pointer
/// artifact is present, otherwise to `executeJitBinary`.
template <typename Policy>
typename Policy::result_type launch(const Policy &policy,
                                    const cudaq::CompiledModule &module,
                                    cudaq::KernelArgs args, std::size_t qpuId) {
  const auto &metadata = module.getMetadata();
  LaunchConfig config{.qpuId = qpuId,
                      .hasConditionalsOnMeasureResults =
                          metadata.hasConditionalsOnMeasureResults,
                      .resources = module.getResources()};
  // Put routed results back in the kernel's qubit order. Explicit measurements
  // are already in execution order.
  if constexpr (std::is_same_v<Policy, cudaq::sample_policy>)
    if (!policy.options.explicit_measurements)
      config.reorderIdx = metadata.reorderIdx;

  return launch(policy, std::move(config), [&module, &args] {
    if (auto fnPtr = module.getFunctionPtr()) {
      [[maybe_unused]] auto res = cudaq::executeFunctionPtrBinary(*fnPtr, args);
    } else {
      [[maybe_unused]] auto res = cudaq::executeJitBinary(module, args);
    }
  });
}

} // namespace nvqir
