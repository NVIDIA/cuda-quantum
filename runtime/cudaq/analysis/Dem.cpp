/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/dem.h"
#include "common/DeviceCodeRegistry.h"
#include "common/ExecutionContext.h"
#include "nvqir/dem/DemScope.h"
#include <stdexcept>

namespace cudaq::detail {

cudaq::dem_result launchDemPolicy(const cudaq::dem_policy &policy,
                                  cudaq::ExecutionContext &ctx,
                                  const dem_policy_launcher &launchPolicy,
                                  const std::string &plugin_name) {
  if (cudaq::kernelHasConditionalFeedback(policy.kernelName))
    throw std::runtime_error(
        "`cudaq::dem_from_kernel`: kernel '" + policy.kernelName +
        "' uses a measurement result in classical control flow or as a "
        "runtime operand (e.g. an observable index). DEM analysis not "
        "supported.");

  // RAII: claim the thread-local analysis-simulator slot backed by the
  // @p plugin_name plugin. The scope starts from a clean simulator and
  // releases the override on every exit path.
  auto demScope = nvqir::dem::make_scope(plugin_name);
  return launchPolicy(policy, ctx);
}

} // namespace cudaq::detail
