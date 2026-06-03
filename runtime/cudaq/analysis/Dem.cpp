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
#include "common/NoiseModel.h"
#include "nvqir/dem/DemScope.h"
#include "cudaq/platform.h"
#include <stdexcept>
#include <string>
#include <utility>

namespace cudaq::detail {

std::string runDemFromKernel(const std::string &kernelName,
                             cudaq::quantum_platform &platform,
                             const cudaq::noise_model *noise,
                             const std::function<void()> &kernel,
                             const std::string &plugin_name) {

  if (cudaq::kernelHasConditionalFeedback(kernelName))
    throw std::runtime_error(
        "`cudaq::dem_from_kernel`: kernel '" + kernelName +
        "' branches on a measurement result. DEM analysis not supported.");

  cudaq::ExecutionContext ctx("dem");
  ctx.kernelName = kernelName;
  ctx.qpuId = cudaq::getCurrentQpuId();
  ctx.asyncExec = false;
  if (noise)
    ctx.noiseModel = noise;

  // RAII: claim the thread-local analysis-simulator slot backed by the `stim`
  // plugin. The scope starts from a clean simulator and releases the override
  // on every exit path.
  auto demScope = nvqir::dem::make_scope(plugin_name);

  platform.with_execution_context(ctx, kernel);

  return std::move(ctx.dem_text);
}

} // namespace cudaq::detail
