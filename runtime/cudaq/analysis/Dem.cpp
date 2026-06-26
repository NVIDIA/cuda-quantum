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
                             const cudaq::dem_options &options,
                             const std::string &plugin_name,
                             cudaq::M2DSparseMatrix *m2d_out,
                             cudaq::M2OSparseMatrix *m2o_out) {

  if (cudaq::kernelHasConditionalFeedback(kernelName))
    throw std::runtime_error(
        "`cudaq::dem_from_kernel`: kernel '" + kernelName +
        "' branches on a measurement result. DEM analysis not supported.");

  cudaq::ExecutionContext ctx("dem");
  ctx.kernelName = kernelName;
  ctx.qpuId = cudaq::getCurrentQpuId();
  ctx.asyncExec = false;
  ctx.dem_opts = options;
  if (noise)
    ctx.noiseModel = noise;
  if (m2d_out || m2o_out)
    ctx.dem_opts.compute_measurement_matrices = true;

  // RAII: claim the thread-local analysis-simulator slot backed by the `stim`
  // plugin. The scope starts from a clean simulator and releases the override
  // on every exit path.
  auto demScope = nvqir::dem::make_scope(plugin_name);

  platform.with_execution_context(ctx, kernel);

  if (m2d_out) {
    m2d_out->num_measurements =
        ctx.dem_opts.measurement_matrices.num_measurements;
    m2d_out->rows = std::move(ctx.dem_opts.measurement_matrices.det_rows);
  }
  if (m2o_out) {
    m2o_out->num_measurements =
        ctx.dem_opts.measurement_matrices.num_measurements;
    m2o_out->rows = std::move(ctx.dem_opts.measurement_matrices.obs_rows);
  }

  return std::move(ctx.dem_text);
}

} // namespace cudaq::detail
