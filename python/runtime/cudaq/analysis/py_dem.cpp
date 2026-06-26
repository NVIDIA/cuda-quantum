/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_dem.h"
#include "common/ExecutionContext.h"
#include "common/NoiseModel.h"
#include "runtime/cudaq/platform/py_alt_launch_kernel.h"
#include "cudaq/algorithms/dem.h"
#include "cudaq/platform.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <optional>
#include <string>

using namespace cudaq;

/// @brief Parse a Python dict of DEM options into a `cudaq::dem_options`
/// struct.
///
/// Recognised keys (all optional):
///   - decompose_errors (bool)
///   - fold_loops (bool)
///   - allow_gauge_detectors (bool)
///   - approximate_disjoint_errors_threshold (float)
///   - ignore_decomposition_failures (bool)
///   - block_decomposition_from_introducing_remnant_edges (bool)
static cudaq::dem_options parseDemOptions(const nanobind::dict &d) {
  cudaq::dem_options opts;
  for (auto [k, v] : d) {
    std::string key = nanobind::cast<std::string>(k);
    if (key == "decompose_errors")
      opts.decompose_errors = nanobind::cast<bool>(v);
    else if (key == "fold_loops")
      opts.fold_loops = nanobind::cast<bool>(v);
    else if (key == "allow_gauge_detectors")
      opts.allow_gauge_detectors = nanobind::cast<bool>(v);
    else if (key == "approximate_disjoint_errors_threshold")
      opts.approximate_disjoint_errors_threshold = nanobind::cast<double>(v);
    else if (key == "ignore_decomposition_failures")
      opts.ignore_decomposition_failures = nanobind::cast<bool>(v);
    else if (key == "block_decomposition_from_introducing_remnant_edges")
      opts.block_decomposition_from_introducing_remnant_edges =
          nanobind::cast<bool>(v);
    else
      throw std::invalid_argument("dem_options: unknown key '" + key + "'");
  }
  return opts;
}

static std::string dem_from_kernel_impl(const std::string &kernelName,
                                        MlirModule kernelMod,
                                        std::optional<noise_model> noise,
                                        nanobind::dict dem_options,
                                        nanobind::args args) {
  auto &platform = cudaq::get_platform();
  args = simplifiedValidateInputArguments(args);

  const cudaq::noise_model *noisePtr = noise ? &(*noise) : nullptr;
  const cudaq::dem_options opts = parseDemOptions(dem_options);
  return cudaq::detail::runDemFromKernel(
      kernelName, platform, noisePtr,
      [&]() {
        [[maybe_unused]] auto result =
            cudaq::marshal_and_launch_module(kernelName, kernelMod, args);
      },
      opts);
}

void cudaq::bindDemFromKernel(nanobind::module_ &mod) {
  mod.def("dem_from_kernel_impl", dem_from_kernel_impl, nanobind::arg(),
          nanobind::arg(), nanobind::arg().none(), nanobind::arg("dem_options"),
          nanobind::arg(), "See python documentation for dem_from_kernel.");
}
