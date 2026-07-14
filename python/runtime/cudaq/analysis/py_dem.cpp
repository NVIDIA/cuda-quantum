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
#include "cudaq/algorithms/dem.h"
#include "cudaq/platform.h"
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
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
///   - return_measurement_matrices (bool)
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
    else if (key == "return_measurement_matrices")
      opts.return_measurement_matrices = nanobind::cast<bool>(v);
    else
      throw std::invalid_argument("dem_options: unknown key '" + key + "'");
  }
  return opts;
}

static void construct_dem_policy(dem_policy *self,
                                 const std::string &kernelName,
                                 const noise_model *noise,
                                 const nanobind::dict &options) {
  new (self) dem_policy();
  self->kernelName = kernelName;
  self->noiseModel = noise;
  self->options = parseDemOptions(options);
}

static nanobind::object launch_dem(const dem_policy &policy,
                                   nanobind::callable callable) {
  auto result = cudaq::detail::launchDem(policy, cudaq::get_platform(),
                                         [&]() { callable(); });
  if (!policy.options.return_measurement_matrices)
    return nanobind::cast(std::move(result.dem));
  // Positional 4-tuple contract with `dem.py`: (dem_text, num_measurements,
  // det_rows, obs_rows).
  return nanobind::make_tuple(nanobind::cast(std::move(result.dem)),
                              nanobind::cast(result.m2d.num_measurements),
                              nanobind::cast(std::move(result.m2d.rows)),
                              nanobind::cast(std::move(result.m2o.rows)));
}

void cudaq::bindDemFromKernel(nanobind::module_ &mod) {
  nanobind::class_<dem_policy>(mod, "DemPolicy")
      .def("__init__", construct_dem_policy, nanobind::arg("kernel_name"),
           nanobind::arg("noise_model").none(), nanobind::arg("options"),
           nanobind::keep_alive<1, 3>())
      .def_prop_ro("kernel_name",
                   [](const dem_policy &policy) { return policy.kernelName; })
      .def_prop_ro("return_measurement_matrices", [](const dem_policy &policy) {
        return policy.options.return_measurement_matrices;
      });

  mod.def("launch_dem", launch_dem, "Policy based DEM launch.",
          nanobind::arg("policy"), nanobind::arg("callable"));
}
