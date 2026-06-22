/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_dem.h"
#include "common/NoiseModel.h"
#include "runtime/cudaq/platform/py_alt_launch_kernel.h"
#include "cudaq/algorithms/dem.h"
#include "cudaq/platform.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <optional>
#include <string>

using namespace cudaq;

static nanobind::object dem_from_kernel_impl(const std::string &kernelName,
                                             MlirModule kernelMod,
                                             std::optional<noise_model> noise,
                                             bool return_m2d,
                                             nanobind::args args) {
  auto &platform = cudaq::get_platform();
  args = simplifiedValidateInputArguments(args);

  const cudaq::noise_model *noisePtr = noise ? &(*noise) : nullptr;
  auto launch = [&]() {
    [[maybe_unused]] auto result =
        cudaq::marshal_and_launch_module(kernelName, kernelMod, args);
  };

  cudaq::M2DSparseMatrix m2d_storage;
  cudaq::M2DSparseMatrix *m2d_ptr = return_m2d ? &m2d_storage : nullptr;
  std::string dem_text = cudaq::detail::runDemFromKernel(
      kernelName, platform, noisePtr, launch, /*plugin_name=*/"stim", m2d_ptr);

  if (!return_m2d)
    return nanobind::cast(std::move(dem_text));

  return nanobind::make_tuple(nanobind::cast(std::move(dem_text)),
                              nanobind::cast(m2d_storage.num_measurements),
                              nanobind::cast(std::move(m2d_storage.rows)));
}

void cudaq::bindDemFromKernel(nanobind::module_ &mod) {
  mod.def("dem_from_kernel_impl", dem_from_kernel_impl, nanobind::arg(),
          nanobind::arg(), nanobind::arg().none(), nanobind::arg("return_m2d"),
          nanobind::arg(), "See python documentation for dem_from_kernel.");
}
