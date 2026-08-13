/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_resource_count.h"
#include "common/Resources.h"
#include "nvqir/resourcecounter/ResourceCounterScope.h"
#include "runtime/cudaq/platform/py_alt_launch_kernel.h"
#include "cudaq/algorithms/estimate/policy.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>

using namespace cudaq;

static Resources
estimate_resources_impl(const std::string &kernelName, MlirModule kernelMod,
                        std::optional<std::function<bool()>> choice,
                        nanobind::args args) {
  auto &platform = cudaq::get_platform();
  args = simplifiedValidateInputArguments(args);

  ExecutionContext ctx("resource-count", 1);
  ctx.kernelName = kernelName;
  // Indicate that this is not an async exec
  ctx.asyncExec = false;

  // Set the choice function for the simulator
  if (!choice) {
    auto seed = cudaq::get_random_seed();
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> rand(0, 1);
    choice = [gen = std::move(gen), rand = std::move(rand)]() mutable {
      return rand(gen);
    };
  }

  estimate_policy policy{
      .kernelName = kernelName,
      .choice = *std::move(choice),
  };
  auto result = detail::launch(policy, 0, ctx, platform, [&]() {
    // Pass nullptr for the compiled slot to disable JIT-artifact caching:
    // the resource-counter hooks are installed in the scope above and only
    // fire while the kernel is freshly JIT-compiled. A cached binary would
    // bypass them.
    [[maybe_unused]] auto result =
        cudaq::marshal_and_launch_module(kernelName, kernelMod, args);
  });
  return result.get_resources();
}

void cudaq::bindCountResources(nanobind::module_ &mod) {
  mod.def("estimate_resources_impl", estimate_resources_impl,
          nanobind::arg("kernel_name"), nanobind::arg("kernel_mod"),
          nanobind::arg("choice").none(), nanobind::arg("args"),
          "See python documentation for estimate_resources.");
}
