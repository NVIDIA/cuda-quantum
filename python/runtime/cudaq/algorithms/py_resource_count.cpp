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

  // RAII: scope is released (and the resource-counter state cleared) on
  // every exit path, including exceptions thrown by the JIT'd kernel.
  auto rcScope = nvqir::resource_counter::make_scope(std::move(*choice));
  platform.with_execution_context(ctx, [&]() {
    [[maybe_unused]] auto result =
        cudaq::marshal_and_launch_module(kernelName, kernelMod, args);
  });
  return nvqir::resource_counter::get_counts(rcScope);
}

void cudaq::bindCountResources(nanobind::module_ &mod) {
  mod.def("estimate_resources_impl", estimate_resources_impl, nanobind::arg(),
          nanobind::arg(), nanobind::arg().none(), nanobind::arg(),
          "See python documentation for estimate_resources.");
}
