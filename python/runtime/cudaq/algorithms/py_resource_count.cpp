/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_resource_count.h"
#include "common/Resources.h"
#include "runtime/cudaq/platform/py_alt_launch_kernel.h"
#include "utils/LinkedLibraryHolder.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include <pybind11/functional.h>

namespace py = pybind11;

using namespace cudaq;

static Resources estimate_resources_impl(
    const std::string &kernelName, MlirModule kernelMod, MlirType returnTy,
    std::optional<std::function<bool()>> choice, py::args args) {
  auto &platform = cudaq::get_platform();
  args = simplifiedValidateInputArguments(args);

  auto ctx = std::make_unique<ExecutionContext>("resource-count", 1);
  ctx->kernelName = kernelName;
  // Indicate that this is not an async exec
  ctx->asyncExec = false;

  // Use the resource counter simulator
  python::detail::switchToResourceCounterSimulator();

  // Set the choice function for the simulator
  if (!choice) {
    auto seed = cudaq::get_random_seed();
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> rand(0, 1);
    choice = [gen = std::move(gen), rand = std::move(rand)]() mutable {
      return rand(gen);
    };
  }
  python::detail::setChoiceFunction(*choice);

  // Set the platform
  platform.set_exec_ctx(ctx.get());
  try {
    // Launch the kernel.
    [[maybe_unused]] auto result =
        cudaq::marshal_and_launch_module(kernelName, kernelMod, returnTy, args);

    // Reset the platform.
    platform.reset_exec_ctx();

    // Save and clone counts data
    Resources counts = *python::detail::getResourceCounts();
    // Switch simulators back
    python::detail::stopUsingResourceCounterSimulator();
    return counts;
  } catch (std::exception &e) {
    // Reset the platform.
    platform.reset_exec_ctx();
    // Switch simulators back
    python::detail::stopUsingResourceCounterSimulator();
    throw e;
  }
}

void cudaq::bindCountResources(py::module &mod) {
  mod.def("estimate_resources_impl", estimate_resources_impl,
          "See python documentation for estimate_resources.");
}
