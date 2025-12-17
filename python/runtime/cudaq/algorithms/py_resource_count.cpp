/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/Resources.h"
#include "runtime/cudaq/platform/py_alt_launch_kernel.h"
#include "utils/LinkedLibraryHolder.h"
#include "utils/OpaqueArguments.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <fmt/core.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace cudaq {
void bindCountResources(py::module &mod) {
  mod.def(
      "estimate_resources",
      [&](py::object kernel, py::args args,
          std::optional<std::function<bool()>> choice) {
        if (py::hasattr(kernel, "compile"))
          kernel.attr("compile")();
        auto &platform = cudaq::get_platform();
        auto kernelName = kernel.attr("name").cast<std::string>();
        auto kernelMod = kernel.attr("module").cast<MlirModule>();
        args = simplifiedValidateInputArguments(args);
        std::unique_ptr<OpaqueArguments> argData(
            toOpaqueArgs(args, kernelMod, kernelName));

        ExecutionContext ctx("resource-count", 1);
        ctx.kernelName = kernelName;
        // Indicate that this is not an async exec
        ctx.asyncExec = false;

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
        platform.with_execution_context(ctx, [&]() {
          pyAltLaunchKernel(kernelName, kernelMod, *argData, {});
        });

        // Save and clone counts data
        Resources counts = *python::detail::getResourceCounts();

        // Switch simulators back
        python::detail::stopUsingResourceCounterSimulator();

        return counts;
      },
      py::arg("kernel"), py::kw_only(), py::arg("choice") = std::nullopt,
      R"#(Performs resource counting on the given quantum kernel
expression and returns an accounting of how many times each gate
was applied, in addition to the total number of gates and qubits used.

Args:
  choice (Any): A choice function called to determine the outcome of
    measurements, in case control flow depends on measurements. Should
    only return either `True` or `False`. Invoking the kernel within
    the choice function is forbidden. Default: returns `True` or `False`
    with 50% probability.
  kernel (:class:`Kernel`): The :class:`Kernel` to count resources on
  *arguments (Optional[Any]): The concrete values to evaluate the kernel 
    function at. Leave empty if the kernel doesn't accept any arguments.

Returns:
  :class:`Resources`:
  A dictionary containing the resource count results for the :class:`Kernel`.)#");
}
} // namespace cudaq
