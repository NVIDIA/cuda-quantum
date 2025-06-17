/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ResourceCounts.h"
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
      "count_resources",
      [&](std::function<bool()> choice, py::object kernel, py::args args) {
        if (py::hasattr(kernel, "compile"))
          kernel.attr("compile")();
        auto &platform = cudaq::get_platform();
        auto kernelName = kernel.attr("name").cast<std::string>();
        auto kernelMod = kernel.attr("module").cast<MlirModule>();
        args = simplifiedValidateInputArguments(args);
        std::unique_ptr<OpaqueArguments> argData(toOpaqueArgs(args, kernelMod, kernelName));

        auto ctx = std::make_unique<ExecutionContext>("resourcecount", 1);
        ctx->kernelName = kernelName;
        // Indicate that this is not an async exec
        ctx->asyncExec = false;

        // Use the resource counter simulator
        __internal__::switchToResourceCounterSimulator();
        // Set the choice function for the simulator
        __internal__::setChoiceFunction(choice);

        // Set the platform
        platform.set_exec_ctx(ctx.get());

        pyAltLaunchKernel(kernelName, kernelMod, *argData, {});

        // Save and clone counts data
        auto counts = resource_counts(*__internal__::getResourceCounts());
        // Switch simulators back
        __internal__::stopUsingResourceCounterSimulator();

        return counts;
      },
      R"#(Performs resource counting on the given quantum kernel
expression and returns an accounting of how many times each gate
was applied, in addition to the total number of gates and qubits used.

Args:
  choice (Any): A choice function called to determine the outcome of
    measurements, in case control flow depends on measurements. Should
    only return either `True` or `False`
  kernel (:class:`Kernel`): The :class:`Kernel` to count resources on
  *arguments (Optional[Any]): The concrete values to evaluate the kernel 
    function at. Leave empty if the kernel doesn't accept any arguments.

Returns:
  :class:`ResourceCounts`:
  A dictionary containing the resource count results for the :class:`Kernel`.)#");
}
} // namespace cudaq
