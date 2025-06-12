/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

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
void pyAltLaunchKernel(const std::string &, MlirModule, OpaqueArguments &,
                       const std::vector<std::string> &);

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
        auto *argData = toOpaqueArgs(args, kernelMod, kernelName);

        auto ctx = std::make_unique<ExecutionContext>("resourcecount", 1);
        ctx->kernelName = kernelName;
        ctx->choice = choice;

        cudaq::setResourceCountingSimulator();

        platform.set_exec_ctx(ctx.get());

        pyAltLaunchKernel(kernelName, kernelMod, *argData, {});

        cudaq::resetSimulator();

        return ctx->resourceCounts;
      },
      R"#(Asynchronously sample the state of the provided `kernel` at the 
specified number of circuit executions (`shots_count`).
When targeting a quantum platform with more than one QPU, the optional
`qpu_id` allows for control over which QPU to enable. Will return a
future whose results can be retrieved via `future.get()`.

Args:
  kernel (:class:`Kernel`): The :class:`Kernel` to execute `shots_count` 
    times on the QPU.
  *arguments (Optional[Any]): The concrete values to evaluate the kernel 
    function at. Leave empty if the kernel doesn't accept any arguments.
  shots_count (Optional[int]): The number of kernel executions on the 
    QPU. Defaults to 1000. Key-word only.
  explicit_measurements (Optional[bool]): A flag to indicate whether or not to 
    concatenate measurements in execution order for the returned sample result.
  qpu_id (Optional[int]): The optional identification for which QPU 
    on the platform to target. Defaults to zero. Key-word only.

Returns:
  :class:`AsyncSampleResult`: 
  A dictionary containing the measurement count results for the :class:`Kernel`.)#");
}
} // namespace cudaq
