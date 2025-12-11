/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
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
void bindPyToStim(py::module &mod, LinkedLibraryHolder &holder) {
  mod.def(
      "to_stim",
      [&](py::object kernel, py::args args,
          std::optional<noise_model> noise_model) {
        if (py::hasattr(kernel, "compile"))
          kernel.attr("compile")();
        // Verify that the target is stim
        if (holder.getTarget().name != "stim")
          throw std::runtime_error(
              "Target is not stim, cannot convert to Stim circuit");
        auto &platform = cudaq::get_platform();
        auto kernelName = kernel.attr("name").cast<std::string>();
        auto kernelMod = kernel.attr("module").cast<MlirModule>();
        args = simplifiedValidateInputArguments(args);
        std::unique_ptr<OpaqueArguments> argData(
            toOpaqueArgs(args, kernelMod, kernelName));

        auto ctx = std::make_unique<ExecutionContext>("to_stim", 1);
        ctx->kernelName = kernelName;
        if (noise_model.has_value()) {
          if (platform.is_remote())
            throw std::runtime_error(
                "Noise model is not supported on remote platforms.");
          platform.set_noise(&noise_model.value());
        }

        platform.set_exec_ctx(ctx.get());
        pyAltLaunchKernel(kernelName, kernelMod, *argData, {});
        platform.reset_exec_ctx();

        std::stringstream ss;
        for (auto &instruction : ctx->kernelTrace) {
          ss << instruction.name << "\n";
        }
        if (noise_model.has_value()) {
          platform.reset_noise();
        }
        return ss.str();
      },
      py::arg("kernel"), py::kw_only(), py::arg("noise_model") = py::none(),
      R"#(Convert the given quantum kernel expression to a Stim circuit.

Args:
  kernel (:class:`Kernel`): The :class:`Kernel` to convert to a Stim circuit.
  *arguments (Optional[Any]): The concrete values to evaluate the kernel 
    function at. Leave empty if the kernel doesn't accept any arguments.
  noise_model (Optional[NoiseModel]): The optional noise model to add noise to the kernel execution.

Returns:
  :class:`str`:
  A Stim circuit for the :class:`Kernel`.)#");
}
} // namespace cudaq
