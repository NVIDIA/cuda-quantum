/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/run.h"
#include "runtime/cudaq/platform/py_alt_launch_kernel.h"
#include "utils/OpaqueArguments.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <string>
#include <tuple>
#include <vector>

namespace cudaq {
namespace details {

std::vector<py::object> readRunResults(mlir::ModuleOp module,
                                       mlir::func::FuncOp kernelFuncOp,
                                       mlir::Type ty, RunResultSpan &results,
                                       std::size_t count) {
  std::vector<py::object> ret;
  std::size_t byteSize = results.lengthInBytes / count;
  for (std::size_t i = 0; i < results.lengthInBytes; i += byteSize) {
    py::object obj =
        convertResult(module, kernelFuncOp, ty, results.data + i, byteSize);
    ret.push_back(obj);
  }
  return ret;
}

static std::tuple<std::string, MlirModule, OpaqueArguments *,
                  mlir::func::FuncOp>
getKernelLaunchParameters(py::object &kernel, py::args args) {
  if (py::len(kernel.attr("arguments")) != args.size())
    throw std::runtime_error("Invalid number of arguments passed to run:" +
                             std::to_string(args.size()) + " expected " +
                             std::to_string(py::len(kernel.attr("arguments"))));

  if (py::hasattr(kernel, "compile"))
    kernel.attr("compile")();

  auto kernelName = kernel.attr("name").cast<std::string>();
  auto kernelMod = kernel.attr("module").cast<MlirModule>();
  args = simplifiedValidateInputArguments(args);
  auto *argData = toOpaqueArgs(args, kernelMod, kernelName);

  auto funcOp = getKernelFuncOp(kernelMod, kernelName);
  return {kernelName, kernelMod, argData, funcOp};
}

std::vector<py::object> pyRunTheKernel(const std::string &name,
                                       MlirModule module, func::FuncOp funcOp,
                                       cudaq::OpaqueArguments &runtimeArgs,
                                       cudaq::quantum_platform &platform,
                                       std::size_t shots_count) {

  auto returnTypes = funcOp.getResultTypes();
  if (returnTypes.empty() || returnTypes.size() > 1)
    throw std::runtime_error(
        "`cudaq.run` only supports kernels that return a value.");

  auto returnTy = returnTypes[0];
  // Disallow returning list / vectors from entry-point kernels.
  if (returnTy.isa<cudaq::cc::StdvecType>()) {
    throw std::runtime_error("`cudaq.run` does not yet support returning "
                             "`list` from entry-point kernels.");
  }

  auto mod = unwrap(module);

  auto [rawArgs, size, returnOffset, thunk] =
      pyAltLaunchKernelBase(name, module, returnTy, runtimeArgs, {}, 0, false);

  auto results = details::runTheKernel(
      [&]() mutable {
        pyLaunchKernel(name, thunk, mod, runtimeArgs, rawArgs, size,
                       returnOffset, {});
      },
      platform, name, shots_count);

  std::free(rawArgs);
  return readRunResults(mod, funcOp, returnTy, results, shots_count);
}

} // namespace details

/// @brief Run `cudaq::run` on the provided kernel.
std::vector<py::object> pyRun(py::object &kernel, py::args args,
                              std::size_t shots_count,
                              std::optional<noise_model> noise_model) {
  if (shots_count == 0)
    return {};

  auto [name, module, argData, func] =
      details::getKernelLaunchParameters(kernel, args);

  auto mod = unwrap(module);
  mod->setAttr(runtime::enableCudaqRun, mlir::UnitAttr::get(mod->getContext()));

  auto &platform = get_platform();
  if (noise_model.has_value()) {
    if (platform.is_remote())
      throw std::runtime_error(
          "Noise model is not supported on remote platforms.");
    // Launch the kernel in the appropriate context.
    platform.set_noise(&noise_model.value());
  }

  auto results = details::pyRunTheKernel(name, module, func, *argData, platform,
                                         shots_count);
  delete argData;

  if (noise_model.has_value())
    platform.reset_noise();

  mod->removeAttr(runtime::enableCudaqRun);
  return results;
}

/// @brief Bind the run cudaq function.
void bindPyRun(py::module &mod) {
  mod.def("run", &pyRun, py::arg("kernel"), py::kw_only(),
          py::arg("shots_count") = 1000, py::arg("noise_model") = py::none(),
          R"#()#");
}
} // namespace cudaq
