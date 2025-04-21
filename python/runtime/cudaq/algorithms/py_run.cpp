/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/algorithms/run.h"
#include "utils/OpaqueArguments.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <string>
#include <tuple>
#include <vector>

#include <iostream>

namespace cudaq {

void pyAltLaunchKernel(const std::string &, MlirModule, OpaqueArguments &,
                       const std::vector<std::string> &);

namespace {

inline unsigned int byteSize(mlir::Type ty) {
  if (isa<ComplexType>(ty)) {
    auto eleTy = cast<ComplexType>(ty).getElementType();
    return 2 * cudaq::opt::convertBitsToBytes(eleTy.getIntOrFloatBitWidth());
  }
  return cudaq::opt::convertBitsToBytes(ty.getIntOrFloatBitWidth());
}

template <typename T>
py::object readPyObject(mlir::Type ty, char *arg) {
  unsigned int bytes = byteSize(ty);
  if (sizeof(T) != bytes) {
    ty.dump();
    throw std::runtime_error(
        "Error reading return value of type (reading bytes: " +
        std::to_string(sizeof(T)) +
        ", bytes available to read: " + std::to_string(bytes) + ")");
  }
  T concrete;
  std::memcpy(&concrete, arg, bytes);
  return py_ext::convert<T>(concrete);
}

std::vector<py::object> readRunResults(mlir::Type ty,
                                       details::RunResultSpan &results) {
  std::vector<py::object> ret;
  for (std::size_t i = 0; i < results.lengthInBytes; i += byteSize(ty)) {
    py::object obj =
        llvm::TypeSwitch<mlir::Type, py::object>(ty)
            .Case([&](IntegerType ty) -> py::object {
              if (ty.getIntOrFloatBitWidth() == 1)
                return readPyObject<bool>(ty, results.data + i);
              if (ty.getIntOrFloatBitWidth() == 32)
                return readPyObject<std::int32_t>(ty, results.data + i);
              return readPyObject<std::int64_t>(ty, results.data + i);
            })
            .Case([&](ComplexType ty) -> py::object {
              auto eleTy = ty.getElementType();
              return llvm::TypeSwitch<mlir::Type, py::object>(eleTy)
                  .Case([&](Float64Type eTy) -> py::object {
                    return readPyObject<std::complex<double>>(ty,
                                                              results.data + i);
                  })
                  .Case([&](Float32Type eTy) -> py::object {
                    return readPyObject<std::complex<float>>(ty,
                                                             results.data + i);
                  })
                  .Default([](Type eTy) -> py::object {
                    eTy.dump();
                    throw std::runtime_error(
                        "Invalid float element type for return "
                        "complex type for cudaq.run.");
                  });
            })
            .Case([&](Float64Type ty) -> py::object {
              return readPyObject<double>(ty, results.data + i);
            })
            .Case([&](Float32Type ty) -> py::object {
              return readPyObject<float>(ty, results.data + i);
            })
            .Default([](Type ty) -> py::object {
              ty.dump();
              throw std::runtime_error("Invalid return type for cudaq.run.");
            });
    ret.push_back(obj);
  }
  return ret;
}

std::tuple<std::string, MlirModule, OpaqueArguments *,
           mlir::ArrayRef<mlir::Type>>
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

  auto returnTypes = getKernelFuncOp(kernelMod, kernelName).getResultTypes();
  return {kernelName, kernelMod, argData, returnTypes};
}
} // namespace

/// @brief Run `cudaq::run` on the provided kernel.
std::vector<py::object> pyRun(py::object &kernel, py::args args,
                              py::int_ shots_count,
                              std::optional<noise_model> noise_model) {
  auto [kernelName, kernelMod, argData, returnTypes] =
      getKernelLaunchParameters(kernel, args);

  if (returnTypes.empty() || returnTypes.size() > 1)
    throw std::runtime_error(
        "cudaq.run only supports kernels that return a value.");

  if (shots_count < 0)
    throw std::runtime_error("Invalid shots_count. Must be non-negative.");

  auto returnTy = returnTypes[0];
  auto mod = unwrap(kernelMod);
  mod->setAttr(cudaq::runtime::enableCudaqRun,
               mlir::UnitAttr::get(mod->getContext()));

  auto &platform = cudaq::get_platform();
  if (noise_model.has_value()) {
    if (platform.is_remote())
      throw std::runtime_error(
          "Noise model is not supported on remote platforms.");
  }

  if (shots_count == 0)
    return {};

  // Launch the kernel in the appropriate context.
  if (noise_model.has_value())
    platform.set_noise(&noise_model.value());

  details::RunResultSpan results = details::runTheKernel(
      [&]() mutable { pyAltLaunchKernel(kernelName, kernelMod, *argData, {}); },
      platform, kernelName, shots_count);
  delete argData;

  if (noise_model.has_value())
    platform.reset_noise();

  mod->removeAttr(cudaq::runtime::enableCudaqRun);
  return readRunResults(returnTy, results);
}

/// @brief Bind the run cudaq function
void bindPyRun(py::module &mod) {
  mod.def("run", &pyRun, py::arg("kernel"), py::kw_only(),
          py::arg("shots_count") = 1000, py::arg("noise_model") = py::none(),
          R"#()#");
}

} // namespace cudaq
