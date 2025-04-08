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

std::vector<py::object> readRunResults(mlir::Type ty, details::RunResultSpan &results) {
    std::vector<py::object> ret;
    for (std::size_t i = 0; i < results.lengthInBytes; i += byteSize(ty)) {
        auto obj = readPyObject<std::int64_t>(ty, results.data + i);
        ret.push_back(obj);
    }
    return ret;
}


std::tuple<std::string, MlirModule, OpaqueArguments *, mlir::Type>
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

  auto returnTy = getKernelFuncOp(kernelMod, kernelName).getResultTypes()[0];
  return {kernelName, kernelMod, argData, returnTy};
}
} // namespace

/// @brief Run `cudaq::run` on the provided kernel.
std::vector<py::object> pyRun(py::object &kernel, py::args args, py::int_ shots_count) {
  auto [kernelName, kernelMod, argData, returnTy] =
      getKernelLaunchParameters(kernel, args);

  py::print("RUN");
  returnTy.dump();

  auto &platform = cudaq::get_platform();
  details::RunResultSpan results = details::runTheKernel(
    [&]() mutable {
      pyAltLaunchKernel(kernelName, kernelMod, *argData, {});
    },
    platform, kernelName, shots_count);

  return readRunResults(returnTy, results);
}

/// @brief Bind the run cudaq function
void bindPyRun(py::module &mod) {
  // mod.def("run", py::overload_cast<py::object &, py::args>(&pyRun),
  //         R"#()#")
    mod.def("run", &pyRun, py::arg("kernel"), py::kw_only(),
          py::arg("shots_count") = 1000,
          R"#()#");
}

} // namespace cudaq
