/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/algorithms/draw.h"
#include "utils/OpaqueArguments.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

#include <iostream>
#include <pybind11/complex.h>
#include <pybind11/stl.h>

namespace cudaq {

void pyAltLaunchKernel(const std::string &, MlirModule, OpaqueArguments &,
                       const std::vector<std::string> &);

/// @brief Run `cudaq::get_state` on the provided kernel and spin operator.
std::string pyDraw(py::object &kernel, py::args args) {

  if (py::len(kernel.attr("arguments")) != args.size())
    throw std::runtime_error("Invalid number of arguments passed to draw.");

  if (py::hasattr(kernel, "compile"))
    kernel.attr("compile")();

  auto kernelName = kernel.attr("name").cast<std::string>();
  auto kernelMod = kernel.attr("module").cast<MlirModule>();
  args = simplifiedValidateInputArguments(args);
  auto *argData = toOpaqueArgs(args);

  return details::extractTrace([&]() mutable {
    pyAltLaunchKernel(kernelName, kernelMod, *argData, {});
    delete argData;
  });
}

/// @brief Bind the draw cudaq function
void bindPyDraw(py::module &mod) { mod.def("draw", &pyDraw, ""); }

} // namespace cudaq
