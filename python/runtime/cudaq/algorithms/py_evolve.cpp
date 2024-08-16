/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "py_evolve.h"
#include "LinkedLibraryHolder.h"
#include "common/ArgumentWrapper.h"
#include "common/Logger.h"
#include "cudaq/algorithms/evolve.h"
#include "utils/OpaqueArguments.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include <pybind11/complex.h>
#include <pybind11/stl.h>

namespace cudaq {

void pyAltLaunchKernel(const std::string &, MlirModule, OpaqueArguments &,
                       const std::vector<std::string> &);

evolve_result pyEvolve(py::object kernel, std::vector<spin_op> observables = {}) {
  if (py::hasattr(kernel, "compile"))
    kernel.attr("compile")();

  auto kernelName = kernel.attr("name").cast<std::string>();
  auto kernelMod = kernel.attr("module").cast<MlirModule>();
  auto *argData = new cudaq::OpaqueArguments();

  return evolve([&]() mutable {
    pyAltLaunchKernel(kernelName, kernelMod, *argData, {});
    delete argData;
  }, observables);
}

async_evolve_result pyEvolveAsync(py::object kernel, std::vector<spin_op> observables = {}, std::size_t qpu_id = 0) {
  if (py::hasattr(kernel, "compile"))
    kernel.attr("compile")();

  auto kernelName = kernel.attr("name").cast<std::string>();
  auto kernelMod = kernel.attr("module").cast<MlirModule>();
  auto *argData = new cudaq::OpaqueArguments();

  py::gil_scoped_release release;
  return evolve_async(
      [kernelMod, argData, kernelName]() mutable {
        pyAltLaunchKernel(kernelName, kernelMod, *argData, {});
        delete argData;
      }, observables, qpu_id);
}

/// @brief Bind the get_state cudaq function
void bindPyEvolve(py::module &mod) {

  mod.def(
      "evolve",
      [](py::object kernel) {
        return pyEvolve(kernel);
      },
      "");
  mod.def(
      "evolve",
      [](py::object kernel, std::vector<spin_op> observables) {
        return pyEvolve(kernel, observables);
      },
      "");

  mod.def(
      "evolve_async",
      [](py::object kernel, std::size_t qpu_id) {
        return pyEvolveAsync(kernel, {}, qpu_id);
      },
      py::arg("kernel"), py::arg("qpu_id") = 0,
      "");
  mod.def(
      "evolve_async",
      [](py::object kernel, std::vector<spin_op> observables, std::size_t qpu_id) {
        return pyEvolveAsync(kernel, observables, qpu_id);
      },
      py::arg("kernel"), py::arg("observables"), py::arg("qpu_id") = 0,
      "");
}

} // namespace cudaq
