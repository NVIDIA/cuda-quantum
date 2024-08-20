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

evolve_result pyEvolve(state initial_state, py::object kernel, std::vector<spin_op> observables = {}) {
  if (py::hasattr(kernel, "compile"))
    kernel.attr("compile")();

  auto kernelName = kernel.attr("name").cast<std::string>();
  auto kernelMod = kernel.attr("module").cast<MlirModule>();

  auto res = evolve(initial_state, [kernelMod, kernelName](state state) mutable {
    auto *argData = new cudaq::OpaqueArguments();
    valueArgument(*argData, &state);
    pyAltLaunchKernel(kernelName, kernelMod, *argData, {});
    delete argData;
  }, observables);
  return res;
}

evolve_result pyEvolve(state initial_state, std::vector<py::object> kernels, std::vector<std::vector<spin_op>> observables = {}) {
  std::vector<std::function<void(state)>> launchFcts = {};
  for (py::object kernel : kernels) {
    if (py::hasattr(kernel, "compile"))
      kernel.attr("compile")();

    auto kernelName = kernel.attr("name").cast<std::string>();
    auto kernelMod = kernel.attr("module").cast<MlirModule>();

    launchFcts.push_back([kernelMod, kernelName](state state) mutable {
      auto *argData = new cudaq::OpaqueArguments();
      valueArgument(*argData, &state);
      pyAltLaunchKernel(kernelName, kernelMod, *argData, {});
      delete argData;
    });
  }

  return evolve(initial_state, launchFcts, observables);
}

async_evolve_result pyEvolveAsync(state initial_state, py::object kernel, std::vector<spin_op> observables = {}, std::size_t qpu_id = 0) {
  if (py::hasattr(kernel, "compile"))
    kernel.attr("compile")();

  auto kernelName = kernel.attr("name").cast<std::string>();
  auto kernelMod = kernel.attr("module").cast<MlirModule>();

  py::gil_scoped_release release;
  return evolve_async(initial_state,
    [kernelMod, kernelName](state state) mutable {
      auto *argData = new cudaq::OpaqueArguments();
      valueArgument(*argData, &state);
      pyAltLaunchKernel(kernelName, kernelMod, *argData, {});
      delete argData;
    }, observables, qpu_id);
}

async_evolve_result pyEvolveAsync(state initial_state, std::vector<py::object> kernels, std::vector<std::vector<spin_op>> observables = {}, std::size_t qpu_id = 0) {
  std::vector<std::function<void(state)>> launchFcts = {};
  for (py::object kernel : kernels) {
    if (py::hasattr(kernel, "compile"))
      kernel.attr("compile")();

    auto kernelName = kernel.attr("name").cast<std::string>();
    auto kernelMod = kernel.attr("module").cast<MlirModule>();

    launchFcts.push_back([kernelMod, kernelName](state state) mutable {
      auto *argData = new cudaq::OpaqueArguments();
      valueArgument(*argData, &state);
      pyAltLaunchKernel(kernelName, kernelMod, *argData, {});
      delete argData;
    });
  }

  py::gil_scoped_release release;
  return evolve_async(initial_state, launchFcts, observables, qpu_id);
}

/// @brief Bind the get_state cudaq function
void bindPyEvolve(py::module &mod) {

  // Note: vector versions need to be first, otherwise the incorrect 
  // overload is used.
  mod.def(
      "evolve",
      [](state initial_state, std::vector<py::object> kernels) {
        return pyEvolve(initial_state, kernels);
      },
      "");
  mod.def(
      "evolve",
      [](state initial_state, std::vector<py::object> kernels, std::vector<std::vector<spin_op>> observables) {
        return pyEvolve(initial_state, kernels, observables);
      },
      "");
  mod.def(
      "evolve",
      [](state initial_state, py::object kernel) {
        return pyEvolve(initial_state, kernel);
      },
      "");
  mod.def(
      "evolve",
      [](state initial_state, py::object kernel, std::vector<spin_op> observables) {
        return pyEvolve(initial_state, kernel, observables);
      },
      "");

  // Note: vector versions need to be first, otherwise the incorrect 
  // overload is used.
  mod.def(
      "evolve_async",
      [](state initial_state, std::vector<py::object> kernels, std::size_t qpu_id) {
        return pyEvolveAsync(initial_state, kernels, {}, qpu_id);
      },
      py::arg("initial_state"), py::arg("kernels"), py::arg("qpu_id") = 0,
      "");
  mod.def(
      "evolve_async",
      [](state initial_state, std::vector<py::object> kernels, std::vector<std::vector<spin_op>> observables, std::size_t qpu_id) {
        return pyEvolveAsync(initial_state, kernels, observables, qpu_id);
      },
      py::arg("initial_state"), py::arg("kernels"), py::arg("observables"), py::arg("qpu_id") = 0,
      "");
  mod.def(
      "evolve_async",
      [](state initial_state, py::object kernel, std::size_t qpu_id) {
        return pyEvolveAsync(initial_state, kernel, {}, qpu_id);
      },
      py::arg("initial_state"), py::arg("kernel"), py::arg("qpu_id") = 0,
      "");
  mod.def(
      "evolve_async",
      [](state initial_state, py::object kernel, std::vector<spin_op> observables, std::size_t qpu_id) {
        return pyEvolveAsync(initial_state, kernel, observables, qpu_id);
      },
      py::arg("initial_state"), py::arg("kernel"), py::arg("observables"), py::arg("qpu_id") = 0,
      "");
}

} // namespace cudaq
