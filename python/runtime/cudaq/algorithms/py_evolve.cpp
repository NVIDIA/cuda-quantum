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
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>

namespace cudaq {

void pyAltLaunchKernel(const std::string &, MlirModule, OpaqueArguments &,
                       const std::vector<std::string> &);

template <typename numeric_type>
using spin_op_creator = std::function<spin_op(std::map<std::string, numeric_type>)>;

template <typename numeric_type>
evolve_result pyEvolve(state initial_state, 
                       py::object kernel, 
                       std::map<std::string, numeric_type> params,
                       std::vector<spin_op_creator<numeric_type>> observables = {}) {
  if (py::hasattr(kernel, "compile"))
    kernel.attr("compile")();

  auto kernelName = kernel.attr("name").cast<std::string>();
  auto kernelMod = kernel.attr("module").cast<MlirModule>();

  // FIXME: update evolve.h to eliminate this
  std::vector<std::function<spin_op()>> spin_ops = {};
  for (auto observable : observables) {
    spin_ops.push_back([observable, params]() { return observable(params); });
  }

  auto res = evolve(initial_state, [kernelMod, kernelName](state state) mutable {
    auto *argData = new cudaq::OpaqueArguments();
    valueArgument(*argData, &state);
    pyAltLaunchKernel(kernelName, kernelMod, *argData, {});
    delete argData;
  }, spin_ops);
  return res;
}

template <typename numeric_type>
evolve_result pyEvolve(state initial_state, 
                       std::vector<py::object> kernels, 
                       std::vector<std::map<std::string, numeric_type>> params,
                       std::vector<spin_op_creator<numeric_type>> observables = {}) {
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

  // FIXME: update evolve.h to eliminate this
  std::vector<std::vector<std::function<spin_op()>>> spin_ops = {};
  for (auto parameters : params) {
    std::vector<std::function<spin_op()>> ops = {};
    for (auto observable : observables) {
      ops.push_back([observable, parameters]() { return observable(parameters); });
    }
    spin_ops.push_back(ops);
  }

  return evolve(initial_state, launchFcts, spin_ops);
}

template <typename numeric_type>
async_evolve_result pyEvolveAsync(state initial_state, 
                                  py::object kernel, 
                                  std::map<std::string, numeric_type> params,
                                  std::vector<spin_op_creator<numeric_type>> observables = {}, 
                                  std::size_t qpu_id = 0) {
  if (py::hasattr(kernel, "compile"))
    kernel.attr("compile")();

  auto kernelName = kernel.attr("name").cast<std::string>();
  auto kernelMod = kernel.attr("module").cast<MlirModule>();

  // FIXME: update evolve.h to eliminate this
  std::vector<std::function<spin_op()>> spin_ops = {};
  for (auto observable : observables) {
    spin_ops.push_back([observable, params]() { return observable(params); });
  }

  py::gil_scoped_release release;
  return evolve_async(initial_state,
    [kernelMod, kernelName](state state) mutable {
      auto *argData = new cudaq::OpaqueArguments();
      valueArgument(*argData, &state);
      pyAltLaunchKernel(kernelName, kernelMod, *argData, {});
      delete argData;
    }, spin_ops, qpu_id);
}

template <typename numeric_type>
async_evolve_result pyEvolveAsync(state initial_state, 
                                  std::vector<py::object> kernels, 
                                  std::vector<std::map<std::string, numeric_type>> params,
                                  std::vector<spin_op_creator<numeric_type>> observables = {}, 
                                  std::size_t qpu_id = 0) {
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

  // FIXME: update evolve.h to eliminate this
  std::vector<std::vector<std::function<spin_op()>>> spin_ops = {};
  for (auto parameters : params) {
    std::vector<std::function<spin_op()>> ops = {};
    for (auto observable : observables) {
      ops.push_back([observable, parameters]() { return observable(parameters); });
    }
    spin_ops.push_back(ops);
  }

  py::gil_scoped_release release;
  return evolve_async(initial_state, launchFcts, spin_ops, qpu_id);
}


/// @brief Bind the get_state cudaq function
void bindPyEvolve(py::module &mod) {

  // Note: vector versions need to be first, otherwise the incorrect 
  // overload is used.
  mod.def(
      "evolve",
      [](state initial_state, std::vector<py::object> kernels) {
        return pyEvolve<long>(initial_state, kernels, {});
      },
      "");
  mod.def(
      "evolve",
      [](state initial_state, std::vector<py::object> kernels, std::vector<std::map<std::string, long>> params, std::vector<spin_op_creator<long>> observables) {
        return pyEvolve(initial_state, kernels, params, observables);
      },
      "");
  mod.def(
      "evolve",
      [](state initial_state, std::vector<py::object> kernels, std::vector<std::map<std::string, double>> params, std::vector<spin_op_creator<double>> observables) {
        return pyEvolve(initial_state, kernels, params, observables);
      },
      "");
  mod.def(
      "evolve",
      [](state initial_state, std::vector<py::object> kernels, std::vector<std::map<std::string, std::complex<double>>> params, std::vector<spin_op_creator<std::complex<double>>> observables) {
        return pyEvolve(initial_state, kernels, params, observables);
      },
      "");
  mod.def(
      "evolve",
      [](state initial_state, py::object kernel) {
        return pyEvolve(initial_state, kernel, std::map<std::string, long>{});
      },
      "");
  mod.def(
      "evolve",
      [](state initial_state, py::object kernel, std::map<std::string, long> params, std::vector<spin_op_creator<long>> observables) {
        return pyEvolve(initial_state, kernel, params, observables);
      },
      "");
  mod.def(
      "evolve",
      [](state initial_state, py::object kernel, std::map<std::string, double> params, std::vector<spin_op_creator<double>> observables) {
        return pyEvolve(initial_state, kernel, params, observables);
      },
      "");
  mod.def(
      "evolve",
      [](state initial_state, py::object kernel, std::map<std::string, std::complex<double>> params, std::vector<spin_op_creator<std::complex<double>>> observables) {
        return pyEvolve(initial_state, kernel, params, observables);
      },
      "");

  // Note: vector versions need to be first, otherwise the incorrect 
  // overload is used.
  mod.def(
      "evolve_async",
      [](state initial_state, std::vector<py::object> kernels, std::size_t qpu_id) {
        return pyEvolveAsync<long>(initial_state, kernels, {}, {}, qpu_id);
      },
      py::arg("initial_state"), py::arg("kernels"), py::arg("qpu_id") = 0,
      "");
  mod.def(
      "evolve_async",
      [](state initial_state, std::vector<py::object> kernels, std::vector<std::map<std::string, long>> params, std::vector<spin_op_creator<long>> observables, std::size_t qpu_id) {
        return pyEvolveAsync(initial_state, kernels, params, observables, qpu_id);
      },
      py::arg("initial_state"), py::arg("kernels"), py::arg("params"), py::arg("observables"), py::arg("qpu_id") = 0,
      "");
  mod.def(
      "evolve_async",
      [](state initial_state, std::vector<py::object> kernels, std::vector<std::map<std::string, double>> params, std::vector<spin_op_creator<double>> observables, std::size_t qpu_id) {
        return pyEvolveAsync(initial_state, kernels, params, observables, qpu_id);
      },
      py::arg("initial_state"), py::arg("kernels"), py::arg("params"), py::arg("observables"), py::arg("qpu_id") = 0,
      "");
  mod.def(
      "evolve_async",
      [](state initial_state, std::vector<py::object> kernels, std::vector<std::map<std::string, std::complex<double>>> params, std::vector<spin_op_creator<std::complex<double>>> observables, std::size_t qpu_id) {
        return pyEvolveAsync(initial_state, kernels, params, observables, qpu_id);
      },
      py::arg("initial_state"), py::arg("kernels"), py::arg("params"), py::arg("observables"), py::arg("qpu_id") = 0,
      "");
  mod.def(
      "evolve_async",
      [](state initial_state, py::object kernel, std::size_t qpu_id) {
        return pyEvolveAsync(initial_state, kernel, std::map<std::string, long>{}, {}, qpu_id);
      },
      py::arg("initial_state"), py::arg("kernel"), py::arg("qpu_id") = 0,
      "");
  mod.def(
      "evolve_async",
      [](state initial_state, py::object kernel, std::map<std::string, long> params, std::vector<spin_op_creator<long>> observables, std::size_t qpu_id) {
        return pyEvolveAsync(initial_state, kernel, params, observables, qpu_id);
      },
      py::arg("initial_state"), py::arg("kernel"), py::arg("params"), py::arg("observables"), py::arg("qpu_id") = 0,
      "");
  mod.def(
      "evolve_async",
      [](state initial_state, py::object kernel, std::map<std::string, double> params, std::vector<spin_op_creator<double>> observables, std::size_t qpu_id) {
        return pyEvolveAsync(initial_state, kernel, params, observables, qpu_id);
      },
      py::arg("initial_state"), py::arg("kernel"), py::arg("params"), py::arg("observables"), py::arg("qpu_id") = 0,
      "");
  mod.def(
      "evolve_async",
      [](state initial_state, py::object kernel, std::map<std::string, std::complex<double>> params, std::vector<spin_op_creator<std::complex<double>>> observables, std::size_t qpu_id) {
        return pyEvolveAsync(initial_state, kernel, params, observables, qpu_id);
      },
      py::arg("initial_state"), py::arg("kernel"), py::arg("params"), py::arg("observables"), py::arg("qpu_id") = 0,
      "");

}

} // namespace cudaq
