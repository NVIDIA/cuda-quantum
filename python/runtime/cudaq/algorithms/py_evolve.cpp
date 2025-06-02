/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "py_evolve.h"
#include "LinkedLibraryHolder.h"
#include "common/ArgumentWrapper.h"
#include "common/Logger.h"
#include "cudaq/algorithms/evolve_internal.h"
#include "utils/OpaqueArguments.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace cudaq {

void pyAltLaunchKernel(const std::string &, MlirModule, OpaqueArguments &,
                       const std::vector<std::string> &);

template <typename numeric_type>
using spin_op_creator =
    std::function<spin_op(std::map<std::string, numeric_type>)>;

template <typename numeric_type>
evolve_result
pyEvolve(state initial_state, py::object kernel,
         std::map<std::string, numeric_type> params,
         std::vector<spin_op_creator<numeric_type>> observables = {},
         int shots_count = -1) {
  if (py::hasattr(kernel, "compile"))
    kernel.attr("compile")();

  auto kernelName = kernel.attr("name").cast<std::string>();
  auto kernelMod = kernel.attr("module").cast<MlirModule>();

  std::vector<spin_op> spin_ops = {};
  for (auto &observable : observables) {
    spin_ops.push_back(observable(params));
  }

  auto res = __internal__::evolve(
      initial_state,
      [kernelMod, kernelName](state state) mutable {
        auto *argData = new cudaq::OpaqueArguments();
        valueArgument(*argData, &state);
        pyAltLaunchKernel(kernelName, kernelMod, *argData, {});
        delete argData;
      },
      spin_ops, shots_count);
  return res;
}

template <typename numeric_type>
evolve_result
pyEvolve(state initial_state, std::vector<py::object> kernels,
         std::vector<std::map<std::string, numeric_type>> params,
         std::vector<spin_op_creator<numeric_type>> observables = {},
         int shots_count = -1) {
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

  std::vector<std::vector<spin_op>> spin_ops = {};
  for (auto parameters : params) {
    std::vector<spin_op> ops = {};
    for (auto &observable : observables) {
      ops.push_back(observable(parameters));
    }
    spin_ops.push_back(std::move(ops));
  }

  return __internal__::evolve(initial_state, launchFcts, spin_ops, shots_count);
}

template <typename numeric_type>
async_evolve_result
pyEvolveAsync(state initial_state, py::object kernel,
              std::map<std::string, numeric_type> params,
              std::vector<spin_op_creator<numeric_type>> observables = {},
              std::size_t qpu_id = 0,
              std::optional<cudaq::noise_model> noise_model = std::nullopt,
              int shots_count = -1) {
  if (py::hasattr(kernel, "compile"))
    kernel.attr("compile")();

  auto kernelMod =
      wrap(unwrap(kernel.attr("module").cast<MlirModule>()).clone());
  auto kernelName = kernel.attr("name").cast<std::string>();

  std::vector<spin_op> spin_ops = {};
  for (auto observable : observables) {
    spin_ops.push_back(observable(params));
  }

  py::gil_scoped_release release;
  return __internal__::evolve_async(
      initial_state,
      [kernelMod, kernelName](state state) mutable {
        auto *argData = new cudaq::OpaqueArguments();
        valueArgument(*argData, &state);
        pyAltLaunchKernel(kernelName, kernelMod, *argData, {});
        delete argData;
      },
      spin_ops, qpu_id, noise_model, shots_count);
}

template <typename numeric_type>
async_evolve_result
pyEvolveAsync(state initial_state, std::vector<py::object> kernels,
              std::vector<std::map<std::string, numeric_type>> params,
              std::vector<spin_op_creator<numeric_type>> observables = {},
              std::size_t qpu_id = 0,
              std::optional<cudaq::noise_model> noise_model = std::nullopt,
              int shots_count = -1) {
  std::vector<std::function<void(state)>> launchFcts = {};
  for (py::object kernel : kernels) {
    if (py::hasattr(kernel, "compile"))
      kernel.attr("compile")();

    // IMPORTANT: we need to make sure no Python data is accessed in the async.
    // functor.
    auto kernelMod =
        wrap(unwrap(kernel.attr("module").cast<MlirModule>()).clone());
    auto kernelName = kernel.attr("name").cast<std::string>();
    launchFcts.push_back(
        [kernelMod = std::move(kernelMod), kernelName](state state) mutable {
          cudaq::OpaqueArguments argData;
          valueArgument(argData, &state);
          pyAltLaunchKernel(kernelName, kernelMod, argData, {});
        });
  }

  std::vector<std::vector<spin_op>> spin_ops = {};
  for (auto parameters : params) {
    std::vector<spin_op> ops = {};
    for (auto observable : observables) {
      ops.push_back(observable(parameters));
    }
    spin_ops.push_back(std::move(ops));
  }

  py::gil_scoped_release release;
  return __internal__::evolve_async(initial_state, launchFcts, spin_ops, qpu_id,
                                    noise_model, shots_count);
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
      [](state initial_state, std::vector<py::object> kernels,
         std::vector<std::map<std::string, long>> params,
         std::vector<spin_op_creator<long>> observables, int shots_count = -1) {
        return pyEvolve(initial_state, kernels, params, observables,
                        shots_count);
      },
      "");
  mod.def(
      "evolve",
      [](state initial_state, std::vector<py::object> kernels,
         std::vector<std::map<std::string, double>> params,
         std::vector<spin_op_creator<double>> observables,
         int shots_count = -1) {
        return pyEvolve(initial_state, kernels, params, observables,
                        shots_count);
      },
      "");
  mod.def(
      "evolve",
      [](state initial_state, std::vector<py::object> kernels,
         std::vector<std::map<std::string, std::complex<double>>> params,
         std::vector<spin_op_creator<std::complex<double>>> observables,
         int shots_count = -1) {
        return pyEvolve(initial_state, kernels, params, observables,
                        shots_count);
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
      [](state initial_state, py::object kernel,
         std::map<std::string, long> params,
         std::vector<spin_op_creator<long>> observables, int shots_count = -1) {
        return pyEvolve(initial_state, kernel, params, observables,
                        shots_count);
      },
      "");
  mod.def(
      "evolve",
      [](state initial_state, py::object kernel,
         std::map<std::string, double> params,
         std::vector<spin_op_creator<double>> observables,
         int shots_count = -1) {
        return pyEvolve(initial_state, kernel, params, observables,
                        shots_count);
      },
      "");
  mod.def(
      "evolve",
      [](state initial_state, py::object kernel,
         std::map<std::string, std::complex<double>> params,
         std::vector<spin_op_creator<std::complex<double>>> observables,
         int shots_count = -1) {
        return pyEvolve(initial_state, kernel, params, observables,
                        shots_count);
      },
      "");

  // Note: vector versions need to be first, otherwise the incorrect
  // overload is used.
  mod.def(
      "evolve_async",
      [](state initial_state, std::vector<py::object> kernels,
         std::size_t qpu_id,
         std::optional<cudaq::noise_model> noise_model = std::nullopt) {
        return pyEvolveAsync<long>(initial_state, kernels, {}, {}, qpu_id,
                                   noise_model);
      },
      py::arg("initial_state"), py::arg("kernels"), py::arg("qpu_id") = 0,
      py::arg("noise_model") = std::nullopt, "");
  mod.def(
      "evolve_async",
      [](state initial_state, std::vector<py::object> kernels,
         std::vector<std::map<std::string, long>> params,
         std::vector<spin_op_creator<long>> observables, std::size_t qpu_id,
         std::optional<cudaq::noise_model> noise_model = std::nullopt,
         int shots_count = -1) {
        return pyEvolveAsync(initial_state, kernels, params, observables,
                             qpu_id, noise_model, shots_count);
      },
      py::arg("initial_state"), py::arg("kernels"), py::arg("params"),
      py::arg("observables"), py::arg("qpu_id") = 0,
      py::arg("noise_model") = std::nullopt, py::arg("shots_count") = -1, "");
  mod.def(
      "evolve_async",
      [](state initial_state, std::vector<py::object> kernels,
         std::vector<std::map<std::string, double>> params,
         std::vector<spin_op_creator<double>> observables, std::size_t qpu_id,
         std::optional<cudaq::noise_model> noise_model = std::nullopt,
         int shots_count = -1) {
        return pyEvolveAsync(initial_state, kernels, params, observables,
                             qpu_id, noise_model, shots_count);
      },
      py::arg("initial_state"), py::arg("kernels"), py::arg("params"),
      py::arg("observables"), py::arg("qpu_id") = 0,
      py::arg("noise_model") = std::nullopt, py::arg("shots_count") = -1, "");
  mod.def(
      "evolve_async",
      [](state initial_state, std::vector<py::object> kernels,
         std::vector<std::map<std::string, std::complex<double>>> params,
         std::vector<spin_op_creator<std::complex<double>>> observables,
         std::size_t qpu_id,
         std::optional<cudaq::noise_model> noise_model = std::nullopt,
         int shots_count = -1) {
        return pyEvolveAsync(initial_state, kernels, params, observables,
                             qpu_id, noise_model, shots_count);
      },
      py::arg("initial_state"), py::arg("kernels"), py::arg("params"),
      py::arg("observables"), py::arg("qpu_id") = 0,
      py::arg("noise_model") = std::nullopt, py::arg("shots_count") = -1, "");
  mod.def(
      "evolve_async",
      [](state initial_state, py::object kernel, std::size_t qpu_id,
         std::optional<cudaq::noise_model> noise_model = std::nullopt) {
        return pyEvolveAsync(initial_state, kernel,
                             std::map<std::string, long>{}, {}, qpu_id,
                             noise_model);
      },
      py::arg("initial_state"), py::arg("kernel"), py::arg("qpu_id") = 0,
      py::arg("noise_model") = std::nullopt, "");
  mod.def(
      "evolve_async",
      [](state initial_state, py::object kernel,
         std::map<std::string, long> params,
         std::vector<spin_op_creator<long>> observables, std::size_t qpu_id,
         std::optional<cudaq::noise_model> noise_model = std::nullopt,
         int shots_count = -1) {
        return pyEvolveAsync(initial_state, kernel, params, observables, qpu_id,
                             noise_model, shots_count);
      },
      py::arg("initial_state"), py::arg("kernel"), py::arg("params"),
      py::arg("observables"), py::arg("qpu_id") = 0,
      py::arg("noise_model") = std::nullopt, py::arg("shots_count") = -1, "");
  mod.def(
      "evolve_async",
      [](state initial_state, py::object kernel,
         std::map<std::string, double> params,
         std::vector<spin_op_creator<double>> observables, std::size_t qpu_id,
         std::optional<cudaq::noise_model> noise_model = std::nullopt,
         int shots_count = -1) {
        return pyEvolveAsync(initial_state, kernel, params, observables, qpu_id,
                             noise_model, shots_count);
      },
      py::arg("initial_state"), py::arg("kernel"), py::arg("params"),
      py::arg("observables"), py::arg("qpu_id") = 0,
      py::arg("noise_model") = std::nullopt, py::arg("shots_count") = -1, "");
  mod.def(
      "evolve_async",
      [](state initial_state, py::object kernel,
         std::map<std::string, std::complex<double>> params,
         std::vector<spin_op_creator<std::complex<double>>> observables,
         std::size_t qpu_id,
         std::optional<cudaq::noise_model> noise_model = std::nullopt,
         int shots_count = -1) {
        return pyEvolveAsync(initial_state, kernel, params, observables, qpu_id,
                             noise_model, shots_count);
      },
      py::arg("initial_state"), py::arg("kernel"), py::arg("params"),
      py::arg("observables"), py::arg("qpu_id") = 0,
      py::arg("noise_model") = std::nullopt, py::arg("shots_count") = -1, "");
  mod.def(
      "evolve_async",
      [](std::function<evolve_result()> evolveFunctor, std::size_t qpu_id = 0) {
        py::gil_scoped_release release;
        return __internal__::evolve_async(evolveFunctor, qpu_id);
      },
      py::arg("evolve_function"), py::arg("qpu_id") = 0);
}

} // namespace cudaq
