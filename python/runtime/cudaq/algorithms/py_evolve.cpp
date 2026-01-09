/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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
#include "runtime/cudaq/platform/py_alt_launch_kernel.h"
#include "utils/OpaqueArguments.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace cudaq {

template <typename numeric_type>
using spin_op_creator =
    std::function<spin_op(std::map<std::string, numeric_type>)>;

// Helper to determine if an object is a Python kernel builder object (PyKernel)
static bool isPyKernelObject(py::object &kernel) {
  const std::string kernelTypeName =
      py::hasattr(kernel, "__class__")
          ? kernel.attr("__class__").attr("__name__").cast<std::string>()
          : "";
  return (kernelTypeName == "PyKernel");
}

template <typename numeric_type>
evolve_result
pyEvolve(state initial_state, py::object kernel,
         std::map<std::string, numeric_type> params,
         std::vector<spin_op_creator<numeric_type>> observables = {},
         int shots_count = -1) {
  if (!isPyKernelObject(kernel))
    throw std::runtime_error(
        "The provided kernel to pyEvolve is not a valid PyKernel object.");

  if (py::hasattr(kernel, "compile"))
    kernel.attr("compile")();

  auto kernelName = kernel.attr("name").cast<std::string>();
  auto kernelMod = unwrap(kernel.attr("module").cast<MlirModule>());

  std::vector<spin_op> spin_ops = {};
  for (auto &observable : observables) {
    spin_ops.push_back(observable(params));
  }

  auto res = __internal__::evolve(
      initial_state,
      [kernelMod, kernelName](state state) mutable {
        auto *argData = new cudaq::OpaqueArguments();
        valueArgument(*argData, &state);
        auto *ctx = kernelMod->getContext();
        auto retTy = mlir::NoneType::get(ctx);
        [[maybe_unused]] auto result =
            clean_launch_module(kernelName, kernelMod, retTy, *argData);
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
         int shots_count = -1, bool save_intermediate_states = true) {
  if (!std::all_of(kernels.begin(), kernels.end(),
                   [](py::object &kernel) { return isPyKernelObject(kernel); }))
    throw std::runtime_error(
        "One or more of the provided kernels to pyEvolve is not a valid "
        "PyKernel object.");

  std::vector<std::function<void(state)>> launchFcts = {};
  for (py::object kernel : kernels) {
    if (py::hasattr(kernel, "compile"))
      kernel.attr("compile")();

    auto kernelName = kernel.attr("name").cast<std::string>();
    auto kernelMod = unwrap(kernel.attr("module").cast<MlirModule>());

    launchFcts.push_back([kernelMod, kernelName](state state) mutable {
      auto *argData = new cudaq::OpaqueArguments();
      valueArgument(*argData, &state);
      auto *ctx = kernelMod->getContext();
      auto retTy = mlir::NoneType::get(ctx);
      [[maybe_unused]] auto result =
          clean_launch_module(kernelName, kernelMod, retTy, *argData);
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

  return __internal__::evolve(initial_state, launchFcts, spin_ops, shots_count,
                              save_intermediate_states);
}

template <typename numeric_type>
async_evolve_result
pyEvolveAsync(state initial_state, py::object kernel,
              std::map<std::string, numeric_type> params,
              std::vector<spin_op_creator<numeric_type>> observables = {},
              std::size_t qpu_id = 0,
              std::optional<cudaq::noise_model> noise_model = std::nullopt,
              int shots_count = -1) {
  if (!isPyKernelObject(kernel))
    throw std::runtime_error(
        "The provided kernel to pyEvolveAsync is not a valid PyKernel object.");

  if (py::hasattr(kernel, "compile"))
    kernel.attr("compile")();

  auto kernelMod = unwrap(kernel.attr("module").cast<MlirModule>()).clone();
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
        auto *ctx = kernelMod->getContext();
        auto retTy = mlir::NoneType::get(ctx);
        [[maybe_unused]] auto result =
            clean_launch_module(kernelName, kernelMod, retTy, *argData);
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
              int shots_count = -1, bool save_intermediate_states = true) {
  if (!std::all_of(kernels.begin(), kernels.end(),
                   [](py::object &kernel) { return isPyKernelObject(kernel); }))
    throw std::runtime_error(
        "One or more of the provided kernels to pyEvolveAsync is not a valid "
        "PyKernel object.");

  std::vector<std::function<void(state)>> launchFcts = {};
  for (py::object kernel : kernels) {
    if (py::hasattr(kernel, "compile"))
      kernel.attr("compile")();

    // IMPORTANT: we need to make sure no Python data is accessed in the async.
    // functor.
    auto kernelMod = unwrap(kernel.attr("module").cast<MlirModule>()).clone();
    auto kernelName = kernel.attr("name").cast<std::string>();
    launchFcts.push_back(
        [kernelMod = std::move(kernelMod), kernelName](state state) mutable {
          cudaq::OpaqueArguments argData;
          valueArgument(argData, &state);
          auto *ctx = kernelMod->getContext();
          auto retTy = mlir::NoneType::get(ctx);
          [[maybe_unused]] auto result =
              clean_launch_module(kernelName, kernelMod, retTy, argData);
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
                                    noise_model, shots_count,
                                    save_intermediate_states);
}

#define DEFINE_PARAM_TYPE_OVERLOAD_VEC(type, pyMod)                            \
  pyMod.def(                                                                   \
      "evolve",                                                                \
      [](state initial_state, std::vector<py::object> kernels,                 \
         std::vector<std::map<std::string, type>> params = {},                 \
         std::vector<spin_op_creator<type>> observables = {},                  \
         int shots_count = -1, bool save_intermediate_states = true) {         \
        return pyEvolve(initial_state, kernels, params, observables,           \
                        shots_count, save_intermediate_states);                \
      },                                                                       \
      "Evolve the given initial_state with the provided kernel and "           \
      "parameters.",                                                           \
      py::arg("initial_state"), py::arg("kernels"),                            \
      py::arg("params") = std::vector<std::map<std::string, type>>{},          \
      py::arg("observables") = std::vector<spin_op_creator<type>>{},           \
      py::arg("shots_count") = -1,                                             \
      py::arg("save_intermediate_states") = true);

#define DEFINE_PARAM_TYPE_OVERLOAD(type, pyMod)                                \
  pyMod.def(                                                                   \
      "evolve",                                                                \
      [](state initial_state, py::object kernel,                               \
         std::map<std::string, type> params = {},                              \
         std::vector<spin_op_creator<type>> observables = {},                  \
         int shots_count = -1) {                                               \
        return pyEvolve(initial_state, kernel, params, observables,            \
                        shots_count);                                          \
      },                                                                       \
      "Evolve the given initial_state with the provided kernel and "           \
      "parameters.",                                                           \
      py::arg("initial_state"), py::arg("kernels"),                            \
      py::arg("params") = std::map<std::string, type>{},                       \
      py::arg("observables") = std::vector<spin_op_creator<type>>{},           \
      py::arg("shots_count") = -1);

#define DEFINE_ASYNC_PARAM_TYPE_OVERLOAD_VEC(type, pyMod)                      \
  pyMod.def(                                                                   \
      "evolve_async",                                                          \
      [](state initial_state, std::vector<py::object> kernels,                 \
         std::vector<std::map<std::string, type>> params = {},                 \
         std::vector<spin_op_creator<type>> observables = {},                  \
         std::size_t qpu_id = 0,                                               \
         std::optional<cudaq::noise_model> noise_model = std::nullopt,         \
         int shots_count = -1, bool save_intermediate_states = true) {         \
        return pyEvolveAsync(initial_state, kernels, params, observables,      \
                             qpu_id, noise_model, shots_count,                 \
                             save_intermediate_states);                        \
      },                                                                       \
      "Asynchronously evolve the given initial_state with "                    \
      "the provided kernel and parameters.",                                   \
      py::arg("initial_state"), py::arg("kernels"),                            \
      py::arg("params") = std::vector<std::map<std::string, type>>{},          \
      py::arg("observables") = std::vector<spin_op_creator<type>>{},           \
      py::arg("qpu_id") = 0, py::arg("noise_model") = std::nullopt,            \
      py::arg("shots_count") = -1,                                             \
      py::arg("save_intermediate_states") = true);

#define DEFINE_ASYNC_PARAM_TYPE_OVERLOAD(type, pyMod)                          \
  pyMod.def(                                                                   \
      "evolve_async",                                                          \
      [](state initial_state, py::object kernel,                               \
         std::map<std::string, type> params = {},                              \
         std::vector<spin_op_creator<type>> observables = {},                  \
         std::size_t qpu_id = 0,                                               \
         std::optional<cudaq::noise_model> noise_model = std::nullopt,         \
         int shots_count = -1) {                                               \
        return pyEvolveAsync(initial_state, kernel, params, observables,       \
                             qpu_id, noise_model, shots_count);                \
      },                                                                       \
      "Asynchronously evolve the given initial_state with "                    \
      "the provided kernel and parameters.",                                   \
      py::arg("initial_state"), py::arg("kernels"),                            \
      py::arg("params") = std::map<std::string, type>{},                       \
      py::arg("observables") = std::vector<spin_op_creator<type>>{},           \
      py::arg("qpu_id") = 0, py::arg("noise_model") = std::nullopt,            \
      py::arg("shots_count") = -1);

/// @brief Bind the evolve cudaq function for circuit simulator
void bindPyEvolve(py::module &mod) {
  // Sync evolve overloads
  DEFINE_PARAM_TYPE_OVERLOAD_VEC(long, mod);
  DEFINE_PARAM_TYPE_OVERLOAD_VEC(double, mod);
  DEFINE_PARAM_TYPE_OVERLOAD_VEC(std::complex<double>, mod);
  DEFINE_PARAM_TYPE_OVERLOAD(long, mod);
  DEFINE_PARAM_TYPE_OVERLOAD(double, mod);
  DEFINE_PARAM_TYPE_OVERLOAD(std::complex<double>, mod);

  // Async evolve overloads
  DEFINE_ASYNC_PARAM_TYPE_OVERLOAD_VEC(long, mod);
  DEFINE_ASYNC_PARAM_TYPE_OVERLOAD_VEC(double, mod);
  DEFINE_ASYNC_PARAM_TYPE_OVERLOAD_VEC(std::complex<double>, mod);
  DEFINE_ASYNC_PARAM_TYPE_OVERLOAD(long, mod);
  DEFINE_ASYNC_PARAM_TYPE_OVERLOAD(double, mod);
  DEFINE_ASYNC_PARAM_TYPE_OVERLOAD(std::complex<double>, mod);
}

} // namespace cudaq
