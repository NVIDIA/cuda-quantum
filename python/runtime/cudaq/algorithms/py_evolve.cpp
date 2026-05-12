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
#include "cudaq/algorithms/evolve_internal.h"
#include "cudaq/runtime/logger/logger.h"
#include "runtime/cudaq/platform/py_alt_launch_kernel.h"
#include "utils/OpaqueArguments.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace cudaq {

template <typename numeric_type>
using spin_op_creator =
    std::function<spin_op(std::map<std::string, numeric_type>)>;

// Helper to determine if an object is a Python kernel builder object (PyKernel)
static bool isPyKernelObject(nanobind::object &kernel) {
  const std::string kernelTypeName =
      nanobind::hasattr(kernel, "__class__")
          ? nanobind::cast<std::string>(
                kernel.attr("__class__").attr("__name__"))
          : "";
  return (kernelTypeName == "PyKernel");
}

template <typename numeric_type>
evolve_result
pyEvolve(state initial_state, nanobind::object kernel,
         std::map<std::string, numeric_type> params,
         std::vector<spin_op_creator<numeric_type>> observables = {},
         int shots_count = -1) {
  if (!isPyKernelObject(kernel))
    throw std::runtime_error(
        "The provided kernel to pyEvolve is not a valid PyKernel object.");

  if (nanobind::hasattr(kernel, "compile"))
    kernel.attr("compile")();

  auto kernelName = nanobind::cast<std::string>(kernel.attr("name"));
  auto kernelMod = unwrap(nanobind::cast<MlirModule>(kernel.attr("module")));

  std::vector<spin_op> spin_ops = {};
  for (auto &observable : observables) {
    spin_ops.push_back(observable(params));
  }

  auto res = __internal__::evolve(
      initial_state,
      [kernelMod, kernelName](state state) mutable {
        auto *argData = new cudaq::OpaqueArguments();
        valueArgument(*argData, &state);
        [[maybe_unused]] auto result =
            clean_launch_module(kernelName, kernelMod, *argData);
        delete argData;
      },
      spin_ops, shots_count);
  return res;
}

template <typename numeric_type>
evolve_result
pyEvolve(state initial_state, std::vector<nanobind::object> kernels,
         std::vector<std::map<std::string, numeric_type>> params,
         std::vector<spin_op_creator<numeric_type>> observables = {},
         int shots_count = -1, bool save_intermediate_states = true) {
  if (!std::all_of(
          kernels.begin(), kernels.end(),
          [](nanobind::object &kernel) { return isPyKernelObject(kernel); }))
    throw std::runtime_error(
        "One or more of the provided kernels to pyEvolve is not a valid "
        "PyKernel object.");

  std::vector<std::function<void(state)>> launchFcts = {};
  for (nanobind::object kernel : kernels) {
    if (nanobind::hasattr(kernel, "compile"))
      kernel.attr("compile")();

    auto kernelName = nanobind::cast<std::string>(kernel.attr("name"));
    auto kernelMod = unwrap(nanobind::cast<MlirModule>(kernel.attr("module")));

    launchFcts.push_back([kernelMod, kernelName](state state) mutable {
      auto *argData = new cudaq::OpaqueArguments();
      valueArgument(*argData, &state);
      [[maybe_unused]] auto result =
          clean_launch_module(kernelName, kernelMod, *argData);
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
pyEvolveAsync(state initial_state, nanobind::object kernel,
              std::map<std::string, numeric_type> params,
              std::vector<spin_op_creator<numeric_type>> observables = {},
              std::size_t qpu_id = 0,
              std::optional<cudaq::noise_model> noise_model = std::nullopt,
              int shots_count = -1) {
  if (!isPyKernelObject(kernel))
    throw std::runtime_error(
        "The provided kernel to pyEvolveAsync is not a valid PyKernel object.");

  if (nanobind::hasattr(kernel, "compile"))
    kernel.attr("compile")();

  auto kernelMod =
      unwrap(nanobind::cast<MlirModule>(kernel.attr("module"))).clone();
  auto kernelName = nanobind::cast<std::string>(kernel.attr("name"));

  std::vector<spin_op> spin_ops = {};
  for (auto observable : observables) {
    spin_ops.push_back(observable(params));
  }

  nanobind::gil_scoped_release release;
  return __internal__::evolve_async(
      initial_state,
      [kernelMod, kernelName](state state) mutable {
        auto *argData = new cudaq::OpaqueArguments();
        valueArgument(*argData, &state);
        [[maybe_unused]] auto result =
            clean_launch_module(kernelName, kernelMod, *argData);
        delete argData;
      },
      spin_ops, qpu_id, noise_model, shots_count);
}

template <typename numeric_type>
async_evolve_result
pyEvolveAsync(state initial_state, std::vector<nanobind::object> kernels,
              std::vector<std::map<std::string, numeric_type>> params,
              std::vector<spin_op_creator<numeric_type>> observables = {},
              std::size_t qpu_id = 0,
              std::optional<cudaq::noise_model> noise_model = std::nullopt,
              int shots_count = -1, bool save_intermediate_states = true) {
  if (!std::all_of(
          kernels.begin(), kernels.end(),
          [](nanobind::object &kernel) { return isPyKernelObject(kernel); }))
    throw std::runtime_error(
        "One or more of the provided kernels to pyEvolveAsync is not a valid "
        "PyKernel object.");

  std::vector<std::function<void(state)>> launchFcts = {};
  for (nanobind::object kernel : kernels) {
    if (nanobind::hasattr(kernel, "compile"))
      kernel.attr("compile")();

    // IMPORTANT: we need to make sure no Python data is accessed in the async.
    // functor.
    auto kernelMod =
        unwrap(nanobind::cast<MlirModule>(kernel.attr("module"))).clone();
    auto kernelName = nanobind::cast<std::string>(kernel.attr("name"));
    launchFcts.push_back(
        [kernelMod = std::move(kernelMod), kernelName](state state) mutable {
          cudaq::OpaqueArguments argData;
          valueArgument(argData, &state);
          [[maybe_unused]] auto result =
              clean_launch_module(kernelName, kernelMod, argData);
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

  nanobind::gil_scoped_release release;
  return __internal__::evolve_async(initial_state, launchFcts, spin_ops, qpu_id,
                                    noise_model, shots_count,
                                    save_intermediate_states);
}

#define DEFINE_PARAM_TYPE_OVERLOAD_VEC(type, pyMod)                            \
  pyMod.def(                                                                   \
      "evolve",                                                                \
      [](state initial_state, std::vector<nanobind::object> kernels,           \
         std::vector<std::map<std::string, type>> params = {},                 \
         std::vector<spin_op_creator<type>> observables = {},                  \
         int shots_count = -1, bool save_intermediate_states = true) {         \
        return pyEvolve(initial_state, kernels, params, observables,           \
                        shots_count, save_intermediate_states);                \
      },                                                                       \
      "Evolve the given initial_state with the provided kernel and "           \
      "parameters.",                                                           \
      nanobind::arg("initial_state"), nanobind::arg("kernels"),                \
      nanobind::arg("params") = std::vector<std::map<std::string, type>>{},    \
      nanobind::arg("observables") = std::vector<spin_op_creator<type>>{},     \
      nanobind::arg("shots_count") = -1,                                       \
      nanobind::arg("save_intermediate_states") = true);

#define DEFINE_PARAM_TYPE_OVERLOAD(type, pyMod)                                \
  pyMod.def(                                                                   \
      "evolve",                                                                \
      [](state initial_state, nanobind::object kernel,                         \
         std::map<std::string, type> params = {},                              \
         std::vector<spin_op_creator<type>> observables = {},                  \
         int shots_count = -1) {                                               \
        return pyEvolve(initial_state, kernel, params, observables,            \
                        shots_count);                                          \
      },                                                                       \
      "Evolve the given initial_state with the provided kernel and "           \
      "parameters.",                                                           \
      nanobind::arg("initial_state"), nanobind::arg("kernels"),                \
      nanobind::arg("params") = std::map<std::string, type>{},                 \
      nanobind::arg("observables") = std::vector<spin_op_creator<type>>{},     \
      nanobind::arg("shots_count") = -1);

#define DEFINE_ASYNC_PARAM_TYPE_OVERLOAD_VEC(type, pyMod)                      \
  pyMod.def(                                                                   \
      "evolve_async",                                                          \
      [](state initial_state, std::vector<nanobind::object> kernels,           \
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
      nanobind::arg("initial_state"), nanobind::arg("kernels"),                \
      nanobind::arg("params") = std::vector<std::map<std::string, type>>{},    \
      nanobind::arg("observables") = std::vector<spin_op_creator<type>>{},     \
      nanobind::arg("qpu_id") = 0,                                             \
      nanobind::arg("noise_model") = std::nullopt,                             \
      nanobind::arg("shots_count") = -1,                                       \
      nanobind::arg("save_intermediate_states") = true);

#define DEFINE_ASYNC_PARAM_TYPE_OVERLOAD(type, pyMod)                          \
  pyMod.def(                                                                   \
      "evolve_async",                                                          \
      [](state initial_state, nanobind::object kernel,                         \
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
      nanobind::arg("initial_state"), nanobind::arg("kernels"),                \
      nanobind::arg("params") = std::map<std::string, type>{},                 \
      nanobind::arg("observables") = std::vector<spin_op_creator<type>>{},     \
      nanobind::arg("qpu_id") = 0,                                             \
      nanobind::arg("noise_model") = std::nullopt,                             \
      nanobind::arg("shots_count") = -1);

/// @brief Bind the evolve cudaq function for circuit simulator
void bindPyEvolve(nanobind::module_ &mod) {
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
