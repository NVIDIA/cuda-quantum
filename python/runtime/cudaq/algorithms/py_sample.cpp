/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include <pybind11/stl.h>

#include "py_sample.h"
#include "utils/OpaqueArguments.h"

#include "cudaq.h"
#include "cudaq/algorithms/sample.h"
#include "cudaq/builder.h"
#include "cudaq/platform.h"

#include "common/ExecutionContext.h"
#include "common/MeasureCounts.h"

namespace cudaq {

/// @brief Sample the state produced by the provided builder.
sample_result pySample(kernel_builder<> &builder, py::args args = {},
                       std::size_t shots = 1000) {
  // Ensure the user input is correct.
  auto validatedArgs = validateInputArguments(builder, args);

  cudaq::info("Sampling the provided pythonic kernel.");
  builder.jitCode();
  auto kernelName = builder.name();
  auto &platform = cudaq::get_platform();

  // Map py::args to OpaqueArguments handle
  OpaqueArguments argData;
  packArgs(argData, validatedArgs);
  return details::runSampling(
             [&]() mutable { builder.jitAndInvoke(argData.data()); }, platform,
             kernelName, shots)
      .value();
}

/// @brief Broadcast the sample call over the list-like arguments provided.
std::vector<sample_result> pySampleN(kernel_builder<> &kernel,
                                     py::args args = {},
                                     std::size_t shots = 1000) {
  auto argSet = createArgumentSet(args);
  std::vector<sample_result> results;
  for (auto &a : argSet) {
    results.emplace_back(pySample(kernel, a, shots));
  }

  return results;
}

/// @brief Asynchronously sample the state produced by the provided builder.
/// Return a future-like result.
async_sample_result pySampleAsync(kernel_builder<> &builder,
                                  std::size_t qpu_id = 0,
                                  std::size_t shots = 1000,
                                  py::args args = {}) {
  // Ensure the user input is correct.
  auto validatedArgs = validateInputArguments(builder, args);
  auto &platform = cudaq::get_platform();
  cudaq::info("Asynchronously sampling the provided pythonic kernel.");
  builder.jitCode();
  auto kernelName = builder.name();

  return details::runSamplingAsync(
      [&, a = std::move(validatedArgs)]() mutable {
        OpaqueArguments argData;
        packArgs(argData, a);
        builder.jitAndInvoke(argData.data());
      },
      platform, kernelName, shots, qpu_id);
}

void bindSample(py::module &mod) {

  py::class_<async_sample_result>(
      mod, "AsyncSampleResult",
      "A data-type containing the results of a call to :func:`sample_async`. "
      "The `AsyncSampleResult` contains a future, whose "
      ":class:`SampleResult` "
      "may be returned via an invocation of the `get` method. This "
      "kicks "
      "off a wait on the current thread until the results are available.\n"
      "See `future <https://en.cppreference.com/w/cpp/thread/future>`_ "
      "for more information on this programming pattern.\n")
      .def(py::init([](std::string inJson) {
        async_sample_result f;
        std::istringstream is(inJson);
        is >> f;
        return f;
      }))
      .def("get", &async_sample_result::get,
           "Return the :class:`SampleResult` from the asynchronous sample "
           "execution.\n")
      .def("__str__", [](async_sample_result &res) {
        std::stringstream ss;
        ss << res;
        return ss.str();
      });

  mod.def(
      "sample",
      [&](kernel_builder<> &builder, py::args arguments, std::size_t shots,
          std::optional<noise_model> noise) {
        if (!noise)
          return pySample(builder, arguments, shots);

        set_noise(*noise);
        auto res = pySample(builder, arguments, shots);
        unset_noise();
        return res;
      },
      py::arg("kernel"), py::kw_only(), py::arg("shots_count") = 1000,
      py::arg("noise_model") = py::none(),
      "Sample the state of the provided `kernel` at the specified number "
      "of circuit executions (`shots_count`).\n"
      "\nArgs:\n"
      "  kernel (:class:`Kernel`): The :class:`Kernel` to execute "
      "`shots_count` "
      "times on the QPU.\n"
      "  *arguments (Optional[Any]): The concrete values to evaluate the "
      "kernel "
      "function at. Leave empty if the kernel doesn't accept any arguments.\n"
      "  shots_count (Optional[int]): The number of kernel executions on the "
      "QPU. Defaults "
      "to 1000. Key-word only.\n"
      "  noise_model (Optional[`NoiseModel`]): The optional "
      ":class:`NoiseModel` to add "
      "noise to the kernel execution on the simulator. Defaults to an empty "
      "noise model.\n"
      "\nReturns:\n"
      "  :class:`SampleResult` : A dictionary containing the measurement "
      "count results "
      "for the :class:`Kernel`.\n");

  mod.def(
      "sample_async",
      [&](kernel_builder<> &builder, py::args args, std::size_t shots,
          std::size_t qpu_id) {
        return pySampleAsync(builder, qpu_id, shots, args);
      },
      py::arg("kernel"), py::kw_only(), py::arg("shots_count") = 1000,
      py::arg("qpu_id") = 0,
      "Asynchronously sample the state of the provided `kernel` at the "
      "specified number of circuit executions (`shots_count`).\n"
      "When targeting a quantum platform "
      "with more than one QPU, the optional `qpu_id` allows for control over "
      "which QPU to enable. Will return a future whose results can be "
      "retrieved via "
      "`future.get()`.\n"
      "\nArgs:\n"
      "  kernel (:class:`Kernel`): The :class:`Kernel` to execute "
      "`shots_count` "
      "times on the QPU.\n"
      "  *arguments (Optional[Any]): The concrete values to evaluate the "
      "kernel "
      "function at. Leave empty if the kernel doesn't accept any arguments.\n"
      "  shots_count (Optional[int]): The number of kernel executions on the "
      "QPU. Defaults "
      "to 1000. Key-word only.\n"
      "  qpu_id (Optional[int]): The optional identification for which QPU "
      "on "
      "the platform to target. Defaults to zero. Key-word only.\n"
      "\nReturns:\n"
      "  :class:`AsyncSampleResult` : A dictionary containing the "
      "measurement "
      "count results "
      "for the :class:`Kernel`.\n");

  mod.def(
      "sample_n",
      [](kernel_builder<> &self, py::args args, std::size_t shots,
         std::optional<noise_model> noise) {
        if (!noise)
          return pySampleN(self, args, shots);
        set_noise(*noise);
        auto res = pySampleN(self, args, shots);
        unset_noise();
        return res;
      },
      py::arg("kernel"), py::kw_only(), py::arg("shots_count") = 1000,
      py::arg("noise_model") = py::none(),
      "Broadcast the sample function over the input argument set."
      "For each argument type in the kernel signature, you must provide a"
      "list of arguments of that type. "
      "This function samples the state of the provided `kernel` at each "
      "set of arguments provided for the specified number "
      "of circuit executions (`shots_count`).\n"
      "\nArgs:\n"
      "  kernel (:class:`Kernel`): The :class:`Kernel` to execute "
      "`shots_count` "
      "times on the QPU.\n"
      "  *arguments (Optional[Any]): The concrete values to evaluate the "
      "kernel. Each argument must be a list of instances of the type specified "
      "by the kernel signature."
      "function at. Leave empty if the kernel doesn't accept any arguments.\n"
      "  shots_count (Optional[int]): The number of kernel executions on the "
      "QPU. Defaults "
      "to 1000. Key-word only.\n"
      "  noise_model (Optional[`NoiseModel`]): The optional "
      ":class:`NoiseModel` to add "
      "noise to the kernel execution on the simulator. Defaults to an empty "
      "noise model.\n"
      "\nReturns:\n"
      "  :class:`SampleResult` : A dictionary containing the measurement "
      "count results "
      "for the :class:`Kernel`.\n");
}

} // namespace cudaq
