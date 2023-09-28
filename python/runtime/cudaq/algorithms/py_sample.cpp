/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <pybind11/stl.h>
#include <variant>

#include "py_sample.h"
#include "utils/OpaqueArguments.h"

#include "cudaq.h"
#include "cudaq/algorithms/sample.h"
#include "cudaq/builder.h"
#include "cudaq/platform.h"

#include "common/ExecutionContext.h"
#include "common/Logger.h"
#include "common/MeasureCounts.h"

namespace cudaq {

/// @brief Sample the state produced by the provided builder.
sample_result pySample(kernel_builder<> &builder, py::args args = {},
                       std::size_t shots = 1000,
                       std::optional<noise_model> noise = std::nullopt) {
  // Ensure the user input is correct.
  auto validatedArgs = validateInputArguments(builder, args);

  cudaq::info("Sampling the provided pythonic kernel.");
  builder.jitCode();
  auto kernelName = builder.name();

  // Map py::args to OpaqueArguments handle
  OpaqueArguments argData;
  packArgs(argData, validatedArgs);

  auto &platform = cudaq::get_platform();
  if (noise)
    platform.set_noise(&*noise);

  auto result = details::runSampling(
                    [&]() mutable { builder.jitAndInvoke(argData.data()); },
                    platform, kernelName, shots)
                    .value();
  platform.reset_noise();
  return result;
}

/// @brief Broadcast the sample call over the list-like arguments provided.
std::vector<sample_result>
pySampleN(kernel_builder<> &kernel, py::args args = {},
          std::size_t shots = 1000,
          std::optional<noise_model> noise = std::nullopt) {
  auto argSet = createArgumentSet(args);
  auto N = argSet.size();
  kernel.jitCode();
  auto name = kernel.name();
  auto &platform = cudaq::get_platform();
  if (noise)
    platform.set_noise(&*noise);

  std::vector<sample_result> results;
  for (std::size_t currentIter = 0; auto &a : argSet) {
    auto validatedArgs = validateInputArguments(kernel, a);
    OpaqueArguments argData;
    packArgs(argData, validatedArgs);
    auto ret = details::runSampling(
                   [&]() mutable { kernel.jitAndInvoke(argData.data()); },
                   platform, name, shots, 0, nullptr, currentIter, N)
                   .value();
    currentIter++;
    results.push_back(ret);
  }
  platform.reset_noise();
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
      R"#(A data-type containing the results of a call to :func:`sample_async`. 
The `AsyncSampleResult` models a future-like type, whose 
:class:`SampleResult` may be returned via an invocation of the `get` method. This 
kicks off a wait on the current thread until the results are available.
See `future <https://en.cppreference.com/w/cpp/thread/future>`_ 
for more information on this programming pattern.)#")
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
          std::optional<noise_model> noise)
          -> std::variant<sample_result, std::vector<sample_result>> {
        // Our arguments can be a single set of arguments for the kernel
        // in that case the applicator should just be delegating to pySample
        std::function<std::vector<sample_result>(kernel_builder<> &, py::args,
                                                 std::size_t,
                                                 std::optional<noise_model>)>
            applicator = [](kernel_builder<> &builder, py::args arguments,
                            std::size_t shots,
                            std::optional<noise_model> noise) {
              return std::vector<sample_result>{
                  pySample(builder, arguments, shots, noise)};
            };

        // But our arguments could also be an ArgumentSet, if so
        // we'll delegate to sample_n.
        if (isBroadcastRequest(builder, arguments))
          applicator = [](kernel_builder<> &builder, py::args arguments,
                          std::size_t shots, std::optional<noise_model> noise) {
            return pySampleN(builder, arguments, shots, noise);
          };

        // Execute
        std::vector<sample_result> res =
            applicator(builder, arguments, shots, noise);

        // Return a single value if res.size() == 1
        if (res.size() == 1)
          return {res[0]};

        return {res};
      },
      py::arg("kernel"), py::kw_only(), py::arg("shots_count") = 1000,
      py::arg("noise_model") = py::none(),
      R"#(Sample the state generated by the provided `kernel` at the given kernel 
`arguments` over the specified number of circuit executions (`shots_count`). 
Each argument in `arguments` provided can be a list or ndarray of arguments  
of the specified kernel argument type, and in this case, the `sample` 
functionality will be broadcasted over all argument sets and a list of 
`sample_result` instances will be returned.

Args:
  kernel (:class:`Kernel`): The :class:`Kernel` to execute `shots_count`
    times on the QPU.
  *arguments (Optional[Any]): The concrete values to evaluate the kernel
    function at. Leave empty if the kernel doesn't accept any arguments. For 
    example, if the kernel takes two `float` values as input, the `sample` call 
    should be structured as `cudaq.sample(kernel, firstFloat, secondFloat)`. For 
    broadcasting of the `sample` function, the arguments should be structured as a 
    `list` or `ndarray` of argument values of the specified kernel argument type.
  shots_count (Optional[int]): The number of kernel executions on the QPU.
    Defaults to 1000. Key-word only.
  noise_model (Optional[`NoiseModel`]): The optional :class:`NoiseModel`
    to add noise to the kernel execution on the simulator. Defaults to
    an empty noise model.

Returns:
  :class:`SampleResult` or `list[SampleResult]`: 
  A dictionary containing the measurement count results for the :class:`Kernel`, 
  or a list of such results in the case of `sample` function broadcasting.)#");

  mod.def(
      "sample_async",
      [&](kernel_builder<> &builder, py::args args, std::size_t shots,
          std::size_t qpu_id) {
        return pySampleAsync(builder, qpu_id, shots, args);
      },
      py::arg("kernel"), py::kw_only(), py::arg("shots_count") = 1000,
      py::arg("qpu_id") = 0,
      R"#(Asynchronously sample the state of the provided `kernel` at the 
specified number of circuit executions (`shots_count`).
When targeting a quantum platform with more than one QPU, the optional
`qpu_id` allows for control over which QPU to enable. Will return a
future whose results can be retrieved via `future.get()`.

Args:
  kernel (:class:`Kernel`): The :class:`Kernel` to execute `shots_count` 
    times on the QPU.
  *arguments (Optional[Any]): The concrete values to evaluate the kernel 
    function at. Leave empty if the kernel doesn't accept any arguments.
  shots_count (Optional[int]): The number of kernel executions on the 
    QPU. Defaults to 1000. Key-word only.
  qpu_id (Optional[int]): The optional identification for which QPU 
    on the platform to target. Defaults to zero. Key-word only.

Returns:
  :class:`AsyncSampleResult`: 
  A dictionary containing the measurement count results for the :class:`Kernel`.)#");

  mod.def(
      "sample_n",
      [](kernel_builder<> &self, py::args args, std::size_t shots,
         std::optional<noise_model> noise) {
        PyErr_WarnEx(PyExc_DeprecationWarning,
                     "sample_n() is deprecated, use sample() with the same "
                     "argument-list structure.",
                     1);
        if (!noise)
          return pySampleN(self, args, shots);
        set_noise(*noise);
        auto res = pySampleN(self, args, shots);
        unset_noise();
        return res;
      },
      py::arg("kernel"), py::kw_only(), py::arg("shots_count") = 1000,
      py::arg("noise_model") = py::none(),
      R"#(Broadcast the sample function over the input argument set.
For each argument type in the kernel signature, you must provide a
list of arguments of that type. This function samples the state of
the provided `kernel` at each set of arguments provided for the
specified number of circuit executions (`shots_count`).

Args:
  kernel (:class:`Kernel`): The :class:`Kernel` to execute `shots_count` 
      times on the QPU.
  *arguments (Optional[Any]): The concrete values to evaluate the 
      kernel. Each argument must be a list of instances of the
      type specified by the kernel signature. Leave empty if the kernel
      doesn't accept any arguments.
  shots_count (Optional[int]): The number of kernel executions on the QPU.
      Defaults to 1000. Key-word only.
  noise_model (Optional[`NoiseModel`]): The optional :class:`NoiseModel`
      to add noise to the kernel execution on the simulator. Defaults to an empty 
      noise model.
      
Returns:
  `list[SampleResult]`: 
  A list of dictionary containing the measurement count for each invocation
  of sample for the :class:`Kernel`.)#");
}

} // namespace cudaq
