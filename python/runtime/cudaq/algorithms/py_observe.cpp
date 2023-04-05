/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "cudaq.h"
#include <pybind11/stl.h>

#include "py_observe.h"
#include "utils/OpaqueArguments.h"

#include "cudaq/platform/quantum_platform.h"

#include "common/Logger.h"

namespace cudaq {

/// @brief Default shots value set to -1
constexpr int defaultShotsValue = -1;

/// @brief Default qpu id value set to 0
constexpr int defaultQpuIdValue = 0;

/// @brief For asynchronous execution, we need to construct OpaqueArguments
/// outside of the async lambda invocation. If we don't, then we will be
/// using Python types outside of the current GIL context. Bad things happen
/// then.
std::unordered_map<std::size_t, std::unique_ptr<OpaqueArguments>>
    asyncArgsHolder;

/// @brief Run `cudaq::observe` on the provided kernel and spin operator.
observe_result pyObserve(kernel_builder<> &kernel, spin_op &spin_operator,
                         py::args args = {}, int shots = defaultShotsValue) {
  // Ensure the user input is correct.
  auto validatedArgs = validateInputArguments(kernel, args);
  auto &platform = cudaq::get_platform();

  // TODO: would like to handle errors in the case that
  // `kernel.num_qubits() >= spin_operator.num_qubits()`
  kernel.jitCode();
  auto name = kernel.name();
  // Does this platform expose more than 1 QPU
  // If so, let's distribute the work amongst the QPUs
  if (auto nQpus = platform.num_qpus(); nQpus > 1)
    return details::distributeComputations(
        [&](std::size_t i, spin_op &op) {
          return pyObserveAsync(kernel, op, args, i, shots);
        },
        spin_operator, nQpus);

  // Launch the observation task
  return details::runObservation(
             [&]() mutable {
               OpaqueArguments argData;
               packArgs(argData, validatedArgs);
               kernel.jitAndInvoke(argData.data());
             },
             spin_operator, platform, shots, name)
      .value();
}

/// @brief Broadcast the observe call over the list-like arguments provided.
std::vector<observe_result> pyObserveN(kernel_builder<> &kernel, spin_op &op,
                                       py::args args = {},
                                       std::size_t shots = 1000) {
  auto argSet = createArgumentSet(args);
  std::vector<observe_result> results;
  for (auto &a : argSet) {
    results.emplace_back(pyObserve(kernel, op, a, shots));
  }

  return results;
}

/// @brief Asynchronously run `cudaq::observe` on the provided kernel and
/// spin operator.
async_observe_result pyObserveAsync(kernel_builder<> &kernel,
                                    spin_op &spin_operator, py::args args = {},
                                    std::size_t qpu_id = defaultQpuIdValue,
                                    int shots = defaultShotsValue) {

  // Ensure the user input is correct.
  auto validatedArgs = validateInputArguments(kernel, args);
  std::hash<std::string> hasher;

  // Create a unique integer key that combines the kernel name
  // and the validated args.
  std::size_t uniqueHash = hasher(kernel.name()) + hasher(py::str(args));

  // Add the opaque args to the holder and pack the args into it
  asyncArgsHolder.emplace(uniqueHash, std::make_unique<OpaqueArguments>());
  packArgs(*asyncArgsHolder.at(uniqueHash).get(), validatedArgs);

  // TODO: would like to handle errors in the case that
  // `kernel.num_qubits() >= spin_operator.num_qubits()`
  kernel.jitCode();
  auto name = kernel.name();
  // Get the platform, first check that the given qpu_id is valid
  auto &platform = cudaq::get_platform();

  // Launch the asynchronous execution.
  return details::runObservationAsync(
      [&kernel, uniqueHash]() mutable {
        auto &argData = asyncArgsHolder.at(uniqueHash);
        kernel.jitAndInvoke(argData->data());
      },
      spin_operator, platform, shots, name, qpu_id);
}

void bindObserve(py::module &mod) {

  // FIXME provide ability to inject noise model here
  mod.def(
      "observe",
      [&](kernel_builder<> &kernel, spin_op &spin_operator, py::args arguments,
          int shots, std::optional<noise_model> noise) {
        if (!noise)
          return pyObserve(kernel, spin_operator, arguments, shots);
        set_noise(*noise);
        auto res = pyObserve(kernel, spin_operator, arguments, shots);
        unset_noise();
        return res;
      },
      py::arg("kernel"), py::arg("spin_operator"), py::kw_only(),
      py::arg("shots_count") = defaultShotsValue,
      py::arg("noise_model") = py::none(),
      "Compute the expected value of the `spin_operator` with respect to "
      "the `kernel`. If the kernel accepts arguments, it will be evaluated "
      "with respect to `kernel(*arguments)`.\n"
      "\nArgs:\n"
      "  kernel (:class:`Kernel`): The :class:`Kernel` to evaluate the "
      "expectation "
      "value with respect to.\n"
      "  spin_operator (:class:`SpinOperator`): The Hermitian spin operator to "
      "calculate "
      "the expectation of.\n"
      "  *arguments (Optional[Any]): The concrete values to evaluate the "
      "kernel "
      "function at. Leave empty if the kernel doesn't accept any arguments.\n"
      "  shots_count (Optional[int]): The number of shots to use for QPU "
      "execution. "
      "Defaults to 1 shot. Key-word only.\n"
      "  noise_model (Optional[`NoiseModel`]): The optional "
      ":class:`NoiseModel` to add "
      "noise to the kernel execution on the simulator. Defaults to an empty "
      "noise model.\n"
      "\nReturns:\n"
      "  :class:`ObserveResult` : A data-type containing the expectation value "
      "of the "
      "`spin_operator` with respect to the `kernel(*arguments)`. If "
      "`shots_count` was "
      "provided, the :class:`ObserveResult` will also contain a "
      ":class:`SampleResult` "
      "dictionary.\n");

  /// Expose observe_async, can optionally take the qpu_id to target.
  mod.def(
      "observe_async",
      [&](kernel_builder<> &kernel, spin_op &spin_operator, py::args arguments,
          std::size_t qpu_id, int shots,
          std::optional<noise_model> noise_model) {
        if (!noise_model)
          return pyObserveAsync(kernel, spin_operator, arguments, qpu_id,
                                shots);
        set_noise(*noise_model);
        auto res =
            pyObserveAsync(kernel, spin_operator, arguments, qpu_id, shots);
        unset_noise();
        return res;
      },
      py::arg("kernel"), py::arg("spin_operator"), py::kw_only(),
      py::arg("qpu_id") = defaultQpuIdValue,
      py::arg("shots_count") = defaultShotsValue,
      py::arg("noise_model") = py::none(),
      "Compute the expected value of the `spin_operator` with respect to "
      "the `kernel` asynchronously. If the kernel accepts arguments, it will "
      "be evaluated with respect to `kernel(*arguments)`.\n"
      "When targeting a quantum platform "
      "with more than one QPU, the optional `qpu_id` allows for control over "
      "which QPU to enable. Will return a future whose results can be "
      "retrieved via "
      "`future.get()`.\n"
      "\nArgs:\n"
      "  kernel (:class:`Kernel`): The :class:`Kernel` to evaluate the "
      "expectation "
      "value with respect to.\n"
      "  spin_operator (:class:`SpinOperator`): The Hermitian spin operator to "
      "calculate "
      "the expectation of.\n"
      "  *arguments (Optional[Any]): The concrete values to evaluate the "
      "kernel "
      "function at. Leave empty if the kernel doesn't accept any arguments.\n"
      "  qpu_id (Optional[int]): The optional identification for which QPU on "
      "the platform to target. Defaults to zero. Key-word only.\n"
      "  shots_count (Optional[int]): The number of shots to use for QPU "
      "execution. "
      "Defaults to 1 shot. Key-word only.\n"
      "  noise_model (Optional[`NoiseModel`]): The optional "
      ":class:`NoiseModel` to add "
      "noise to the kernel execution on the simulator. Defaults to an empty "
      "noise model.\n"
      "\nReturns:\n"
      "  :class:`AsyncObserveResult` : A future containing the result of the "
      "call to observe.\n");

  mod.def(
      "observe_n",
      [](kernel_builder<> &self, spin_op &spin_operator, py::args args,
         std::size_t shots, std::optional<noise_model> noise_model) {
        if (!noise_model)
          return pyObserveN(self, spin_operator, args, shots);
        set_noise(*noise_model);
        auto res = pyObserveN(self, spin_operator, args, shots);
        unset_noise();
        return res;
      },
      py::arg("kernel"), py::arg("spin_operator"), py::kw_only(),
      py::arg("shots_count") = 1000, py::arg("noise_model") = py::none(),
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
      "count results for the :class:`Kernel`.\n");
}

} // namespace cudaq
