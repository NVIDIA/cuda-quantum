/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/observe.h"
#include "utils/OpaqueArguments.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

#include <fmt/core.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace cudaq {
inline constexpr int defaultShotsValue = -1;
inline constexpr int defaultQpuIdValue = 0;

void pyAltLaunchKernel(const std::string &, MlirModule, OpaqueArguments &,
                       const std::vector<std::string> &);

void bindObserveAsync(py::module &mod) {
  mod.def(
      "observe_async",
      [&](py::object &kernel, spin_op &spin_operator, py::args args,
          std::size_t qpu_id, int shots) {
        auto kernelBlockArgs = kernel.attr("arguments");
        if (py::len(kernelBlockArgs) != args.size())
          throw std::runtime_error(
              "Invalid number of arguments passed to observe_async.");

        auto &platform = cudaq::get_platform();
        auto *argData = new cudaq::OpaqueArguments();
        args = simplifiedValidateInputArguments(args);
        cudaq::packArgs(*argData, args,
                        [](OpaqueArguments &, py::object &) { return false; });
        auto kernelName = kernel.attr("name").cast<std::string>();
        auto kernelMod = kernel.attr("module").cast<MlirModule>();

        // Launch the asynchronous execution.
        py::gil_scoped_release release;
        auto ret = details::runObservationAsync(
            [argData, kernelName, kernelMod]() mutable {
              pyAltLaunchKernel(kernelName, kernelMod, *argData, {});
              delete argData;
            },
            spin_operator, platform, shots, kernelName, qpu_id);
        return ret;
      },
      py::arg("kernel"), py::arg("spin_operator"), py::kw_only(),
      py::arg("qpu_id") = defaultQpuIdValue,
      py::arg("shots_count") = defaultShotsValue,
      R"#(Compute the expected value of the `spin_operator` with respect to 
the `kernel` asynchronously. If the kernel accepts arguments, it will 
be evaluated with respect to `kernel(*arguments)`. When targeting a
quantum platform with more than one QPU, the optional `qpu_id` allows
for control over which QPU to enable. Will return a future whose results
can be retrieved via `future.get()`.

Args:
  kernel (:class:`Kernel`): The :class:`Kernel` to evaluate the 
    expectation value with respect to.
  spin_operator (:class:`SpinOperator`): The Hermitian spin operator to 
    calculate the expectation of.
  *arguments (Optional[Any]): The concrete values to evaluate the 
    kernel function at. Leave empty if the kernel doesn't accept any arguments.
  qpu_id (Optional[int]): The optional identification for which QPU on 
    the platform to target. Defaults to zero. Key-word only.
  shots_count (Optional[int]): The number of shots to use for QPU 
    execution. Defaults to -1 implying no shots-based sampling. Key-word only.

Returns:
  :class:`AsyncObserveResult`: 
  A future containing the result of the call to observe.)#");
}
} // namespace cudaq