/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/sample.h"
#include "utils/OpaqueArguments.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <fmt/core.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace cudaq {
void pyAltLaunchKernel(const std::string &, MlirModule, OpaqueArguments &,
                       const std::vector<std::string> &);

void bindSampleAsync(py::module &mod) {
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
           py::call_guard<py::gil_scoped_release>(),
           "Return the :class:`SampleResult` from the asynchronous sample "
           "execution.\n")
      .def("__str__", [](async_sample_result &res) {
        std::stringstream ss;
        ss << res;
        return ss.str();
      });

  mod.def(
      "sample_async",
      [&](py::object kernel, py::args args, std::size_t shots,
          std::size_t qpu_id) {
        kernel.inc_ref();
        auto &platform = cudaq::get_platform();
        if (py::hasattr(kernel, "compile"))
          kernel.attr("compile")();

        auto kernelName = kernel.attr("name").cast<std::string>();
        auto kernelMod = kernel.attr("module").cast<MlirModule>();
        auto kernelFunc = getKernelFuncOp(kernelMod, kernelName);

        args = simplifiedValidateInputArguments(args);
        auto *argData = new cudaq::OpaqueArguments();
        cudaq::packArgs(*argData, args, kernelFunc,
                        [](OpaqueArguments &, py::object &) { return false; });

        // The function below will be executed multiple times
        // if the kernel has conditional feedback. In that case,
        // we have to be careful about deleting the `argData` and
        // only do so after the last invocation of that function.

        // Look and see if this is a kernel with conditional feedback
        bool hasQubitMeasurementFeedback =
            unwrap(kernelMod)
                .lookupSymbol<mlir::func::FuncOp>("__nvqpp__mlirgen__" +
                                                  kernelName)
                ->hasAttrOfType<mlir::BoolAttr>("qubitMeasurementFeedback");

        // Should only have C++ going on here, safe to release the GIL
        py::gil_scoped_release release;
        return cudaq::details::runSamplingAsync(
            [argData, kernelName, kernelMod, shots, hasQubitMeasurementFeedback,
             &kernel]() mutable {
              static std::size_t localShots = 0;
              pyAltLaunchKernel(kernelName, kernelMod, *argData, {});
              // delete the raw arg data pointer.
              if (hasQubitMeasurementFeedback) {
                if (localShots == shots - 1)
                  delete argData;
                localShots++;
                kernel.dec_ref();
              }
            },
            platform, kernelName, shots, qpu_id);
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
}
} // namespace cudaq
