/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
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
std::string get_quake_by_name(const std::string &, bool,
                              std::optional<std::string>);

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
          bool explicitMeasurements, std::size_t qpu_id) {
        // The kernel object can be an 'rvalue' created in the `sample_async`
        // call, e.g., `sample_async(kernel_factory())`, where
        // `kernel_factory()` returns a kernel object. Hence, care must be taken
        // to make sure it's still alive in the async. functor that we post to
        // the execution queue.
        // (1) Manually increase the reference count so that the underlying
        // `PyObject` is alive.
        kernel.inc_ref();
        // (2) A unique_ptr with a custom deleter is used to hold the `PyObject`
        // ptr and decrement the reference count when going out of scope.
        auto kernelDeleter = [](PyObject *ptr) {
          // Reacquire the GIL.
          // Note: We've finished the async. execution at this point.
          py::gil_scoped_acquire acquire;
          py::handle(ptr).dec_ref();
        };
        std::unique_ptr<PyObject, decltype(kernelDeleter)> kernelPtr(
            kernel.ptr(), kernelDeleter);

        auto &platform = cudaq::get_platform();
        if (py::hasattr(kernel, "compile"))
          kernel.attr("compile")();

        auto kernelName = kernel.attr("name").cast<std::string>();
        auto kernelMod = kernel.attr("module").cast<MlirModule>();
        args = simplifiedValidateInputArguments(args);

        // This kernel may not have been registered to the quake registry
        // (usually, the first invocation would register the kernel)
        // i.e., `cudaq::kernelHasConditionalFeedback` won't be able to tell if
        // this kernel has qubit measurement feedback on the first invocation.
        if (cudaq::get_quake_by_name(kernelName, false, std::nullopt).empty()) {
          auto moduleOp = unwrap(kernelMod);
          std::string mlirCode;
          llvm::raw_string_ostream outStr(mlirCode);
          mlir::OpPrintingFlags opf;
          moduleOp.print(outStr, opf);
          cudaq::registry::__cudaq_deviceCodeHolderAdd(kernelName.c_str(),
                                                       mlirCode.c_str());
        }

        // The function below will be executed multiple times
        // if the kernel has conditional feedback. In that case,
        // we have to be careful about deleting the `argData` and
        // only do so after the last invocation of that function.
        // Hence, pass it as a unique_ptr for the functor to manage its
        // lifetime.
        std::unique_ptr<OpaqueArguments> argData(
            toOpaqueArgs(args, kernelMod, kernelName));

        // Should only have C++ going on here, safe to release the GIL
        py::gil_scoped_release release;
        return cudaq::details::runSamplingAsync(
            // Notes:
            // (1) no Python data access is allowed in this lambda body.
            // (2) This lambda might be executed multiple times, e.g, when the
            // kernel contains measurement feedback.
            cudaq::detail::make_copyable_function(
                [argData = std::move(argData), kernel = std::move(kernelPtr),
                 kernelName, kernelMod, shots]() mutable {
                  pyAltLaunchKernel(kernelName, kernelMod, *argData, {});
                }),
            platform, kernelName, shots, explicitMeasurements, qpu_id);
      },
      py::arg("kernel"), py::kw_only(), py::arg("shots_count") = 1000,
      py::arg("explicit_measurements") = false, py::arg("qpu_id") = 0,
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
  explicit_measurements (Optional[bool]): A flag to indicate whether or not to 
    concatenate measurements in execution order for the returned sample result.
  qpu_id (Optional[int]): The optional identification for which QPU 
    on the platform to target. Defaults to zero. Key-word only.

Returns:
  :class:`AsyncSampleResult`: 
  A dictionary containing the measurement count results for the :class:`Kernel`.)#");
}
} // namespace cudaq
