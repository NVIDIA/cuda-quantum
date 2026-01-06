/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/sample.h"
#include "cudaq/utils/registry.h"
#include "runtime/cudaq/platform/py_alt_launch_kernel.h"
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
  // Async. result wrapper for Python kernels, which also holds the Python MLIR
  // context.
  //
  // As a kernel is passed to an async. call, its lifetime on the main
  // Python thread decouples from the C++ functor in the execution queue. While
  // we can clone the MLIR module of the kernel in the functor, the context
  // needs to be alive. Hence, we hold the context here to keep it alive on the
  // main Python thread. For example,
  //  `async_handle = sample_async(kernel_factory())`,
  // where `kernel_factory()` returns a kernel object. The `async_handle` would
  // then track a reference (ref count) to the context of the temporary (rval)
  // kernel.
  class py_async_sample_result : public async_sample_result {
  public:
    // Ctors
    py_async_sample_result(async_sample_result &&res, py::object &&mlirCtx)
        : async_sample_result(std::move(res)), ctx(std::move(mlirCtx)){};

  private:
    py::object ctx;
  };

  py::class_<async_sample_result, py_async_sample_result>(
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
        // Check if the kernel has void return type
        if (py::hasattr(kernel, "returnType")) {
          py::object returnType = kernel.attr("returnType");
          if (!returnType.is_none())
            throw std::runtime_error(fmt::format(
                "The `sample_async` API only supports kernels that return None "
                "(void). Consider using `run_async` for kernels that return "
                "values."));
        }
        auto &platform = cudaq::get_platform();
        if (py::hasattr(kernel, "compile"))
          kernel.attr("compile")();
        // Process any callable args
        const auto callableNames = getCallableNames(kernel, args);
        auto kernelName = kernel.attr("name").cast<std::string>();
        // Clone the kernel module
        auto kernelMod = mlirModuleFromOperation(
            wrap(unwrap(kernel.attr("module").cast<MlirModule>())->clone()));
        // Get the MLIR context associated with the kernel
        py::object mlirCtx = kernel.attr("module").attr("context");
        args = simplifiedValidateInputArguments(args);

        // This kernel may not have been registered to the quake registry
        // (usually, the first invocation would register the kernel)
        // i.e., `cudaq::kernelHasConditionalFeedback` won't be able to tell if
        // this kernel has qubit measurement feedback on the first invocation.
        // Thus, add kernel's MLIR code to the registry.
        {
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
            toOpaqueArgs(args, kernelMod, kernelName, getCallableArgHandler()));

        // Should only have C++ going on here, safe to release the GIL
        py::gil_scoped_release release;
        return py_async_sample_result(
            cudaq::details::runSamplingAsync(
                // Notes:
                // (1) no Python data access is allowed in this lambda body.
                // (2) This lambda might be executed multiple times, e.g, when
                // the kernel contains measurement feedback.
                cudaq::detail::make_copyable_function(
                    [argData = std::move(argData), kernelName, kernelMod,
                     callableNames]() mutable {
                      pyAltLaunchKernel(kernelName, kernelMod, *argData,
                                        callableNames);
                    }),
                platform, kernelName, shots, explicitMeasurements, qpu_id),
            std::move(mlirCtx));
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
