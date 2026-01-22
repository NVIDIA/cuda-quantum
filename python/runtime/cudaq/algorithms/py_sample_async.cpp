/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_sample_async.h"
#include "common/DeviceCodeRegistry.h"
#include "cudaq/algorithms/sample.h"
#include "runtime/cudaq/platform/py_alt_launch_kernel.h"
#include "utils/OpaqueArguments.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <fmt/core.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using namespace cudaq;

static async_sample_result sample_async_impl(
    const std::string &shortName, MlirModule module, MlirType returnTy,
    std::size_t shots_count, std::optional<noise_model> noise_model,
    bool explicit_measurements, std::size_t qpu_id, py::args runtimeArgs) {
  mlir::ModuleOp mod = unwrap(module);
  runtimeArgs = simplifiedValidateInputArguments(runtimeArgs);

  std::string kernelName = shortName;
  auto retTy = unwrap(returnTy);
  auto &platform = get_platform();
  if (noise_model.has_value()) {
    if (platform.is_remote())
      throw std::runtime_error(
          "Noise model is not supported on remote platforms.");
    platform.set_noise(&noise_model.value());
  }
  auto fnOp = getKernelFuncOp(mod, shortName);
  auto opaques = marshal_arguments_for_module_launch(mod, runtimeArgs, fnOp);

  // Should only have C++ going on here, safe to release the GIL
  py::gil_scoped_release release;
  return details::runSamplingAsync(
      // Notes:
      // (1) no Python data access is allowed in this lambda body.
      // (2) This lambda might be executed multiple times, e.g, when
      // the kernel contains measurement feedback.
      detail::make_copyable_function([opaques = std::move(opaques), kernelName,
                                      retTy, mod = mod.clone()]() mutable {
        [[maybe_unused]] auto result =
            clean_launch_module(kernelName, mod, retTy, opaques);
      }),
      platform, kernelName, shots_count, explicit_measurements, qpu_id);
}

void cudaq::bindSampleAsync(py::module &mod) {
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

  py::class_<async_sample_result>(mod, "AsyncSampleResultImpl",
                                  R"#(
A data-type containing the results of a call to :func:`sample_async`.  The
`AsyncSampleResult` models a future-like type, whose :class:`SampleResult` may
be returned via an invocation of the `get` method.  This kicks off a wait on the
current thread until the results are available.  See `future
<https://en.cppreference.com/w/cpp/thread/future>`_ for more information on this
programming pattern.
)#")
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
      .def(
          "__str__",
          [](async_sample_result &res) {
            std::stringstream ss;
            ss << res;
            return ss.str();
          },
          "FIXME: document");

  mod.def("sample_async_impl", sample_async_impl, "FIXME: document");
}
