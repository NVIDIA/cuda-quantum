/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/run.h"
#include "runtime/cudaq/platform/py_alt_launch_kernel.h"
#include "utils/OpaqueArguments.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <fmt/core.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace cudaq {

// Internal struct representing buffer to be filled asynchronously.
// When the `ready` future is set, the content of the buffer is filled.
struct async_run_result {
  std::future<void> ready;
  py::buffer buffer;
};

void bindRunAsync(py::module &mod) {
  py::class_<async_run_result>(mod, "AsyncRunResult", "")
      .def(
          "get",
          [](async_run_result &self) {
            self.ready.get();
            return self.buffer;
          },
          "");
  // Internal `run_async` implementation
  mod.def(
      "run_async_internal",
      [&](py::buffer resultBuffer, py::object kernel, py::args args,
          std::size_t shots, std::optional<noise_model> noise_model,
          std::size_t qpu_id) {
        kernel.inc_ref();
        auto &platform = cudaq::get_platform();
        auto kernelName = kernel.attr("name").cast<std::string>();
        auto kernelMod = kernel.attr("module").cast<MlirModule>();
        args = simplifiedValidateInputArguments(args);
        auto *argData = toOpaqueArgs(args, kernelMod, kernelName);
        // Note: we only pass the underlying buffer pointer to the async.
        // functor to load the result data. i.e., no Python object is accessed
        // during the asynchronous execution.
        void *bufferPtr = resultBuffer.request().ptr;
        async_run_result result;
        result.buffer = resultBuffer;
        // Should only have C++ going on here, safe to release the GIL
        py::gil_scoped_release release;

        std::promise<void> promise;
        result.ready = promise.get_future();
        QuantumTask wrapped = detail::make_copyable_function(
            [p = std::move(promise), bufferPtr, shots, &platform, argData,
             kernelName, kernelMod,
             noise_model = std::move(noise_model)]() mutable {
              if (noise_model.has_value())
                platform.set_noise(&noise_model.value());
              details::RunResultSpan span = details::runTheKernel(
                  [&]() mutable {
                    pyAltLaunchKernel(kernelName, kernelMod, *argData, {});
                  },
                  platform, kernelName, shots);

              std::memcpy(bufferPtr, span.data, span.lengthInBytes);
              p.set_value();
              platform.reset_noise();
            });
        platform.enqueueAsyncTask(qpu_id, wrapped);
        return result;
      },
      py::arg("result_buffer"), py::arg("kernel"), py::kw_only(),
      py::arg("shots_count") = 1, py::arg("noise_model") = py::none(),
      py::arg("qpu_id") = 0, "");
}
} // namespace cudaq
