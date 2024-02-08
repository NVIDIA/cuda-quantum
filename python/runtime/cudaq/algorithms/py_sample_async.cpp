/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/sample.h"
#include "utils/OpaqueArguments.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

#include <fmt/core.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace cudaq {
void pyAltLaunchKernel(const std::string &, MlirModule, OpaqueArguments &,
                       const std::vector<std::string> &);

static std::vector<py::object> eagerAsyncSampleArgs;

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
      [&](py::object &kernel, py::args args, std::size_t shots,
          std::size_t qpu_id) {
        auto &platform = cudaq::get_platform();
        auto kernelName = kernel.attr("name").cast<std::string>();

        if (py::hasattr(kernel, "library_mode") &&
            kernel.attr("library_mode").cast<py::bool_>()) {

          // Here we know we are in eager mode. We do not have a
          // MLIR Module, we just have the python kernel as a callback.
          // Great care must be taken here with regards to the
          // GIL and when pythonic data types (PyObject*) gets
          // destructed. First, we release the GIL since we are
          // about to launch a new C++ thread. Within the kernel
          // functor, we must first acquire the GIL so that we can
          // invoke the Python callback with the Python *args...
          // The args were captured by value, so they will be destructed
          // when the lambda implicit type gets destructed, and the
          // GIL will not be acquired at that point, leading to issues
          // in destructing the args. So instead we release ownership
          // of the py::args, and borrow it to a py::object, which
          // we store globally. Then it should have ref_count = 1
          // and when the static vector gets destroyed, the ref_count
          // will drop to 0 and the PyObject will be deallocated.
          py::gil_scoped_release release;
          return cudaq::details::runSamplingAsync(
              [kernelName, kernel, args]() mutable {
                // Acquire the gil and call the callback
                py::gil_scoped_acquire gil;
                kernel(*args);
                // Take ownership of the args so they get
                // deleted when we have GIL ownership
                eagerAsyncSampleArgs.emplace_back(args.release(), true);
              },
              platform, kernelName, shots, qpu_id);
        }

        auto *argData = new cudaq::OpaqueArguments();
        cudaq::packArgs(*argData, args);
        auto kernelMod = kernel.attr("module").cast<MlirModule>();

        // Should only have C++ going on here, safe to release the GIL
        py::gil_scoped_release release;
        return cudaq::details::runSamplingAsync(
            [argData, kernelName, kernelMod]() mutable {
              pyAltLaunchKernel(kernelName, kernelMod, *argData, {});
              // delete the raw arg data pointer.
              delete argData;
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