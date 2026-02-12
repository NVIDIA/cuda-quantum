/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ExecutionContext.h"
#include "common/RecordLogParser.h"
#include "cudaq/platform.h"
#include "cudaq/utils/cudaq_utils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include <fmt/core.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/map.h>

namespace py = nanobind;

namespace nvqir {
std::string_view getQirOutputLog();
void clearQirOutputLog();
} // namespace nvqir

namespace cudaq {

void bindExecutionContext(py::module_ &mod) {
  py::class_<cudaq::ExecutionContext>(mod, "ExecutionContext")
      .def(py::init<std::string>())
      .def(py::init<std::string, std::size_t, std::size_t>(), py::arg("name"),
           py::arg("shots"), py::arg("qpu_id") = 0)
      .def_rw("kernelName", &cudaq::ExecutionContext::kernelName)
      .def_ro("result", &cudaq::ExecutionContext::result)
      .def_rw("asyncExec", &cudaq::ExecutionContext::asyncExec)
      .def_ro("asyncResult", &cudaq::ExecutionContext::asyncResult)
      .def_rw("hasConditionalsOnMeasureResults",
                     &cudaq::ExecutionContext::hasConditionalsOnMeasureResults)
      .def_rw("totalIterations",
                     &cudaq::ExecutionContext::totalIterations)
      .def_rw("batchIteration", &cudaq::ExecutionContext::batchIteration)
      .def_rw("numberTrajectories",
                     &cudaq::ExecutionContext::numberTrajectories)
      .def_rw("explicitMeasurements",
                     &cudaq::ExecutionContext::explicitMeasurements)
      .def_rw("allowJitEngineCaching",
                     &cudaq::ExecutionContext::allowJitEngineCaching)
      .def_ro("invocationResultBuffer",
                    &cudaq::ExecutionContext::invocationResultBuffer)
      .def("unset_jit_engine",
           [&](cudaq::ExecutionContext &execCtx) {
             if (execCtx.jitEng) {
               execCtx.jitEng = std::nullopt;
               execCtx.allowJitEngineCaching = false;
             }
           })
      .def("setSpinOperator",
           [](cudaq::ExecutionContext &ctx, cudaq::spin_op &spin) {
             ctx.spin = spin;
             assert(cudaq::spin_op::canonicalize(spin) == spin);
           })
      .def("getExpectationValue",
           [](cudaq::ExecutionContext &ctx) { return ctx.expectationValue; })
      // ----- Context management using with blocks -----
      // Unlike in C++, we do not support nested execution contexts in Python.
      .def("__enter__",
           [](cudaq::ExecutionContext &ctx) -> ExecutionContext & {
             if (cudaq::getExecutionContext()) {
               throw std::runtime_error("Context already set. Nested execution "
                                        "contexts are not supported in Python");
             }
             auto &platform = cudaq::get_platform();
             platform.configureExecutionContext(ctx);
             cudaq::detail::setExecutionContext(&ctx);
             platform.beginExecution();
             return ctx;
           },
           py::rv_policy::reference)
      .def("__exit__", [](cudaq::ExecutionContext &ctx, py::handle type,
                          py::handle value, py::handle traceback) {
        if (type.is_none()) {
          // No exception, so we finalize the context and reset it
          auto &platform = cudaq::get_platform();
          detail::try_finally(
              [&] {
                platform.finalizeExecutionContext(ctx);
                platform.endExecution();
              },
              detail::resetExecutionContext);
        } else {
          // Reset, silencing any further exceptions
          detail::invoke_no_throw(detail::resetExecutionContext);
        }
        // Return false so exceptions are not suppressed
        return false;
      },
      // nanobind rejects None args by default (unlike pybind11);
      // mark each __exit__ parameter as accepting None.
      py::arg().none(), py::arg().none(), py::arg().none());
  mod.def("supportsConditionalFeedback", []() {
    auto &platform = cudaq::get_platform();
    return platform.supports_conditional_feedback();
  });
  mod.def("supportsExplicitMeasurements", []() {
    auto &platform = cudaq::get_platform();
    return platform.supports_explicit_measurements();
  });
  mod.def("getExecutionContextName",
          []() { return cudaq::getExecutionContext()->name; });
  mod.def(
      "isQuantumDevice",
      [](std::size_t qpuId = 0) {
        auto &platform = cudaq::get_platform();
        auto isRemoteSimulator =
            platform.get_remote_capabilities().isRemoteSimulator;
        return !isRemoteSimulator &&
               (platform.is_remote() || platform.is_emulated());
      },
      py::arg("qpuId") = 0);
  mod.def("getQirOutputLog", []() { return nvqir::getQirOutputLog(); });
  mod.def("clearQirOutputLog", []() { nvqir::clearQirOutputLog(); });
  mod.def("decodeQirOutputLog",
          [](const std::string &outputLog, py::object decodedResults) {
            cudaq::RecordLogParser parser;
            parser.parse(outputLog);
            Py_buffer view;
            if (PyObject_GetBuffer(decodedResults.ptr(), &view,
                                   PyBUF_WRITABLE) != 0)
              throw py::python_error();
            // Get the buffer and length of buffer (in bytes) from the parser.
            auto *origBuffer = parser.getBufferPtr();
            const std::size_t bufferSize = parser.getBufferSize();
            std::memcpy(view.buf, origBuffer, bufferSize);
            PyBuffer_Release(&view);
          });
}
} // namespace cudaq
