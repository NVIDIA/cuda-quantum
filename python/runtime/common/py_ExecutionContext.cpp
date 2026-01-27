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
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include <fmt/core.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace nvqir {
std::string_view getQirOutputLog();
void clearQirOutputLog();
} // namespace nvqir

namespace cudaq {

void bindExecutionContext(py::module &mod) {
  py::class_<cudaq::ExecutionContext>(mod, "ExecutionContext")
      .def(py::init<std::string>())
      .def(py::init<std::string, std::size_t>())
      .def(py::init<std::string, std::size_t, std::size_t>())
      .def_readonly("result", &cudaq::ExecutionContext::result)
      .def_readwrite("asyncExec", &cudaq::ExecutionContext::asyncExec)
      .def_readonly("asyncResult", &cudaq::ExecutionContext::asyncResult)
      .def_readwrite("hasConditionalsOnMeasureResults",
                     &cudaq::ExecutionContext::hasConditionalsOnMeasureResults)
      .def_readwrite("totalIterations",
                     &cudaq::ExecutionContext::totalIterations)
      .def_readwrite("batchIteration", &cudaq::ExecutionContext::batchIteration)
      .def_readwrite("numberTrajectories",
                     &cudaq::ExecutionContext::numberTrajectories)
      .def_readwrite("explicitMeasurements",
                     &cudaq::ExecutionContext::explicitMeasurements)
      .def_readwrite("allowJitEngineCaching",
                     &cudaq::ExecutionContext::allowJitEngineCaching)
      .def_readonly("invocationResultBuffer",
                    &cudaq::ExecutionContext::invocationResultBuffer)
      .def("unset_jit_engine",
           [&](cudaq::ExecutionContext &execCtx) {
             if (execCtx.jitEng) {
               auto *p =
                   reinterpret_cast<mlir::ExecutionEngine *>(execCtx.jitEng);
               delete p;
               execCtx.jitEng = nullptr;
               execCtx.allowJitEngineCaching = false;
             }
           })
      .def("setSpinOperator",
           [](cudaq::ExecutionContext &ctx, cudaq::spin_op &spin) {
             ctx.spin = spin;
             assert(cudaq::spin_op::canonicalize(spin) == spin);
           })
      .def("getExpectationValue",
           [](cudaq::ExecutionContext &ctx) { return ctx.expectationValue; });
  mod.def(
      "setExecutionContext",
      [](cudaq::ExecutionContext &ctx) {
        auto &self = cudaq::get_platform();
        self.set_exec_ctx(&ctx);
      },
      "");
  mod.def(
      "resetExecutionContext",
      []() {
        auto &self = cudaq::get_platform();
        self.reset_exec_ctx();
      },
      "");
  mod.def("supportsConditionalFeedback", []() {
    auto &platform = cudaq::get_platform();
    return platform.supports_conditional_feedback();
  });
  mod.def("supportsExplicitMeasurements", []() {
    auto &platform = cudaq::get_platform();
    return platform.supports_explicit_measurements();
  });
  mod.def("getExecutionContextName", []() {
    auto &self = cudaq::get_platform();
    return self.get_exec_ctx()->name;
  });
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
          [](const std::string &outputLog, py::buffer decodedResults) {
            cudaq::RecordLogParser parser;
            parser.parse(outputLog);
            auto info = decodedResults.request();
            // Get the buffer and length of buffer (in bytes) from the parser.
            auto *origBuffer = parser.getBufferPtr();
            const std::size_t bufferSize = parser.getBufferSize();
            std::memcpy(info.ptr, origBuffer, bufferSize);
          });
}
} // namespace cudaq
