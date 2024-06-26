/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ExecutionContext.h"
#include "cudaq/platform.h"
#include <fmt/core.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace cudaq {

void bindExecutionContext(py::module &mod) {
  py::class_<cudaq::ExecutionContext>(mod, "ExecutionContext")
      .def(py::init<std::string>())
      .def(py::init<std::string, int>())
      .def_readonly("result", &cudaq::ExecutionContext::result)
      .def_readwrite("hasConditionalsOnMeasureResults",
                     &cudaq::ExecutionContext::hasConditionalsOnMeasureResults)
      .def_readwrite("totalIterations",
                     &cudaq::ExecutionContext::totalIterations)
      .def_readwrite("batchIteration", &cudaq::ExecutionContext::batchIteration)
      .def("setSpinOperator", [](cudaq::ExecutionContext &ctx,
                                 cudaq::spin_op &spin) { ctx.spin = &spin; })
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
}
} // namespace cudaq