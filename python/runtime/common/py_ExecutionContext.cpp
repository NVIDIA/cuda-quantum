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
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nvqir {
std::string_view getQirOutputLog();
void clearQirOutputLog();
} // namespace nvqir

namespace {
class PersistJITEngine {};
} // namespace

namespace cudaq {

void bindExecutionContext(nanobind::module_ &mod) {
  nanobind::class_<cudaq::ExecutionContext>(mod, "ExecutionContext")
      .def(nanobind::init<std::string>())
      .def(nanobind::init<std::string, std::size_t, std::size_t>(),
           nanobind::arg("name"), nanobind::arg("shots"),
           nanobind::arg("qpu_id") = 0)
      .def_rw("kernelName", &cudaq::ExecutionContext::kernelName)
      .def_ro("result", &cudaq::ExecutionContext::result)
      .def_rw("asyncExec", &cudaq::ExecutionContext::asyncExec)
      .def_ro("asyncResult", &cudaq::ExecutionContext::asyncResult)
      .def_rw("hasConditionalsOnMeasureResults",
              &cudaq::ExecutionContext::hasConditionalsOnMeasureResults)
      .def_rw("totalIterations", &cudaq::ExecutionContext::totalIterations)
      .def_rw("batchIteration", &cudaq::ExecutionContext::batchIteration)
      .def_rw("numberTrajectories",
              &cudaq::ExecutionContext::numberTrajectories)
      .def_rw("explicitMeasurements",
              &cudaq::ExecutionContext::explicitMeasurements)
      .def_rw("allowJitEngineCaching",
              &cudaq::ExecutionContext::allowJitEngineCaching)
      .def_rw("useParametricJit", &cudaq::ExecutionContext::useParametricJit)
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
      .def(
          "__enter__",
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
          nanobind::rv_policy::reference)
      .def(
          "__exit__",
          [](cudaq::ExecutionContext &ctx, nanobind::object type,
             nanobind::object value, nanobind::object traceback) {
            if (type.is_none()) {
              // Normal exit: finalize results, clean up the simulator,
              // and reset the context (guaranteed even if finalize throws).
              auto &platform = cudaq::get_platform();
              detail::try_finally(
                  [&] {
                    platform.finalizeExecutionContext(ctx);
                    platform.endExecution();
                  },
                  detail::resetExecutionContext);
            } else {
              // The kernel threw. Still need to tear down the platform so
              // the simulator doesn't carry stale state into the next run.
              // Separate invoke_no_throw so the context reset always runs.
              detail::invoke_no_throw([&] {
                auto &platform = cudaq::get_platform();
                platform.finalizeExecutionContext(ctx);
                platform.endExecution();
              });
              // Always reset context, even if the above cleanup failed.
              detail::invoke_no_throw(detail::resetExecutionContext);
            }
            return false;
          },
          nanobind::arg("type").none(), nanobind::arg("value").none(),
          nanobind::arg("traceback").none());
  mod.def(
      "supportsExplicitMeasurements",
      [](std::size_t qpuId = 0) {
        auto &platform = cudaq::get_platform();
        return platform.supports_explicit_measurements(qpuId);
      },
      nanobind::arg("qpu_id") = 0);
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
      nanobind::arg("qpuId") = 0);
  mod.def("getQirOutputLog", []() { return nvqir::getQirOutputLog(); });
  mod.def("clearQirOutputLog", []() { nvqir::clearQirOutputLog(); });
  mod.def("decodeQirOutputLog",
          [](const std::string &outputLog, nanobind::bytearray decodedResults) {
            cudaq::RecordLogParser parser;
            parser.parse(outputLog);
            auto *origBuffer = parser.getBufferPtr();
            const std::size_t bufferSize = parser.getBufferSize();
            std::memcpy(decodedResults.data(), origBuffer, bufferSize);
          });

  nanobind::class_<PersistJITEngine>(
      mod, "reuse_compiler_artifacts",
      "Within this context, CUDAQ will blindly reuse compiled objects."
      "It is up to the user to ensure that there are never two distinct"
      "computations launched within a single context.")
      .def(nanobind::init<>())
      .def("__enter__",
           [](PersistJITEngine &ctx) -> void {
             cudaq::compiler_artifact::enablePersistentJITEngine();
           })
      .def(
          "__exit__",
          [](PersistJITEngine &ctx, nanobind::object type,
             nanobind::object value, nanobind::object traceback) {
            cudaq::compiler_artifact::disablePersistentJITEngine();
          },
          nanobind::arg("type").none(), nanobind::arg("value").none(),
          nanobind::arg("traceback").none());
}
} // namespace cudaq
