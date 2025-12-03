/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/Trace.h"
#include "cudaq/qis/execution_manager.h"
#include "runtime/cudaq/platform/py_alt_launch_kernel.h"
#include "utils/LinkedLibraryHolder.h"
#include "utils/OpaqueArguments.h"

#include <algorithm>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <iterator>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace cudaq {
void bindPyTraceKernel(py::module &mod) {

  py::class_<QuditInfo>(mod, "QuditInfo")
      .def(py::init<std::size_t, std::size_t>())
      .def_readonly("levels", &QuditInfo::levels)
      .def_readonly("id", &QuditInfo::id)
      .def("__repr__", [](const QuditInfo &q) {
        return fmt::format("QuditInfo(levels={}, id={})", q.levels, q.id);
      });

  py::class_<Trace::Instruction>(mod, "TraceInstruction")
      .def(py::init<std::string_view, std::vector<double>,
                    std::vector<QuditInfo>, std::vector<QuditInfo>>())
      .def_readonly("name", &Trace::Instruction::name)
      .def_readonly("params", &Trace::Instruction::params)
      .def_readonly("controls", &Trace::Instruction::controls)
      .def_readonly("targets", &Trace::Instruction::targets)
      .def("__repr__", [](const Trace::Instruction &i) {
        std::vector<std::size_t> controlIds;
        std::transform(i.controls.begin(), i.controls.end(),
                       std::back_inserter(controlIds),
                       [](const QuditInfo &q) { return q.id; });
        std::vector<std::size_t> targetIds;
        std::transform(i.targets.begin(), i.targets.end(),
                       std::back_inserter(targetIds),
                       [](const QuditInfo &q) { return q.id; });
        return fmt::format(
            "Instruction(name={}, params=[{}], controls=[{}], targets=[{}])",
            i.name, fmt::join(i.params, ", "), fmt::join(controlIds, ", "),
            fmt::join(targetIds, ", "));
      });

  py::class_<Trace>(mod, "Trace")
      .def(py::init<>())
      .def("append_instruction", &Trace::appendInstruction)
      .def("get_num_qudits", &Trace::getNumQudits)
      .def(
          "__iter__",
          [](const Trace &t) { return py::make_iterator(t.begin(), t.end()); },
          py::keep_alive<0, 1>());

  mod.def(
      "trace_kernel",
      [&](py::object kernel, py::args args) {
        if (py::hasattr(kernel, "compile"))
          kernel.attr("compile")();
        auto &platform = cudaq::get_platform();
        auto kernelName = kernel.attr("name").cast<std::string>();
        auto kernelMod = kernel.attr("module").cast<MlirModule>();
        args = simplifiedValidateInputArguments(args);
        std::unique_ptr<OpaqueArguments> argData(
            toOpaqueArgs(args, kernelMod, kernelName));

        auto ctx = std::make_unique<ExecutionContext>("tracer", 1);
        ctx->kernelName = kernelName;
        platform.set_exec_ctx(ctx.get());
        pyAltLaunchKernel(kernelName, kernelMod, *argData, {});
        platform.reset_exec_ctx();

        return ctx->kernelTrace;
      },
      py::arg("kernel"), py::kw_only(),
      R"#(Executes the given kernel and returns a :class:`Trace` object 
representing the sequence of operations performed.

Args:
  kernel (:class:`Kernel`): The :class:`Kernel` to trace.
  *arguments (Optional[Any]): The concrete values to evaluate the kernel 
    function at. Leave empty if the kernel doesn't accept any arguments.

Returns:
  :class:`Trace`: The trace of the kernel execution.)#");
}
} // namespace cudaq
