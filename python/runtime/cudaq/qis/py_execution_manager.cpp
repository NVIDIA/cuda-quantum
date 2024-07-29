/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qis/execution_manager.h"
#include <fmt/core.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace cudaq {

void bindExecutionManager(py::module &mod) {

  mod.def(
      "applyQuantumOperation",
      [](const std::string &name, std::vector<double> &params,
         std::vector<std::size_t> &controls, std::vector<std::size_t> &targets,
         bool isAdjoint, cudaq::spin_op &op) {
        std::vector<cudaq::QuditInfo> c, t;
        std::transform(controls.begin(), controls.end(), std::back_inserter(c),
                       [](auto &&el) { return cudaq::QuditInfo(2, el); });
        std::transform(targets.begin(), targets.end(), std::back_inserter(t),
                       [](auto &&el) { return cudaq::QuditInfo(2, el); });
        cudaq::getExecutionManager()->apply(name, params, c, t, isAdjoint, op);
      },
      py::arg("name"), py::arg("params"), py::arg("controls"),
      py::arg("targets"), py::arg("isAdjoint") = false,
      py::arg("op") = cudaq::spin_op());

  mod.def("startAdjointRegion",
          []() { cudaq::getExecutionManager()->startAdjointRegion(); });
  mod.def("endAdjointRegion",
          []() { cudaq::getExecutionManager()->endAdjointRegion(); });

  mod.def("startCtrlRegion", [](std::vector<std::size_t> &controls) {
    cudaq::getExecutionManager()->startCtrlRegion(controls);
  });
  mod.def("endCtrlRegion", [](std::size_t nControls) {
    cudaq::getExecutionManager()->endCtrlRegion(nControls);
  });
  mod.def(
      "measure",
      [](std::size_t id, const std::string &regName) {
        return cudaq::getExecutionManager()->measure(cudaq::QuditInfo(2, id),
                                                     regName);
      },
      py::arg("qubit"), py::arg("register_name") = "");
}
} // namespace cudaq
