/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qis/execution_manager.h"
#include <fmt/core.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace cudaq {

void bindExecutionManager(nb::module_ &mod) {

  mod.def(
      "applyQuantumOperation",
      [](const std::string &name, std::vector<double> &params,
         std::vector<std::size_t> &controls, std::vector<std::size_t> &targets,
         bool isAdjoint, cudaq::spin_op_term &op) {
        std::vector<cudaq::QuditInfo> c, t;
        std::transform(controls.begin(), controls.end(), std::back_inserter(c),
                       [](auto &&el) { return cudaq::QuditInfo(2, el); });
        std::transform(targets.begin(), targets.end(), std::back_inserter(t),
                       [](auto &&el) { return cudaq::QuditInfo(2, el); });
        cudaq::getExecutionManager()->apply(name, params, c, t, isAdjoint, op);
      },
      nb::arg("name"), nb::arg("params"), nb::arg("controls"),
      nb::arg("targets"), nb::arg("isAdjoint") = false,
      nb::arg("op") = cudaq::spin_op::identity());

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
      nb::arg("qubit"), nb::arg("register_name") = "");
}
} // namespace cudaq
