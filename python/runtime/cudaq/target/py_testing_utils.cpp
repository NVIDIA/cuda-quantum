/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_testing_utils.h"
#include "LinkedLibraryHolder.h"
#include "cudaq.h"
#include "cudaq/platform.h"
#include "nvqir/CircuitSimulator.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

namespace nvqir {
void toggleDynamicQubitManagement();
} // namespace nvqir

namespace cudaq {

void bindTestUtils(py::module &mod, LinkedLibraryHolder &holder) {
  auto testingSubmodule = mod.def_submodule("testing");

  testingSubmodule.def(
      "toggleDynamicQubitManagement",
      [&]() { nvqir::toggleDynamicQubitManagement(); }, "");

  testingSubmodule.def(
      "allocateQubits",
      [&](std::size_t numQubits) {
        auto simName = holder.getTarget().simulatorName;
        return holder.getSimulator(simName)->allocateQubits(numQubits);
      },
      py::arg("numQubits"));

  testingSubmodule.def("deallocateQubits",
                       [&](const std::vector<std::size_t> &qubits) {
                         auto simName = holder.getTarget().simulatorName;
                         holder.getSimulator(simName)->deallocateQubits(qubits);
                       });

  testingSubmodule.def("getAndClearOutputLog", [&]() {
    auto simName = holder.getTarget().simulatorName;
    auto log = holder.getSimulator(simName)->outputLog;
    holder.getSimulator(simName)->outputLog.clear();
    return log;
  });
}

} // namespace cudaq
