/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "TestingUtils.h"
#include "common/PluginUtils.h"
#include "cudaq/platform.h"
#include "nvqir/CircuitSimulator.h"
#include <fstream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <regex>
#include <sstream>

namespace py = pybind11;

namespace nvqir {
void toggleBaseProfile();
} // namespace nvqir

namespace cudaq {

void bindTestUtils(py::module &mod, LinkedLibraryHolder &holder) {
  auto testingSubmodule = mod.def_submodule("testing");

  // Vision for all this
  // 
  // cudaq.testing.toggleBaseProfile()
  // qubits, context = cudaq.testing.initialize(numQubits, 1000)
  // .. use llvmlite.jit to execute kernel function
  // results = cudaq.testing.finalize(qubits, context);
  // results.dump()

  testingSubmodule.def(
      "toggleBaseProfile",
      [&]() {
        cudaq::info("Calling initialize_cudaq.");
        nvqir::toggleBaseProfile();
      },
      "");

  testingSubmodule.def(
      "initialize", [&](std::size_t numQubits, std::size_t numShots) {
        cudaq::ExecutionContext *context =
            new cudaq::ExecutionContext("sample", numShots);
        holder.getSimulator("qpp")->setExecutionContext(context);
        return std::make_tuple(
            holder.getSimulator("qpp")->allocateQubits(numQubits), context);
      });

  testingSubmodule.def("finalize", [&](const std::vector<std::size_t> &qubits,
                                       cudaq::ExecutionContext *context) {
    holder.getSimulator("qpp")->deallocateQubits(qubits);
    holder.getSimulator("qpp")->resetExecutionContext();
    auto result = context->result;
    delete context;
    return result;
  });
}

} // namespace cudaq
