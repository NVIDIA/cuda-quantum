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
#include "nvqir/CircuitSimulator.h"
#include "cudaq/algorithms/run/policy.h"
#include "cudaq/algorithms/sample/policy.h"
#include "cudaq/nvqir_launch.h"
#include "cudaq/platform.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nvqir {
void toggleDynamicQubitManagement();
} // namespace nvqir

namespace cudaq {

void bindTestUtils(nanobind::module_ &mod, LinkedLibraryHolder &holder) {
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
      nanobind::arg("numQubits"));

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

  // Run a mock-QPU kernel under a sampling policy and return the sample_result
  // directly. This mirrors the C++ emulation path in BaseRemoteRESTQPU (via
  // nvqir::launch), so the result flows through the policy return value rather
  // than the execution context. Qubits are preallocated because mock bitcode
  // references qubits by static index without allocating them itself.
  testingSubmodule.def(
      "sampleKernel",
      [&](std::size_t numQubits, std::size_t numShots,
          std::function<void()> kernel) {
        auto simName = holder.getTarget().simulatorName;
        auto *sim = holder.getSimulator(simName);
        nvqir::toggleDynamicQubitManagement();
        auto qubits = sim->allocateQubits(numQubits);
        cudaq::sample_policy policy;
        policy.options.shots = numShots;
        auto result = nvqir::launch(policy, {}, [&]() { kernel(); });
        nvqir::toggleDynamicQubitManagement();
        sim->deallocateQubits(qubits);
        return result;
      },
      nanobind::arg("numQubits"), nanobind::arg("numShots"),
      nanobind::arg("kernel"));

  // Run a mock-QPU kernel once under a run policy and return the raw QIR output
  // log for that shot (from run_result), rather than reading the simulator's
  // output log out of band. Callers loop this once per shot.
  testingSubmodule.def(
      "runKernel",
      [&](std::size_t numQubits, std::function<void()> kernel) {
        auto simName = holder.getTarget().simulatorName;
        auto *sim = holder.getSimulator(simName);
        nvqir::toggleDynamicQubitManagement();
        auto qubits = sim->allocateQubits(numQubits);
        cudaq::run_policy policy;
        policy.shots = 1;
        auto result = nvqir::launch(policy, {}, [&]() { kernel(); });
        nvqir::toggleDynamicQubitManagement();
        sim->deallocateQubits(qubits);
        return result.outputLog;
      },
      nanobind::arg("numQubits"), nanobind::arg("kernel"));
}

} // namespace cudaq
