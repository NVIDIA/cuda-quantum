/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "llvm/ADT/StringSwitch.h"

#include "common/Logger.h"
#include "cudaq/qis/managers/BasicExecutionManager.h"
#include "cudaq/qis/qudit.h"
#include "cudaq/spin_op.h"
#include "cudaq/utils/cudaq_utils.h"
#include <complex>
#include <cstring>
#include <functional>
#include <map>
#include <queue>
#include <sstream>
#include <stack>

#include "nvqir/CircuitSimulator.h"

namespace nvqir {
CircuitSimulator *getCircuitSimulatorInternal();
}

namespace {

/// The DefaultExecutionManager will implement allocation, deallocation, and
/// quantum instruction application via calls to the current CircuitSimulator
class DefaultExecutionManager : public cudaq::BasicExecutionManager {

private:
  nvqir::CircuitSimulator *simulator() {
    return nvqir::getCircuitSimulatorInternal();
  }

protected:
  void handleExecutionContextChanged() override {
    simulator()->setExecutionContext(executionContext);
  }

  void handleExecutionContextEnded() override {
    simulator()->resetExecutionContext();
  }

  void executeInstruction(const Instruction &instruction) override {
    // Get the data, create the Qubit* targets
    auto [gateName, parameters, controls, targets, op] = instruction;

    // Map the Qudits to Qubits
    std::vector<std::size_t> localT;
    std::transform(targets.begin(), targets.end(), std::back_inserter(localT),
                   [](auto &&el) { return el.id; });
    std::vector<std::size_t> localC;
    std::transform(controls.begin(), controls.end(), std::back_inserter(localC),
                   [](auto &&el) { return el.id; });

    // Apply the gate
    llvm::StringSwitch<std::function<void()>>(gateName)
        .Case("h", [&]() { simulator()->h(localC, localT[0]); })
        .Case("x", [&]() { simulator()->x(localC, localT[0]); })
        .Case("y", [&]() { simulator()->y(localC, localT[0]); })
        .Case("z", [&]() { simulator()->z(localC, localT[0]); })
        .Case("rx",
              [&]() { simulator()->rx(parameters[0], localC, localT[0]); })
        .Case("ry",
              [&]() { simulator()->ry(parameters[0], localC, localT[0]); })
        .Case("rz",
              [&]() { simulator()->rz(parameters[0], localC, localT[0]); })
        .Case("s", [&]() { simulator()->s(localC, localT[0]); })
        .Case("t", [&]() { simulator()->t(localC, localT[0]); })
        .Case("sdg", [&]() { simulator()->sdg(localC, localT[0]); })
        .Case("tdg", [&]() { simulator()->tdg(localC, localT[0]); })
        .Case("r1",
              [&]() { simulator()->r1(parameters[0], localC, localT[0]); })
        .Case("u1",
              [&]() { simulator()->u1(parameters[0], localC, localT[0]); })
        .Case("u3",
              [&]() {
                simulator()->u3(parameters[0], parameters[1], parameters[2],
                                localC, localT[0]);
              })
        .Case("swap",
              [&]() { simulator()->swap(localC, localT[0], localT[1]); })
        .Case("exp_pauli",
              [&]() {
                simulator()->applyExpPauli(parameters[0], localC, localT, op);
              })
        .Default([&]() {
          throw std::runtime_error("[DefaultExecutionManager] invalid gate "
                                   "application requested " +
                                   gateName + ".");
        })();
  }

public:
  DefaultExecutionManager() {
    cudaq::info("[DefaultExecutionManager] Creating the {} backend.",
                simulator()->name());
  }
  virtual ~DefaultExecutionManager() = default;

  std::size_t allocateQudit(std::size_t n_levels) override {
    return simulator()->allocateQudit();
  }

  void deallocateQudit(const cudaq::QuditInfo &q) override {
    simulator()->deallocateQudit(q.id);
  }

  int measure(const cudaq::QuditInfo &q) override {
    return simulator()->mz(q.id);
  }

  cudaq::SpinMeasureResult measure(const cudaq::spin_op &op) override {
    return simulator()->measure(op);
  }

  void reset(const cudaq::QuditInfo &q) override {
    simulator()->resetQubit(q.id);
  }
};

} // namespace

CUDAQ_REGISTER_EXECUTION_MANAGER(DefaultExecutionManager)
