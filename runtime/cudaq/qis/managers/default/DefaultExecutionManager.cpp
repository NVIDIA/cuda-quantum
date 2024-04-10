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
  /// @brief To improve `qudit` allocation, we defer
  /// single `qudit` allocation requests until the first
  /// encountered `apply` call.
  std::vector<cudaq::QuditInfo> requestedAllocations;

  /// @brief Allocate all requested `qudits`.
  void flushRequestedAllocations() {
    if (requestedAllocations.empty())
      return;

    allocateQudits(requestedAllocations);
    requestedAllocations.clear();
  }

protected:
  void allocateQudit(const cudaq::QuditInfo &q) override {
    requestedAllocations.emplace_back(2, q.id);
  }

  void allocateQudits(const std::vector<cudaq::QuditInfo> &qudits) override {
    simulator()->allocateQubits(qudits.size());
  }

  void initializeState(const std::vector<cudaq::QuditInfo> &targets,
                       const void *state,
                       cudaq::simulation_precision precision) override {
    // Here we have qubits in requestedAllocations
    // want to allocate and set state
    simulator()->allocateQubits(requestedAllocations.size(), state, precision);
    requestedAllocations.clear();
  }

  void deallocateQudit(const cudaq::QuditInfo &q) override {

    // Before trying to deallocate, make sure the qudit hasn't
    // been requested but not allocated.
    auto iter =
        std::find(requestedAllocations.begin(), requestedAllocations.end(), q);
    if (iter != requestedAllocations.end()) {
      requestedAllocations.erase(iter);
      return;
    }

    simulator()->deallocate(q.id);
  }

  void deallocateQudits(const std::vector<cudaq::QuditInfo> &qudits) override {
    std::vector<std::size_t> local;
    for (auto &q : qudits) {
      auto iter = std::find(requestedAllocations.begin(),
                            requestedAllocations.end(), q);
      if (iter != requestedAllocations.end()) {
        requestedAllocations.erase(iter);
      } else {
        local.push_back(q.id);
      }
    }

    simulator()->deallocateQubits(local);
  }

  void handleExecutionContextChanged() override {
    requestedAllocations.clear();
    simulator()->setExecutionContext(executionContext);
  }

  void handleExecutionContextEnded() override {
    simulator()->resetExecutionContext();
  }

  void executeInstruction(const Instruction &instruction) override {
    flushRequestedAllocations();

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

  int measureQudit(const cudaq::QuditInfo &q,
                   const std::string &registerName) override {
    flushRequestedAllocations();
    return simulator()->mz(q.id, registerName);
  }

  void flushGateQueue() override {
    synchronize();
    flushRequestedAllocations();
    simulator()->flushGateQueue();
  }

  void measureSpinOp(const cudaq::spin_op &op) override {
    flushRequestedAllocations();
    simulator()->flushGateQueue();

    if (executionContext->canHandleObserve) {
      auto result = simulator()->observe(*executionContext->spin.value());
      executionContext->expectationValue = result.expectation();
      executionContext->result = result.raw_data();
      return;
    }

    assert(op.num_terms() == 1 && "Number of terms is not 1.");

    cudaq::info("Measure {}", op.to_string(false));
    std::vector<std::size_t> qubitsToMeasure;
    std::vector<std::function<void(bool)>> basisChange;
    op.for_each_pauli([&](cudaq::pauli type, std::size_t qubitIdx) {
      if (type != cudaq::pauli::I)
        qubitsToMeasure.push_back(qubitIdx);

      if (type == cudaq::pauli::Y)
        basisChange.emplace_back([&, qubitIdx](bool reverse) {
          simulator()->rx(!reverse ? M_PI_2 : -M_PI_2, qubitIdx);
        });
      else if (type == cudaq::pauli::X)
        basisChange.emplace_back(
            [&, qubitIdx](bool) { simulator()->h(qubitIdx); });
    });

    // Change basis, flush the queue
    if (!basisChange.empty()) {
      for (auto &basis : basisChange)
        basis(false);

      simulator()->flushGateQueue();
    }

    // Get whether this is shots-based
    int shots = 0;
    if (executionContext->shots > 0)
      shots = executionContext->shots;

    // Sample and give the data to the context
    cudaq::ExecutionResult result = simulator()->sample(qubitsToMeasure, shots);
    executionContext->expectationValue = result.expectationValue;
    executionContext->result = cudaq::sample_result(result);

    // Restore the state.
    if (!basisChange.empty()) {
      std::reverse(basisChange.begin(), basisChange.end());
      for (auto &basis : basisChange)
        basis(true);

      simulator()->flushGateQueue();
    }
  }

public:
  DefaultExecutionManager() {
    cudaq::info("[DefaultExecutionManager] Creating the {} backend.",
                simulator()->name());
  }
  virtual ~DefaultExecutionManager() = default;

  void resetQudit(const cudaq::QuditInfo &q) override {
    flushRequestedAllocations();
    simulator()->resetQubit(q.id);
  }
};

} // namespace

CUDAQ_REGISTER_EXECUTION_MANAGER(DefaultExecutionManager)
