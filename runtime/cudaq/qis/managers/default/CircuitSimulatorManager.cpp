/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "common/Logger.h"
#include "common/PluginUtils.h"

#include "common/FmtCore.h"
#include "cudaq/qis/managers/BasicExecutionManager.h"
#include "cudaq/qis/qudit.h"
#include "cudaq/spin_op.h"
#include "cudaq/utils/cudaq_utils.h"
#include "nvqir/CircuitSimulator.h"
#include "llvm/ADT/StringSwitch.h"
#include <complex>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <queue>
#include <sstream>
#include <stack>

using namespace nvqir;

namespace cudaq {
thread_local nvqir::CircuitSimulator *simulator;
inline static constexpr std::string_view GetCircuitSimulatorSymbol =
    "getCircuitSimulator";

struct ExternallyProvidedSimGenerator {
  nvqir::CircuitSimulator *simulator;
  ExternallyProvidedSimGenerator(nvqir::CircuitSimulator *sim)
      : simulator(sim) {}
  auto operator()() { return simulator->clone(); }
};

static std::unique_ptr<ExternallyProvidedSimGenerator> externSimGenerator;

void setCircuitSimulator(nvqir::CircuitSimulator *sim) {
  simulator = sim;
  if (externSimGenerator) {
    auto ptr = externSimGenerator.release();
    delete ptr;
  }
  externSimGenerator = std::make_unique<ExternallyProvidedSimGenerator>(sim);
  cudaq::info("[runtime] Setting the circuit simulator to {}.", sim->name());
}

CircuitSimulator *getCircuitSimulatorInternal() {
  if (simulator)
    return simulator;

  if (externSimGenerator) {
    simulator = (*externSimGenerator)();
    return simulator;
  }

  simulator = cudaq::getUniquePluginInstance<CircuitSimulator>(
      GetCircuitSimulatorSymbol);
  cudaq::info("Creating the {} backend.", simulator->name());
  return simulator;
}

} // namespace cudaq
namespace {

/// @brief
class CircuitSimulatorManager : public cudaq::BasicExecutionManager {
private:
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
    cudaq::simulator->allocateQubits(qudits.size());
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

    cudaq::simulator->deallocate(q.id);
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

    cudaq::simulator->deallocateQubits(local);
  }

  void handleExecutionContextChanged() override {
    cudaq::simulator->setExecutionContext(executionContext);
  }

  void handleExecutionContextEnded() override {
    cudaq::simulator->resetExecutionContext();
  }

  void executeInstruction(const Instruction &instruction) override {
    flushRequestedAllocations();

    // Get the data, create the Qubit* targets
    auto [gateName, parameters, controls, targets] = instruction;

    // Map the Qudits to Qubits
    std::vector<std::size_t> localT;
    std::transform(targets.begin(), targets.end(), std::back_inserter(localT),
                   [](auto &&el) { return el.id; });
    std::vector<std::size_t> localC;
    std::transform(controls.begin(), controls.end(), std::back_inserter(localC),
                   [](auto &&el) { return el.id; });

    // FIXME Could probably get rid of this with matrices
    auto functor =
        llvm::StringSwitch<std::function<void()>>(gateName)
            .Case("h", [&]() { cudaq::simulator->h(localC, localT[0]); })
            .Case("x", [&]() { cudaq::simulator->x(localC, localT[0]); })
            .Case("y", [&]() { cudaq::simulator->y(localC, localT[0]); })
            .Case("z", [&]() { cudaq::simulator->z(localC, localT[0]); })
            .Case("rx",
                  [&]() {
                    cudaq::simulator->rx(parameters[0], localC, localT[0]);
                  })
            .Case("ry",
                  [&]() {
                    cudaq::simulator->ry(parameters[0], localC, localT[0]);
                  })
            .Case("rz",
                  [&]() {
                    cudaq::simulator->rz(parameters[0], localC, localT[0]);
                  })
            .Case("s", [&]() { cudaq::simulator->s(localC, localT[0]); })
            .Case("t", [&]() { cudaq::simulator->t(localC, localT[0]); })
            .Case("sdg", [&]() { cudaq::simulator->sdg(localC, localT[0]); })
            .Case("tdg", [&]() { cudaq::simulator->tdg(localC, localT[0]); })
            .Case("r1",
                  [&]() {
                    cudaq::simulator->r1(parameters[0], localC, localT[0]);
                  })
            // .Case("u1", [&]() { cudaq::simulator->tdg(localC, localT[0]); })
            // .Case("u3", [&]() { cudaq::simulator->tdg(localC, localT[0]); })
            .Case(
                "swap",
                [&]() { cudaq::simulator->swap(localC, localT[0], localT[1]); })
            .Default([&]() {
              throw std::runtime_error("[CircuitSimulatorManager] invalid gate "
                                       "application requested " +
                                       gateName + ".");
            });

    functor();
  }

  int measureQudit(const cudaq::QuditInfo &q) override {
    return cudaq::simulator->mz(q.id);
  }

  void measureSpinOp(const cudaq::spin_op &op) override {
    cudaq::simulator->flushGateQueue();

    if (executionContext->canHandleObserve) {
      auto result = cudaq::simulator->observe(*executionContext->spin.value());
      executionContext->expectationValue = result.expectationValue;
      executionContext->result = cudaq::sample_result(result);
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
          cudaq::simulator->rx(!reverse ? M_PI_2 : -M_PI_2, qubitIdx);
        });
      else if (type == cudaq::pauli::X)
        basisChange.emplace_back(
            [&, qubitIdx](bool) { cudaq::simulator->h(qubitIdx); });
    });

    // Change basis, flush the queue
    if (!basisChange.empty()) {
      for (auto &basis : basisChange)
        basis(false);

      cudaq::simulator->flushGateQueue();
    }

    // Get whether this is shots-based
    int shots = 0;
    if (executionContext->shots > 0)
      shots = executionContext->shots;

    // Sample and give the data to the context
    cudaq::ExecutionResult result =
        cudaq::simulator->sample(qubitsToMeasure, shots);
    executionContext->expectationValue = result.expectationValue;
    executionContext->result = cudaq::sample_result(result);

    // Restore the state.
    if (!basisChange.empty()) {
      std::reverse(basisChange.begin(), basisChange.end());
      for (auto &basis : basisChange)
        basis(true);

      cudaq::simulator->flushGateQueue();
    }

    return;
  }

public:
  CircuitSimulatorManager() {
    // Get the default linked circuit simulator
    cudaq::getCircuitSimulatorInternal();
    cudaq::info("[CircuitSimulatorManager] Creating the {} backend.",
                cudaq::simulator->name());
  }

  ~CircuitSimulatorManager() = default;

  /// @brief Reset the qubit state
  void resetQudit(const cudaq::QuditInfo &qudit) override {
    cudaq::simulator->resetQubit(qudit.id);
  }
};

} // namespace

CUDAQ_REGISTER_EXECUTION_MANAGER(CircuitSimulatorManager)
