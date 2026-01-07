/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/Logger.h"
#include "cudaq/operators.h"
#include "cudaq/qis/managers/BasicExecutionManager.h"
#include "cudaq/qis/qudit.h"
#include "cudaq/utils/cudaq_utils.h"
#include "nvqir/CircuitSimulator.h"
#include "llvm/ADT/StringSwitch.h"
#include <span>

namespace nvqir {
CircuitSimulator *getCircuitSimulatorInternal();
}

namespace cudaq {

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
    // want to allocate and set state.
    // There could be previous 'default' allocations whereby we just cached them
    // in requestedAllocations.
    // These default allocations need to be dispatched separately.
    if (!requestedAllocations.empty() &&
        targets.size() != requestedAllocations.size()) {
      assert(targets.size() < requestedAllocations.size());
      // This assumes no qubit reuse, aka the qubits are allocated in order.
      // This is consistent with the Kronecker product assumption in
      // CircuitSimulator.
      for (std::size_t i = 0; i < requestedAllocations.size() - 1; ++i) {
        // Verify this assumption to make sure the simulator set
        // the state of appropriate qubits.
        const auto &thisAlloc = requestedAllocations[i];
        const auto &nextAlloc = requestedAllocations[i + 1];
        if (nextAlloc.id != (thisAlloc.id + 1)) {
          std::stringstream errorMsg;
          errorMsg << "Out of order allocation detected. This is not supported "
                      "by simulator backends. Qubit allocations: [ ";
          for (const auto &alloc : requestedAllocations) {
            errorMsg << alloc.id << " ";
          }
          errorMsg << "]";
          throw std::logic_error(errorMsg.str());
        }
      }
      const auto numDefaultAllocs =
          requestedAllocations.size() - targets.size();
      simulator()->allocateQubits(numDefaultAllocs);
      // The targets will be allocated in a specific state.
      simulator()->allocateQubits(targets.size(), state, precision);
    } else {
      simulator()->allocateQubits(requestedAllocations.size(), state,
                                  precision);
    }
    requestedAllocations.clear();
  }

  void initializeState(const std::vector<cudaq::QuditInfo> &targets,
                       const cudaq::SimulationState *state) override {
    // Note: a void* ptr doesn't provide enough info to the simulators, hence
    // need a dedicated code path.
    // TODO: simplify/combine the two code paths (raw vector and state).
    if (!requestedAllocations.empty() &&
        targets.size() != requestedAllocations.size()) {
      assert(targets.size() < requestedAllocations.size());
      const auto numDefaultAllocs =
          requestedAllocations.size() - targets.size();
      simulator()->allocateQubits(numDefaultAllocs);
      // The targets will be allocated in a specific state.
      simulator()->allocateQubits(targets.size(), state);
    } else {
      simulator()->allocateQubits(requestedAllocations.size(), state);
    }
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
    if (!requestedAllocations.empty()) {
      CUDAQ_INFO("[DefaultExecutionManager] Flushing remaining {} allocations "
                 "at handleExecutionContextEnded.",
                 requestedAllocations.size());
      // If there are pending allocations, flush them to the simulator.
      // Making sure the simulator's state is consistent with the number of
      // allocations even though the circuit might be empty.
      simulator()->allocateQubits(requestedAllocations.size());
      requestedAllocations.clear();
    }
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
          if (cudaq::customOpRegistry::getInstance().isOperationRegistered(
                  gateName)) {
            const auto &op =
                cudaq::customOpRegistry::getInstance().getOperation(gateName);
            auto data = op.unitary(parameters);
            simulator()->applyCustomOperation(data, localC, localT, gateName);
            return;
          }
          throw std::runtime_error("[DefaultExecutionManager] invalid gate "
                                   "application requested " +
                                   gateName + ".");
        })();
  }

  void applyNoise(const kraus_channel &channel,
                  const std::vector<QuditInfo> &targets) override {
    if (isInTracerMode())
      return;

    flushGateQueue();

    if (channel.empty())
      if (!simulator()->isValidNoiseChannel(channel.noise_type))
        throw std::runtime_error("this is not a valid kraus channel name (" +
                                 channel.get_type_name() +
                                 "), no "
                                 "kraus ops available to construct it.");

    std::vector<std::size_t> localT;
    std::transform(targets.begin(), targets.end(), std::back_inserter(localT),
                   [](auto &&el) { return el.id; });
    CUDAQ_INFO(
        "[DefaultExecutionManager] Applying fine-grain kraus channel {}.",
        channel.get_type_name());
    simulator()->applyNoise(channel, localT);
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
    simulator()->measureSpinOp(op);
  }

public:
  DefaultExecutionManager() {
    CUDAQ_INFO("[DefaultExecutionManager] Creating the {} backend.",
               simulator()->name());
  }
  virtual ~DefaultExecutionManager() = default;

  void resetQudit(const cudaq::QuditInfo &q) override {
    flushRequestedAllocations();
    simulator()->resetQubit(q.id);
  }
};

} // namespace cudaq

CUDAQ_REGISTER_EXECUTION_MANAGER(DefaultExecutionManager, default)

extern "C" {
/// C interface to the (default) execution manager's methods.
///
/// This supplies an interface to allocate and deallocate qubits, reset a
/// qubit, measure a qubit, and apply the gates defined by CUDA-Q.

std::int64_t __nvqpp__cudaq_em_allocate() {
  return cudaq::getExecutionManager()->allocateQudit();
}

void __nvqpp__cudaq_em_apply(const char *gateName, std::int64_t numParams,
                             const double *params,
                             const std::span<std::size_t> &ctrls,
                             const std::span<std::size_t> &targets,
                             bool isAdjoint) {
  std::vector<double> pv{params, params + numParams};
  auto fromSpan = [&](const std::span<std::size_t> &qubitSpan)
      -> std::vector<cudaq::QuditInfo> {
    std::vector<cudaq::QuditInfo> result;
    for (std::size_t qb : qubitSpan)
      result.emplace_back(2u, qb);
    return result;
  };
  std::vector<cudaq::QuditInfo> cv = fromSpan(ctrls);
  std::vector<cudaq::QuditInfo> tv = fromSpan(targets);
  cudaq::getExecutionManager()->apply(gateName, pv, cv, tv, isAdjoint);
}

std::int32_t __nvqpp__cudaq_em_measure(const std::span<std::size_t> &targets,
                                       const char *tagName) {
  cudaq::QuditInfo qubit{2u, targets[0]};
  std::string tag{tagName};
  return cudaq::getExecutionManager()->measure(qubit, tag);
}

void __nvqpp__cudaq_em_reset(const std::span<std::size_t> &targets) {
  cudaq::QuditInfo qubit{2u, targets[0]};
  cudaq::getExecutionManager()->reset(qubit);
}

void __nvqpp__cudaq_em_return(const std::span<std::size_t> &targets) {
  cudaq::QuditInfo qubit{2u, targets[0]};
  cudaq::getExecutionManager()->returnQudit(qubit);
}
}
