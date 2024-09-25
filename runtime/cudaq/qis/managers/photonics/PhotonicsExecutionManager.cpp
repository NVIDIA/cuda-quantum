/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "common/Logger.h"
// #include "common/PluginUtils.h"
#include "cudaq/qis/managers/BasicExecutionManager.h"
#include "cudaq/qis/qudit.h"
// #include "cudaq/spin_op.h"
#include "cudaq/utils/cudaq_utils.h"
#include "nvqir/photonics/PhotonicCircuitSimulator.h"

#include "llvm/ADT/StringSwitch.h"
#include <cmath>
#include <complex>
#include <cstring>
#include <functional>
#include <iostream>
#include <sstream>

namespace nvqir {
PhotonicCircuitSimulator *getPhotonicCircuitSimulatorInternal();
}

namespace cudaq {

struct PhotonicsState : public cudaq::SimulationState {
  /// @brief The state. This class takes ownership move semantics.
  qpp::ket state;

  /// @brief The qudit-levels (`qumodes`)
  std::size_t levels;

  PhotonicsState(qpp::ket &&data, std::size_t lvl)
      : state(std::move(data)), levels(lvl) {}

  /// TODO: Rename the API to be generic
  std::size_t getNumQubits() const override {
    return (std::log2(state.size()) / std::log2(levels));
  }

  std::complex<double> overlap(const cudaq::SimulationState &other) override {
    throw "not supported for this photonics simulator";
  }

  std::complex<double>
  getAmplitude(const std::vector<int> &basisState) override {
    if (getNumQubits() != basisState.size())
      throw std::runtime_error(fmt::format(
          "[photonics] getAmplitude with an invalid number of bits in the "
          "basis state: expected {}, provided {}.",
          getNumQubits(), basisState.size()));

    // Convert the basis state to an index value
    const std::size_t idx = std::accumulate(
        std::make_reverse_iterator(basisState.end()),
        std::make_reverse_iterator(basisState.begin()), 0ull,
        [&](std::size_t acc, int bit) { return (acc * levels) + bit; });
    return state[idx];
  }

  Tensor getTensor(std::size_t tensorIdx = 0) const override {
    if (tensorIdx != 0)
      throw std::runtime_error("[photonics] invalid tensor requested.");
    return Tensor{
        reinterpret_cast<void *>(
            const_cast<std::complex<double> *>(state.data())),
        std::vector<std::size_t>{static_cast<std::size_t>(state.size())},
        getPrecision()};
  }

  /// @brief Return all tensors that represent this state
  std::vector<Tensor> getTensors() const override { return {getTensor()}; }

  /// @brief Return the number of tensors that represent this state.
  std::size_t getNumTensors() const override { return 1; }

  std::complex<double>
  operator()(std::size_t tensorIdx,
             const std::vector<std::size_t> &indices) override {
    if (tensorIdx != 0)
      throw std::runtime_error("[photonics] invalid tensor requested.");
    if (indices.size() != 1)
      throw std::runtime_error("[photonics] invalid element extraction.");
    return state[indices[0]];
  }

  std::unique_ptr<SimulationState>
  createFromSizeAndPtr(std::size_t size, void *ptr, std::size_t) override {
    throw "not supported for this photonics simulator";
  }

  void dump(std::ostream &os) const override { os << state << "\n"; }

  precision getPrecision() const override {
    return cudaq::SimulationState::precision::fp64;
  }

  void destroyState() override {
    qpp::ket k;
    state = k;
  }
};

/// @brief The `PhotonicsExecutionManager` implements allocation, deallocation,
/// and quantum instruction application for the photonics execution manager.
class PhotonicsExecutionManager : public BasicExecutionManager {
private:
  nvqir::PhotonicCircuitSimulator *photonic_simulator() {
    return nvqir::getPhotonicCircuitSimulatorInternal();
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

  bool isInTracerMode() {
    return executionContext && executionContext->name == "tracer";
  }

protected:
  /// @brief Qudit allocation method: a zeroState is first initialized, the
  /// following ones are added via kron operators
  void allocateQudit(const cudaq::QuditInfo &q) override {
    requestedAllocations.emplace_back(q.levels, q.id);
  }

  /// @brief Allocate a set of `qudits` with a single call.
  void allocateQudits(const std::vector<cudaq::QuditInfo> &qudits) override {
    photonic_simulator()->setLevels(qudits[0].levels);
    photonic_simulator()->allocateQudits(qudits.size());
    // for (auto &q : qudits)
    //   allocateQudit(q);
  }

  void initializeState(const std::vector<cudaq::QuditInfo> &targets,
                       const void *state,
                       cudaq::simulation_precision precision) override {
    // Here we have qudits in requestedAllocations
    // want to allocate and set state.
    // There could be previous 'default' allocations whereby we just cached them
    // in requestedAllocations.
    // These default allocations need to be dispatched separately.
    if (!requestedAllocations.empty() &&
        targets.size() != requestedAllocations.size()) {
      assert(targets.size() < requestedAllocations.size());
      // This assumes no qudit reuse, aka the qudits are allocated in order.
      // This is consistent with the Kronecker product assumption in
      // CircuitSimulator.
      for (std::size_t i = 0; i < requestedAllocations.size() - 1; ++i) {
        // Verify this assumption to make sure the simulator set
        // the state of appropriate qudits.
        const auto &thisAlloc = requestedAllocations[i];
        const auto &nextAlloc = requestedAllocations[i + 1];
        if (nextAlloc.id != (thisAlloc.id + 1)) {
          std::stringstream errorMsg;
          errorMsg << "Out of order allocation detected. This is not supported "
                      "by simulator backends. Qudit allocations: [ ";
          for (const auto &alloc : requestedAllocations) {
            errorMsg << alloc.id << " ";
          }
          errorMsg << "]";
          throw std::logic_error(errorMsg.str());
        }
      }
      const auto numDefaultAllocs =
          requestedAllocations.size() - targets.size();
      photonic_simulator()->allocateQudits(numDefaultAllocs);
      // The targets will be allocated in a specific state.
      photonic_simulator()->allocateQudits(targets.size(), state, precision);
    } else {
      photonic_simulator()->allocateQudits(requestedAllocations.size(), state,
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
      photonic_simulator()->allocateQudits(numDefaultAllocs);
      // The targets will be allocated in a specific state.
      photonic_simulator()->allocateQudits(targets.size(), state);
    } else {
      photonic_simulator()->allocateQudits(requestedAllocations.size(), state);
    }
    requestedAllocations.clear();
  }

  /// @brief Qudit deallocation method
  void deallocateQudit(const cudaq::QuditInfo &q) override {}

  /// @brief Deallocate a set of `qudits` with a single call.
  void deallocateQudits(const std::vector<cudaq::QuditInfo> &qudits) override {}

  /// @brief Handler for when the photonics execution context changes
  void handleExecutionContextChanged() override {
    requestedAllocations.clear();
    photonic_simulator()->setExecutionContext(executionContext);
  }

  /// @brief Handler for when the current execution context has ended. It
  /// returns samples to the execution context if it is "sample".
  void handleExecutionContextEnded() override {
    if (!requestedAllocations.empty()) {
      cudaq::info(
          "[PhotonicsExecutionManager] Flushing remaining {} allocations "
          "at handleExecutionContextEnded.",
          requestedAllocations.size());
      // If there are pending allocations, flush them to the simulator.
      // Making sure the simulator's state is consistent with the number of
      // allocations even though the circuit might be empty.
      photonic_simulator()->allocateQudits(requestedAllocations.size());
      requestedAllocations.clear();
    }
    photonic_simulator()->resetExecutionContext();
  }

  void executeInstruction(const Instruction &instruction) override {
    flushRequestedAllocations();

    // Get the data, create the Qudit* targets
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
        .Case("plus", [&]() { photonic_simulator()->plus(localC, localT[0]); })
        .Case("beam_splitter",
              [&]() {
                photonic_simulator()->beam_splitter(parameters[0], localC,
                                                    localT);
              })
        .Case("phase_shift",
              [&]() {
                photonic_simulator()->phase_shift(parameters[0], localC,
                                                  localT[0]);
              })
        .Default([&]() {
          if (auto iter = registeredOperations.find(gateName);
              iter != registeredOperations.end()) {
            auto data = iter->second->unitary(parameters);
            photonic_simulator()->applyCustomOperation(data, localC, localT,
                                                       gateName);
            return;
          }
          throw std::runtime_error("[PhotonicsExecutionManager] invalid gate "
                                   "application requested " +
                                   gateName + ".");
        })();
  }

  int measureQudit(const cudaq::QuditInfo &q,
                   const std::string &registerName) override {
    flushRequestedAllocations();
    return photonic_simulator()->mz(q.id, registerName);
  }

  void flushGateQueue() override {
    synchronize();
    flushRequestedAllocations();
    photonic_simulator()->flushGateQueue();
  }

  int measure(const cudaq::QuditInfo &target,
              const std::string registerName = "") override {
    if (isInTracerMode())
      return 0;

    // We hit a measure, need to exec / clear instruction queue
    synchronize();

    // Instruction executed, run the measure call
    return measureQudit(target, registerName);
  }

  /// @brief Measure the state in the basis described by the given `spin_op`.
  void measureSpinOp(const cudaq::spin_op &) override {
    throw "spin_op operation (cudaq::observe()) is not supported for this "
          "photonics simulator";
  }

public:
  PhotonicsExecutionManager() {
    cudaq::info("[PhotonicsExecutionManager] Creating the {} backend.",
                photonic_simulator()->name());
  }

  virtual ~PhotonicsExecutionManager() = default;

  void resetQudit(const cudaq::QuditInfo &q) override {
    flushRequestedAllocations();
    photonic_simulator()->resetQudit(q.id);
  }

}; // PhotonicsExecutionManager

} // namespace cudaq

CUDAQ_REGISTER_EXECUTION_MANAGER(PhotonicsExecutionManager, photonics)
