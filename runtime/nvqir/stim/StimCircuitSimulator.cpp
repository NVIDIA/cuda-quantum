/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "nvqir/CircuitSimulator.h"
#include "nvqir/Gates.h"
#include "stim.h"

#include <bit>
#include <iostream>
#include <set>
#include <span>

using namespace cudaq;

namespace nvqir {

/// @brief The StimCircuitSimulator implements the CircuitSimulator
/// base class to provide a simulator delegating to the Stim library from
/// https://github.com/quantumlib/Stim.
class StimCircuitSimulator : public nvqir::CircuitSimulatorBase<double> {
protected:
  stim::Circuit stimCircuit;
  std::mt19937_64 randomEngine;

  /// @brief Grow the state vector by one qubit.
  void addQubitToState() override { addQubitsToState(1); }

  /// @brief Override the default sized allocation of qubits
  /// here to be a bit more efficient than the default implementation
  void addQubitsToState(std::size_t qubitCount,
                        const void *stateDataIn = nullptr) override {
    if (stateDataIn)
      throw std::runtime_error("The Stim simulator does not support "
                               "initialization of qubits from state data.");
    return;
  }

  /// @brief Reset the qubit state.
  void deallocateStateImpl() override { stimCircuit.clear(); }

  /// @brief Apply the noise channel on \p qubits
  void applyNoiseChannel(const std::string_view gateName,
                         const std::vector<std::size_t> &controls,
                         const std::vector<std::size_t> &targets,
                         const std::vector<double> &params) override {
    // Do nothing if no execution context
    if (!executionContext)
      return;

    // Do nothing if no noise model
    if (!executionContext->noiseModel)
      return;

    // Get the name as a string
    std::string gName(gateName);

    // Cast size_t to uint32_t
    std::vector<std::uint32_t> stimTargets;
    stimTargets.reserve(controls.size() + targets.size());
    for (auto q : controls)
      stimTargets.push_back(static_cast<std::uint32_t>(q));
    for (auto q : targets)
      stimTargets.push_back(static_cast<std::uint32_t>(q));

    // Get the Kraus channels specified for this gate and qubits
    auto krausChannels = executionContext->noiseModel->get_channels(
        gName, controls, targets, params);

    // If none, do nothing
    if (krausChannels.empty())
      return;

    // TODO
    return;
  }

  void applyGate(const GateApplicationTask &task) override {
    std::string gateName(task.operationName);
    std::transform(gateName.begin(), gateName.end(), gateName.begin(),
                   ::toupper);
    std::vector<std::uint32_t> stimTargets;

    // These CUDA-Q rotation gates have the same name as Stim "reset" gates.
    // Stim is a Clifford simulator, so it doesn't actually support rotational
    // gates. Throw exceptions if they are encountered here.
    // TODO - consider adding support for specific rotations (e.g. pi/2).
    if (gateName == "RX" || gateName == "RY" || gateName == "RZ")
      throw std::runtime_error(
          fmt::format("Gate not supported by Stim simulator: {}. Note that "
                      "Stim can only simulate Clifford gates.",
                      task.operationName));

    if (task.controls.size() > 1)
      throw std::runtime_error(
          "Gates with >1 controls not supported by stim simulator");
    if (task.controls.size() >= 1)
      gateName = "C" + gateName;
    for (auto c : task.controls)
      stimTargets.push_back(c);
    for (auto t : task.targets)
      stimTargets.push_back(t);
    try {
      stimCircuit.safe_append_u(gateName, stimTargets);
    } catch (std::out_of_range &e) {
      throw std::runtime_error(
          fmt::format("Gate not supported by Stim simulator: {}. Note that "
                      "Stim can only simulate Clifford gates.",
                      e.what()));
    }
  }

  /// @brief Set the current state back to the |0> state.
  void setToZeroState() override { return; }

  /// @brief Override the calculateStateDim because this is not a state vector
  /// simulator.
  std::size_t calculateStateDim(const std::size_t numQubits) override {
    return 0;
  }

  /// @brief Measure the qubit and return the result. Collapse the
  /// state vector.
  bool measureQubit(const std::size_t index) override { return false; }

  QubitOrdering getQubitOrdering() const override { return QubitOrdering::msb; }

public:
  StimCircuitSimulator() {
    // Populate the correct name so it is printed correctly during
    // deconstructor.
    summaryData.name = name();
  }
  virtual ~StimCircuitSimulator() = default;

  void setRandomSeed(std::size_t seed) override {
    randomEngine = std::mt19937_64(seed);
  }

  bool canHandleObserve() override { return false; }

  /// @brief Reset the qubit
  /// @param index 0-based index of qubit to reset
  void resetQubit(const std::size_t index) override {
    flushGateQueue();
    flushAnySamplingTasks();
    stimCircuit.safe_append_u(
        "R", std::vector<std::uint32_t>{static_cast<std::uint32_t>(index)});
  }

  /// @brief Sample the multi-qubit state.
  cudaq::ExecutionResult sample(const std::vector<std::size_t> &qubits,
                                const int shots) override {
    std::vector<std::uint32_t> stimTargetQubits(qubits.begin(), qubits.end());
    stimCircuit.safe_append_u("M", stimTargetQubits);
    if (false) {
      std::stringstream ss;
      ss << stimCircuit << '\n';
      cudaq::log("Stim circuit is\n{}", ss.str());
    }
    auto ref_sample = stim::TableauSimulator<
        stim::MAX_BITWORD_WIDTH>::reference_sample_circuit(stimCircuit);
    stim::simd_bit_table<stim::MAX_BITWORD_WIDTH> sample =
        stim::sample_batch_measurements(stimCircuit, ref_sample, shots,
                                        randomEngine, false);
    size_t bits_per_sample = stimCircuit.count_measurements();
    std::vector<std::string> sequentialData;
    sequentialData.reserve(shots);
    // Only retain the final "qubits.size()" measurements. All other
    // measurements were mid-circuit measurements that have been previously
    // accounted for and saved.
    assert(bits_per_sample >= qubits.size());
    std::size_t first_bit_to_save = bits_per_sample - qubits.size();
    CountsDictionary counts;
    for (std::size_t shot = 0; shot < shots; shot++) {
      std::string aShot(qubits.size(), '0');
      for (std::size_t b = first_bit_to_save; b < bits_per_sample; b++) {
        aShot[b - first_bit_to_save] = sample[b][shot] ? '1' : '0';
      }
      counts[aShot]++;
      sequentialData.push_back(std::move(aShot));
    }
    ExecutionResult result(counts);
    result.sequentialData = std::move(sequentialData);
    return result;
  }

  bool isStateVectorSimulator() const override { return false; }

  std::string name() const override { return "stim"; }
  NVQIR_SIMULATOR_CLONE_IMPL(StimCircuitSimulator)
};

} // namespace nvqir

#ifndef __NVQIR_QPP_TOGGLE_CREATE
/// Register this Simulator with NVQIR.
NVQIR_REGISTER_SIMULATOR(nvqir::StimCircuitSimulator, stim)
#endif
