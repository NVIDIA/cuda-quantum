/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "nvqir/CircuitSimulator.h"
#include "stim.h"
#include <random>

namespace nvqir {

/// @brief Collection of information about a noise type in Stim
struct StimNoiseType {
  /// The name of the error mechanism in Stim
  std::string stim_name;
  /// Whether the error mechanism flips X; one per error mechanism per target
  std::vector<bool> flips_x;
  /// Whether the error mechanism flips Z; one per error mechanism per target
  std::vector<bool> flips_z;
  /// One probability per error mechanism
  std::vector<double> params;
  /// The number of targets for the noise type
  int num_targets = 1;
};

/// @brief The StimCircuitSimulator implements the CircuitSimulator
/// base class to provide a simulator delegating to the Stim library from
/// https://github.com/quantumlib/Stim.
class StimCircuitSimulator : public nvqir::CircuitSimulatorBase<double> {
protected:
  // Follow Stim naming convention (W) for bit width (required for templates).
  static constexpr std::size_t W = stim::MAX_BITWORD_WIDTH;

  /// @brief Number of measurements performed so far.
  std::size_t num_measurements = 0;

  /// @brief Top-level random engine. Stim simulator RNGs are based off of this
  /// engine.
  std::mt19937_64 randomEngine;

  /// @brief Stim Tableau simulator (noiseless)
  std::unique_ptr<stim::TableauSimulator<W>> tableau;

  /// @brief Stim Frame/Flip simulator (used to generate multiple shots)
  std::unique_ptr<stim::FrameSimulator<W>> sampleSim;

  /// @brief Error counter for MSM generation. This is only used for "msm" and
  /// "msm_size" execution contexts.
  std::size_t msm_err_count = 0;

  /// @brief Error ID counter for MSM generation. This is only used for "msm"
  /// and "msm_size" execution contexts.
  std::size_t msm_id_counter = 0;

  /// @brief Whether or not the execution context name is "msm" (value is cached
  /// for speed)
  bool is_msm_mode = false;

  std::optional<StimNoiseType>
  isValidStimNoiseChannel(const cudaq::kraus_channel &channel) const;

  /// @brief Grow the state vector by one qubit.
  void addQubitToState() override;

  /// @brief Get the batch size to use for the Stim simulator.
  std::size_t getBatchSize();

  /// @brief Return the number of rows and columns needed for a Parity Check
  /// Matrix
  std::optional<std::pair<std::size_t, std::size_t>>
  generateMSMSize() override;

  void generateMSM() override;

  /// @brief Override the default sized allocation of qubits
  /// here to be a bit more efficient than the default implementation
  void addQubitsToState(std::size_t qubitCount,
                        const void *stateDataIn = nullptr) override;

  /// @brief Reset the qubit state.
  void deallocateStateImpl() override;

  /// @brief Apply operation to all Stim simulators.
  /// This is the key method that derived classes should override to replace
  /// the frame simulator execution.
  virtual void applyOpToSims(const std::string &gate_name,
                             const std::vector<uint32_t> &targets);

  /// @brief Apply the noise channel on \p qubits
  void applyNoiseChannel(const std::string_view gateName,
                         const std::vector<std::size_t> &controls,
                         const std::vector<std::size_t> &targets,
                         const std::vector<double> &params) override;

  bool isValidNoiseChannel(const cudaq::noise_model_type &type) const override;

  void applyNoise(const cudaq::kraus_channel &channel,
                  const std::vector<std::size_t> &qubits) override;

  virtual void applyNoise(const cudaq::kraus_channel &channel,
                          const std::vector<std::uint32_t> &qubits);

  void applyGate(const GateApplicationTask &task) override;

  /// @brief Set the current state back to the |0> state.
  void setToZeroState() override;

  /// @brief Override the calculateStateDim because this is not a state vector
  /// simulator.
  std::size_t calculateStateDim(const std::size_t numQubits) override;

  /// @brief Measure the qubit and return the result.
  bool measureQubit(const std::size_t index) override;

  QubitOrdering getQubitOrdering() const override;

public:
  StimCircuitSimulator();
  virtual ~StimCircuitSimulator() = default;

  void setRandomSeed(std::size_t seed) override;

  bool canHandleObserve() override;

  /// @brief Reset the qubit
  /// @param index 0-based index of qubit to reset
  void resetQubit(const std::size_t index) override;

  /// @brief Sample the multi-qubit state. If \p qubits is empty and
  /// explicitMeasurements is set, this returns all previously saved
  /// measurements.
  cudaq::ExecutionResult sample(const std::vector<std::size_t> &qubits,
                                const int shots) override;

  bool isStateVectorSimulator() const override;

  std::string name() const override;
  NVQIR_SIMULATOR_CLONE_IMPL(StimCircuitSimulator)
};

} // namespace nvqir
