/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
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

  std::optional<std::string>
  isValidStimNoiseChannel(const kraus_channel &channel) const {

    // Check the old way first
    if (channel.noise_type == cudaq::noise_model_type::bit_flip_channel)
      return "X_ERROR";

    if (channel.noise_type == cudaq::noise_model_type::phase_flip_channel)
      return "Z_ERROR";

    if (channel.noise_type == cudaq::noise_model_type::depolarization_channel)
      return "DEPOLARIZE1";

    return std::nullopt;
  }

  /// @brief Grow the state vector by one qubit.
  void addQubitToState() override { addQubitsToState(1); }

  /// @brief Get the batch size to use for the Stim sample simulator.
  std::size_t getBatchSize() {
    // Default to single shot
    std::size_t batch_size = 1;
    if (getExecutionContext() && getExecutionContext()->name == "sample" &&
        !getExecutionContext()->hasConditionalsOnMeasureResults)
      batch_size = getExecutionContext()->shots;
    return batch_size;
  }

  /// @brief Override the default sized allocation of qubits
  /// here to be a bit more efficient than the default implementation
  void addQubitsToState(std::size_t qubitCount,
                        const void *stateDataIn = nullptr) override {
    if (stateDataIn)
      throw std::runtime_error("The Stim simulator does not support "
                               "initialization of qubits from state data.");

    if (!tableau) {
      cudaq::info("Creating new Stim Tableau simulator");
      // Bump the randomEngine before cloning and giving to the Tableau
      // simulator.
      randomEngine.discard(
          std::uniform_int_distribution<int>(1, 30)(randomEngine));
      tableau = std::make_unique<stim::TableauSimulator<W>>(
          std::mt19937_64(randomEngine), /*num_qubits=*/0, /*sign_bias=*/+0);
    }
    if (!sampleSim) {
      auto batch_size = getBatchSize();
      cudaq::info("Creating new Stim frame simulator with batch size {}",
                  batch_size);
      // Bump the randomEngine before cloning and giving to the sample
      // simulator.
      randomEngine.discard(
          std::uniform_int_distribution<int>(1, 30)(randomEngine));
      sampleSim = std::make_unique<stim::FrameSimulator<W>>(
          stim::CircuitStats(),
          stim::FrameSimulatorMode::STORE_MEASUREMENTS_TO_MEMORY, batch_size,
          std::mt19937_64(randomEngine));
      sampleSim->reset_all();
    }
  }

  /// @brief Reset the qubit state.
  void deallocateStateImpl() override {
    tableau.reset();
    // Update the randomEngine so that future invocations will use the updated
    // RNG state.
    if (sampleSim)
      randomEngine = std::move(sampleSim->rng);
    sampleSim.reset();
    num_measurements = 0;
  }

  /// @brief Apply operation to all Stim simulators.
  void applyOpToSims(const std::string &gate_name,
                     const std::vector<uint32_t> &targets) {
    if (targets.empty())
      return;
    stim::Circuit tempCircuit;
    cudaq::info("Calling applyOpToSims {} - {}", gate_name, targets);
    tempCircuit.safe_append_u(gate_name, targets);
    tableau->safe_do_circuit(tempCircuit);
    sampleSim->safe_do_circuit(tempCircuit);
  }

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
        gName, targets, controls, params);

    // If none, do nothing
    if (krausChannels.empty())
      return;

    cudaq::info("Applying {} kraus channels to qubits {}", krausChannels.size(),
                stimTargets);

    stim::Circuit noiseOps;
    for (auto &channel : krausChannels) {
      if (auto stimName = isValidStimNoiseChannel(channel))
        noiseOps.safe_append_u(stimName.value(), stimTargets,
                               channel.parameters);
    }
    // Only apply the noise operations to the sample simulator (not the Tableau
    // simulator).
    sampleSim->safe_do_circuit(noiseOps);
  }

  bool isValidNoiseChannel(const cudaq::noise_model_type &type) const override {
    kraus_channel c;
    c.noise_type = type;
    return isValidStimNoiseChannel(c).has_value();
  }

  void applyNoise(const cudaq::kraus_channel &channel,
                  const std::vector<std::size_t> &qubits) override {
    flushGateQueue();
    cudaq::info("[stim] apply kraus channel {}", channel.get_type_name());
    stim::Circuit noiseOps;
    std::vector<std::uint32_t> stimTargets;
    stimTargets.reserve(qubits.size());
    for (auto q : qubits)
      stimTargets.push_back(static_cast<std::uint32_t>(q));

    // If we have a valid operation, apply it
    if (auto stimName = isValidStimNoiseChannel(channel)) {
      noiseOps.safe_append_u(stimName.value(), stimTargets, channel.parameters);
      sampleSim->safe_do_circuit(noiseOps);
    }
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
    else if (gateName == "SDG")
      gateName = "S_DAG";

    if (task.controls.size() > 1)
      throw std::runtime_error(
          "Gates with >1 controls not supported by Stim simulator");
    if (task.controls.size() >= 1)
      gateName = "C" + gateName;
    for (auto c : task.controls)
      stimTargets.push_back(c);
    for (auto t : task.targets)
      stimTargets.push_back(t);
    try {
      applyOpToSims(gateName, stimTargets);
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

  /// @brief Measure the qubit and return the result.
  bool measureQubit(const std::size_t index) override {
    // Perform measurement
    applyOpToSims(
        "M", std::vector<std::uint32_t>{static_cast<std::uint32_t>(index)});
    num_measurements++;

    // Get the tableau bit that was just generated.
    const std::vector<bool> &v = tableau->measurement_record.storage;
    const bool tableauBit = *v.crbegin();

    // Get the mid-circuit sample to be XOR-ed with tableauBit.
    bool sampleSimBit =
        sampleSim->m_record.storage[num_measurements - 1][/*shot=*/0];

    // Calculate the result.
    bool result = tableauBit ^ sampleSimBit;

    return result;
  }

  QubitOrdering getQubitOrdering() const override { return QubitOrdering::msb; }

public:
  StimCircuitSimulator() : randomEngine(std::random_device{}()) {
    // Populate the correct name so it is printed correctly during
    // deconstructor.
    summaryData.name = name();
    // Set supportsBufferedSample = true to tell the base class that this
    // simulator knows how to buffer the results across multiple sample()
    // invocations.
    supportsBufferedSample = true;
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
    applyOpToSims(
        "R", std::vector<std::uint32_t>{static_cast<std::uint32_t>(index)});
  }

  /// @brief Sample the multi-qubit state. If \p qubits is empty and
  /// explicitMeasurements is set, this returns all previously saved
  /// measurements.
  cudaq::ExecutionResult sample(const std::vector<std::size_t> &qubits,
                                const int shots) override {
    if (executionContext->explicitMeasurements && qubits.empty() &&
        num_measurements == 0)
      throw std::runtime_error(
          "The sampling option `explicit_measurements` is not supported on a "
          "kernel without any measurement operation.");

    bool populateResult = [&]() {
      if (executionContext->explicitMeasurements)
        return qubits.empty();
      return true;
    }();
    assert(shots <= sampleSim->batch_size);
    std::vector<std::uint32_t> stimTargetQubits(qubits.begin(), qubits.end());
    applyOpToSims("M", stimTargetQubits);
    num_measurements += stimTargetQubits.size();

    if (!populateResult)
      return cudaq::ExecutionResult();

    // Generate a reference sample
    const std::vector<bool> &v = tableau->measurement_record.storage;
    stim::simd_bits<W> ref(v.size());
    for (size_t k = 0; k < v.size(); k++)
      ref[k] ^= v[k];

    // Now XOR results on a per-shot basis
    stim::simd_bit_table<W> sample = sampleSim->m_record.storage;
    auto nShots = sampleSim->batch_size;

    // This is a slightly modified version of `sample_batch_measurements`, where
    // we already have the `sample` from the frame simulator. It also places the
    // `sample` in a layout amenable to the order of the loops below (shot
    // major).
    sample = sample.transposed();
    if (ref.not_zero())
      for (size_t s = 0; s < nShots; s++)
        sample[s].word_range_ref(0, ref.num_simd_words) ^= ref;

    size_t bits_per_sample = num_measurements;
    std::vector<std::string> sequentialData;
    // Only retain the final "qubits.size()" measurements. All other
    // measurements were mid-circuit measurements that have been previously
    // accounted for and saved.
    assert(bits_per_sample >= qubits.size());
    std::size_t first_bit_to_save = executionContext->explicitMeasurements
                                        ? 0
                                        : bits_per_sample - qubits.size();
    CountsDictionary counts;
    sequentialData.reserve(shots);
    for (std::size_t shot = 0; shot < shots; shot++) {
      std::string aShot(bits_per_sample - first_bit_to_save, '0');
      for (std::size_t b = first_bit_to_save; b < bits_per_sample; b++)
        aShot[b - first_bit_to_save] = sample[shot][b] ? '1' : '0';
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
