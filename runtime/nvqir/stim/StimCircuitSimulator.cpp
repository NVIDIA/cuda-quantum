/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "nvqir/CircuitSimulator.h"
#include "stim.h"

using namespace cudaq;

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
  isValidStimNoiseChannel(const kraus_channel &channel) const {

    // Check the old way first
    switch (channel.noise_type) {
    case cudaq::noise_model_type::bit_flip_channel:
    case cudaq::noise_model_type::x_error:
      return StimNoiseType{.stim_name = "X_ERROR",
                           .flips_x = {true},
                           .flips_z = {false},
                           .params = {channel.parameters[0]}};
    case cudaq::noise_model_type::y_error:
      return StimNoiseType{.stim_name = "Y_ERROR",
                           .flips_x = {true},
                           .flips_z = {true},
                           .params = {channel.parameters[0]}};
    case cudaq::noise_model_type::phase_flip_channel:
    case cudaq::noise_model_type::z_error:
      return StimNoiseType{.stim_name = "Z_ERROR",
                           .flips_x = {false},
                           .flips_z = {true},
                           .params = {channel.parameters[0]}};
    case cudaq::noise_model_type::depolarization_channel:
    case cudaq::noise_model_type::depolarization1:
      return StimNoiseType{
          .stim_name = "DEPOLARIZE1",
          .flips_x = {true, true, false},
          .flips_z = {false, true, true},
          .params = std::vector<double>(3, channel.parameters[0] / 3.0)};
    case cudaq::noise_model_type::depolarization2: {
      constexpr bool x_err[4] = {false, true, true, false}; // X errors for IXYZ
      constexpr bool z_err[4] = {false, false, true, true}; // Z errors for IXYZ
      StimNoiseType ret{.stim_name = "DEPOLARIZE2", .num_targets = 2};
      ret.params = std::vector<double>(15, channel.parameters[0] / 15.0);
      // Generate the entries for p/15: IX, IY, IZ, XI, XX, XY, XZ, YI, YX,
      // YY, YZ, ZI, ZX, ZY, ZZ
      for (int q1_err = 0; q1_err < 4; q1_err++) {   // qubit 1 loop
        for (int q2_err = 0; q2_err < 4; q2_err++) { // qubit 2 loop
          if (q1_err == 0 && q2_err == 0)            // skip II
            continue;
          // Push back the values for the two qubits, for both x and z errors
          ret.flips_x.insert(ret.flips_x.end(), {x_err[q1_err], x_err[q2_err]});
          ret.flips_z.insert(ret.flips_z.end(), {z_err[q1_err], z_err[q2_err]});
        }
      }
      return ret;
    }
    case cudaq::noise_model_type::pauli1: {
      // Either X error, Y error, or Z error happens, each with its own
      // probability that is specified in the 3 channel parameters.
      static_assert(cudaq::pauli1::num_parameters == 3);
      assert(channel.parameters.size() == cudaq::pauli1::num_parameters);
      return StimNoiseType{.stim_name = "PAULI_CHANNEL_1",
                           .flips_x = {true, true, false},
                           .flips_z = {false, true, true},
                           .params = channel.parameters};
    }
    case cudaq::noise_model_type::pauli2: {
      static_assert(cudaq::pauli2::num_parameters == 15);
      assert(channel.parameters.size() == cudaq::pauli2::num_parameters);
      StimNoiseType ret{.stim_name = "PAULI_CHANNEL_2", .num_targets = 2};
      // Generate the entries for: IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ,
      // ZI, ZX, ZY, ZZ
      std::vector<bool> x_err{false, true, true, false}; // X errors for IXYZ
      std::vector<bool> z_err{false, false, true, true}; // Z errors for IXYZ
      for (int q1_err = 0; q1_err < 4; q1_err++) {       // qubit 1 loop
        for (int q2_err = 0; q2_err < 4; q2_err++) {     // qubit 2 loop
          if (q1_err == 0 && q2_err == 0)                // skip II
            continue;
          // Push back the values for the two qubits, for both x and z errors
          ret.flips_x.insert(ret.flips_x.end(), {x_err[q1_err], x_err[q2_err]});
          ret.flips_z.insert(ret.flips_z.end(), {z_err[q1_err], z_err[q2_err]});
        }
      }
      ret.params = channel.parameters;
      return ret;
    }
    case cudaq::noise_model_type::amplitude_damping_channel:
    case cudaq::noise_model_type::amplitude_damping:
    case cudaq::noise_model_type::phase_damping:
    case cudaq::noise_model_type::unknown:
      return std::nullopt;
    }

    return std::nullopt;
  }

  /// @brief Grow the state vector by one qubit.
  void addQubitToState() override { addQubitsToState(1); }

  /// @brief Get the batch size to use for the Stim simulator.
  std::size_t getBatchSize() {
    // Default to single shot
    std::size_t batch_size = 1;
    auto *executionContext = getExecutionContext();
    if (executionContext && executionContext->name == "sample" &&
        !executionContext->hasConditionalsOnMeasureResults)
      batch_size = executionContext->shots;
    else if (executionContext && executionContext->name == "msm")
      batch_size =
          executionContext->msm_dimensions.value_or(std::make_pair(1, 1))
              .second;
    return batch_size;
  }

  /// @brief Return the number of rows and columns needed for a Parity Check
  /// Matrix
  std::optional<std::pair<std::size_t, std::size_t>>
  generateMSMSize() override {
    return std::make_pair(num_measurements, msm_err_count);
  }

  void generateMSM() override {
    const auto num_cols = getBatchSize();
    stim::simd_bit_table<W> msmSample = sampleSim->m_record.storage;
    // Disabled because it's too verbose, but left here as comments for
    // reference.
    // CUDAQ_INFO("msmSample is {} {}\n{}", msm_err_count, num_cols,
    //             msmSample.str(num_measurements, num_cols).c_str());

    // Now it's msmSample[error_mechanism_index][measure_idx]
    msmSample = msmSample.transposed();
    CountsDictionary counts;
    std::vector<std::string> sequentialData;
    sequentialData.reserve(num_cols);
    for (std::size_t shot = 0; shot < num_cols; shot++) {
      std::string aShot(num_measurements, '0');
      for (std::size_t b = 0; b < num_measurements; b++)
        aShot[b] = msmSample[shot][b] ? '1' : '0';
      counts[aShot]++;
      sequentialData.push_back(std::move(aShot));
    }
    ExecutionResult result(counts);
    result.sequentialData = std::move(sequentialData);
    executionContext->result = result;
  }

  /// @brief Override the default sized allocation of qubits
  /// here to be a bit more efficient than the default implementation
  void addQubitsToState(std::size_t qubitCount,
                        const void *stateDataIn = nullptr) override {
    if (stateDataIn)
      throw std::runtime_error("The Stim simulator does not support "
                               "initialization of qubits from state data.");

    if (!tableau) {
      CUDAQ_INFO("Creating new Stim Tableau simulator");
      // Bump the randomEngine before cloning and giving to the Tableau
      // simulator.
      randomEngine.discard(
          std::uniform_int_distribution<int>(1, 30)(randomEngine));
      tableau = std::make_unique<stim::TableauSimulator<W>>(
          std::mt19937_64(randomEngine), /*num_qubits=*/0, /*sign_bias=*/+0);
    }
    if (!sampleSim) {
      is_msm_mode = executionContext && executionContext->name == "msm";
      std::size_t anticipated_num_measurements = 0;
      std::size_t num_msm_cols = 0;
      if (is_msm_mode) {
        auto dims =
            executionContext->msm_dimensions.value_or(std::make_pair(1, 1));
        anticipated_num_measurements = dims.first;
        num_msm_cols = dims.second;
        executionContext->msm_probabilities.emplace();
        executionContext->msm_probabilities->reserve(num_msm_cols);
        executionContext->msm_prob_err_id.emplace();
        executionContext->msm_prob_err_id->reserve(num_msm_cols);
      }

      // If possible, provide a non-empty stim::CircuitStats in order to avoid
      // reallocations during execution.
      stim::CircuitStats circuit_stats;
      circuit_stats.num_measurements = anticipated_num_measurements;

      auto batch_size = getBatchSize();
      CUDAQ_INFO("Creating new Stim frame simulator with batch size {}",
                 batch_size);
      // Bump the randomEngine before cloning and giving to the sample
      // simulator.
      randomEngine.discard(
          std::uniform_int_distribution<int>(1, 30)(randomEngine));
      sampleSim = std::make_unique<stim::FrameSimulator<W>>(
          circuit_stats, stim::FrameSimulatorMode::STORE_MEASUREMENTS_TO_MEMORY,
          batch_size, std::mt19937_64(randomEngine));
      if (is_msm_mode) {
        sampleSim->guarantee_anticommutation_via_frame_randomization = false;
      }
      sampleSim->reset_all();
      msm_err_count = 0;
      msm_id_counter = 0;
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
    msm_err_count = 0;
    msm_id_counter = 0;
    is_msm_mode = false;
  }

  /// @brief Apply operation to all Stim simulators.
  void applyOpToSims(const std::string &gate_name,
                     const std::vector<uint32_t> &targets) {
    if (targets.empty())
      return;
    stim::Circuit tempCircuit;
    CUDAQ_INFO("Calling applyOpToSims {} - {}", gate_name, targets);
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

    CUDAQ_INFO("Applying {} kraus channels to qubits {}", krausChannels.size(),
               stimTargets);

    for (auto &channel : krausChannels)
      applyNoise(channel, stimTargets);
  }

  bool isValidNoiseChannel(const cudaq::noise_model_type &type) const override {
    kraus_channel c;
    c.noise_type = type;
    return isValidStimNoiseChannel(c).has_value();
  }

  void applyNoise(const cudaq::kraus_channel &channel,
                  const std::vector<std::size_t> &qubits) override {
    flushGateQueue();
    std::vector<std::uint32_t> stimTargets(qubits.begin(), qubits.end());
    applyNoise(channel, stimTargets);
  }

  void applyNoise(const cudaq::kraus_channel &channel,
                  const std::vector<std::uint32_t> &qubits) {
    CUDAQ_INFO("[stim] apply kraus channel {}, is_msm_mode = {}",
               channel.get_type_name(), is_msm_mode);

    // If we have a valid operation, apply it
    if (auto res = isValidStimNoiseChannel(channel)) {
      if (is_msm_mode) {
        // If the noise operation is the first operation done to a qubit, the
        // x_table and z_table may not be sized for the qubits. If that is the
        // case, then we simply perform a reset on the qubit to essentially
        // allocate it, which ensures the tables are resized to the correct
        // size.
        auto max_qubit = *std::max_element(qubits.begin(), qubits.end());
        if (sampleSim->num_qubits < max_qubit + 1)
          applyOpToSims("R", std::vector<std::uint32_t>{max_qubit});

        // Apply the errors found in res directly into sampleSim, as if they
        // definitely happened, 1 mechanism at a time. (For example, a
        // depolarization channel will manifest as 3 possible error mechanisms:
        // an X error, Y error, or Z error.)
        std::size_t num_mechanisms = res->params.size();
        std::size_t flip_ix = 0;
        for (std::size_t m = 0; m < num_mechanisms; m++) {
          // In this mode, the "shot" is an alias for the MSM error count.
          std::size_t shot = msm_err_count;
          if (msm_err_count < sampleSim->batch_size) {
            for (std::size_t t = 0; t < res->num_targets; t++, flip_ix++) {
              sampleSim->x_table[qubits[t]][shot] ^= res->flips_x[flip_ix];
              sampleSim->z_table[qubits[t]][shot] ^= res->flips_z[flip_ix];
            }
            executionContext->msm_probabilities->push_back(res->params[m]);
            executionContext->msm_prob_err_id->push_back(msm_id_counter);
            msm_err_count++;
          }
        }
        msm_id_counter++;
      } else {
        stim::Circuit noiseOps;
        noiseOps.safe_append_u(res.value().stim_name, qubits,
                               channel.parameters);
        // Only apply the noise operations to the sample simulator (not the
        // Tableau simulator).
        sampleSim->safe_do_circuit(noiseOps);

        // Increment the error count by the number of mechanisms
        msm_err_count += res->params.size();
      }
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
    if (!sampleSim)
      throw std::runtime_error("Stim simulator state is not initialized. "
                               "Cannot sample from uninitialized state.");
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
