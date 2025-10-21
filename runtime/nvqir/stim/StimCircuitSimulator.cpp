/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "StimState.h"
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

  /// @brief Whether or not the execution context name is "generate_data" (value
  /// is cached for speed)
  bool is_generate_data_mode = false;
  std::size_t replay_columns = 0;

  bool is_replay_errors_mode = false;

  size_t error_log_vec_index = 0;
  size_t noise_application_index = 0;
  size_t last_column_touched = 0;

  ExecutionContext *exe_ctx;

  ErrorLogType error_log;

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

  std::pair<std::vector<std::vector<uint8_t>>,
            std::vector<std::vector<uint8_t>>>
  get_x_z_tables_as_vectors(stim::FrameSimulator<W> *sampleSim) {
    auto *executionContext = getExecutionContext();
    auto batch_size = executionContext->shots;
    auto num_qubits = sampleSim->num_qubits;

    std::vector<std::vector<uint8_t>> x_run;
    std::vector<std::vector<uint8_t>> z_run;

    x_run.reserve(batch_size);
    z_run.reserve(batch_size);

    for (size_t shot = 0; shot < batch_size; shot++) {
      std::vector<uint8_t> x_shot(num_qubits);
      std::vector<uint8_t> z_shot(num_qubits);

      for (std::size_t q = 0; q < num_qubits; q++) {
        x_shot[q] = sampleSim->x_table[q][shot] ? 1 : 0;
        z_shot[q] = sampleSim->z_table[q][shot] ? 1 : 0;
      }

      x_run.push_back(std::move(x_shot));
      z_run.push_back(std::move(z_shot));
    }

    return std::make_pair(std::move(x_run), std::move(z_run));
  }

  std::vector<uint8_t> xor_vectors(const std::vector<uint8_t> &a,
                                   const std::vector<uint8_t> &b) {
    // Determine the length of the longer vector
    size_t max_size = std::max(a.size(), b.size());

    // Create copies of the vectors, padded with zeros if necessary
    std::vector<uint8_t> a_padded = a;
    std::vector<uint8_t> b_padded = b;

    a_padded.resize(max_size, 0);
    b_padded.resize(max_size, 0);

    // XOR element-wise
    std::vector<uint8_t> result(max_size);
    for (size_t i = 0; i < max_size; ++i) {
      result[i] = a_padded[i] ^ b_padded[i];
    }

    return result;
  }

  /// @brief Find indices where vector elements equal 1
  std::vector<size_t> find_one_indices(const std::vector<uint8_t> &vec) const {
    std::vector<size_t> indices;
    indices.reserve(vec.size()); // Optimize allocation
    for (size_t i = 0; i < vec.size(); i++) {
      if (vec[i] == 1) {
        indices.push_back(i);
      }
    }
    return indices;
  }

  StimData serialize_frame_simulator(stim::FrameSimulator<W> *sampleSim) {
    StimData data;
    CUDAQ_INFO("Serializing Stim Frame Simulator data");

    auto *executionContext = getExecutionContext();
    auto batch_size = executionContext->shots;
    CUDAQ_INFO("batch_size: {}", batch_size);
    std::size_t num_qubits = sampleSim->num_qubits;

    // 0: num_qubits
    std::size_t *num_qubits_ptr = new std::size_t(num_qubits);
    CUDAQ_INFO("num_qubits: {}", *num_qubits_ptr);
    data.push_back({num_qubits_ptr, 1});

    // 1: msm_err_count
    std::size_t *msm_err_count_ptr = new std::size_t(msm_err_count);
    CUDAQ_INFO("msm_err_count: {}", *msm_err_count_ptr);
    data.push_back({msm_err_count_ptr, 1});

    // 2,3: x_output and z_output
    uint8_t *x_output = new uint8_t[num_qubits * batch_size];
    uint8_t *z_output = new uint8_t[num_qubits * batch_size];

    for (int shot = 0; shot < batch_size; shot++) {
      for (std::size_t q = 0; q < num_qubits; q++) {
        CUDAQ_INFO("q {}: x = {}, z = {}", q,
                    static_cast<int>(sampleSim->x_table[q][shot]),
                    static_cast<int>(sampleSim->z_table[q][shot]));

        x_output[q + shot * num_qubits] = sampleSim->x_table[q][shot] ? 1 : 0;
        z_output[q + shot * num_qubits] = sampleSim->z_table[q][shot] ? 1 : 0;
      }
    }

    data.push_back({x_output, num_qubits * batch_size});
    data.push_back({z_output, num_qubits * batch_size});

    return data;
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
    else if (executionContext && executionContext->name == "generate_data")
      batch_size = executionContext->shots;
    else if (executionContext && executionContext->name == "replay_errors")
      batch_size = executionContext->replay_columns;
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
      is_generate_data_mode =
          executionContext && executionContext->name == "generate_data";
      is_replay_errors_mode = executionContext &&
                              executionContext->name == "replay_errors" &&
                              !executionContext->get_error_data().empty();

      if (is_generate_data_mode) {
        CUDAQ_INFO("Generate data mode enabled");
        noise_application_index = 0; // Reset for generation
      }

      if (is_replay_errors_mode) {
        CUDAQ_INFO("Replay errors mode enabled with {} logged errors",
                   executionContext->get_error_data().size());
        noise_application_index = 0; // Reset for replay matching
        error_log_vec_index = 0;     // Reset replay cursor to start
        last_column_touched = 0;
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
      if (is_msm_mode || is_generate_data_mode || is_replay_errors_mode) {
        sampleSim->guarantee_anticommutation_via_frame_randomization = false;
      }
      sampleSim->reset_all();
      msm_err_count = 0;
      msm_id_counter = 0;
      error_log_vec_index = 0;
      noise_application_index = 0;
      last_column_touched = 0;
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
    error_log_vec_index = 0;
    noise_application_index = 0;
    last_column_touched = 0;
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
      auto max_qubit = *std::max_element(qubits.begin(), qubits.end());
      if (is_msm_mode) {
        // If the noise operation is the first operation done to a qubit, the
        // x_table and z_table may not be sized for the qubits. If that is the
        // case, then we simply perform a reset on the qubit to essentially
        // allocate it, which ensures the tables are resized to the correct
        // size.
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
      } else if (is_generate_data_mode) {
        CUDAQ_INFO("Generating data for noise operation ID {}",
                   error_log_vec_index);

        // allocate the qubits if needed
        if (sampleSim->num_qubits < max_qubit + 1)
          applyOpToSims("R", std::vector<std::uint32_t>{max_qubit});

        CUDAQ_INFO("Applying noise operation {} to qubits {}", res->stim_name,
                   fmt::join(qubits, ", "));

        stim::Circuit noiseOps;
        noiseOps.safe_append_u(res.value().stim_name, qubits,
                               channel.parameters);

        auto tables = get_x_z_tables_as_vectors(sampleSim.get());
        // Only apply the noise operations to the sample simulator (not the
        // Tableau simulator).
        sampleSim->safe_do_circuit(noiseOps);

        auto tables_after = get_x_z_tables_as_vectors(sampleSim.get());

        // Initialize the log entry for this noise operation
        ErrorByShotLogEntry error_log_entry;

        // Get batch size from the execution context
        auto batch_size = executionContext->shots;

        // tables[0] = all X data, tables[1] = all Z data
        // Each is a vector of shots, each shot is a vector of qubits
        const auto &x_before = tables.first; // vector of shots
        const auto &z_before = tables.second;
        const auto &x_after = tables_after.first;
        const auto &z_after = tables_after.second;

        // Process each shot
        for (size_t shot = 0; shot < batch_size; shot++) {
          // XOR to find which qubits flipped in this shot
          auto x_diff = xor_vectors(x_before[shot], x_after[shot]);
          auto z_diff = xor_vectors(z_before[shot], z_after[shot]);

          // Extract the qubit indices where flips occurred
          auto x_flipped_qubits = find_one_indices(x_diff);
          auto z_flipped_qubits = find_one_indices(z_diff);

          // Debug output
          if (!x_flipped_qubits.empty() || !z_flipped_qubits.empty()) {
            CUDAQ_INFO(
                "Shot {}: X errors on qubits [{}], Z errors on qubits [{}]",
                shot, fmt::join(x_flipped_qubits, ", "),
                fmt::join(z_flipped_qubits, ", "));
          }

          // Store the indices for this shot (even if empty - maintains shot
          // alignment)
          if (!x_flipped_qubits.empty() && !z_flipped_qubits.empty()) {
            error_log_entry.first.push_back(x_flipped_qubits);
            error_log_entry.second.push_back(z_flipped_qubits);
            replay_columns += 2;
          }
        }

        // Record the entire batch of errors under one ID
        if (error_log_entry.first.empty() && error_log_entry.second.empty()) {
          CUDAQ_INFO("No errors occurred for noise operation ID {}",
                     error_log_vec_index);
        } else {
          CUDAQ_INFO("Recording errors for noise operation ID {}",
                     error_log_vec_index);
          executionContext->record_error_data(error_log_vec_index,
                                              error_log_entry);
          executionContext->update_replay_columns(replay_columns);
        }

        error_log_vec_index++;
      } else if (is_replay_errors_mode) {
        CUDAQ_INFO("In replay mode: Noise application index: {}", noise_application_index);
        CUDAQ_INFO("Replaying errors for noise operation ID {}", error_log_vec_index);

        if (sampleSim->num_qubits < max_qubit + 1)
          applyOpToSims("R", std::vector<std::uint32_t>{max_qubit});

        // Ensure we have errors to replay
        const auto &errors_to_replay = executionContext->get_error_data();
        if (errors_to_replay.empty()) {
          CUDAQ_INFO("No errors to replay");
          return;
        }

        // Check if we've exhausted the error log
        if (noise_application_index >= errors_to_replay.size()) {
          CUDAQ_INFO("All logged errors have been replayed");
          return;
        }

        // wrong: CUDAQ_INFO("Replaying error entry {} of {}",
        // noise_application_index, errors_to_replay.size());

        // Fetch the error entry for this noise operation
        const auto &error_entry = errors_to_replay[error_log_vec_index];
        const size_t logged_error_id = std::get<0>(error_entry);
        const auto &error_log_entry = std::get<1>(error_entry);

        // Verify this is the correct error to replay
        if (noise_application_index != logged_error_id) {
          CUDAQ_INFO("Skipping - current noise application index {} doesn't "
                     "match logged ID {}",
                     noise_application_index, logged_error_id);
          noise_application_index++;
          return;
        }

        // Extract X and Z error data
        const auto &x_errors_per_shot = error_log_entry.first;
        const auto &z_errors_per_shot = error_log_entry.second;

        // Get the number of shots to replay
        size_t num_shots = x_errors_per_shot.size();
        CUDAQ_INFO("Replaying {} shots for error ID {}", num_shots, error_log_vec_index);

        if (last_column_touched + num_shots > getBatchSize()) {
          throw std::runtime_error(fmt::format(
              "Not enough columns in Stim FrameSimulator to replay errors. "
              "Needed {}, but only have {}.",
              last_column_touched + num_shots, getBatchSize()));
        }
        // Apply the logged errors to each shot
        for (size_t shot = 0; shot < num_shots; shot++) {
          const auto &x_qubits = x_errors_per_shot[shot];
          const auto &z_qubits = z_errors_per_shot[shot];

          // Apply X errors
          for (uint8_t qubit : x_qubits) {
            sampleSim->x_table[qubit][last_column_touched + shot] ^= 1;
            CUDAQ_INFO("  Shot {}: Applied X error to qubit {}", shot, qubit);
          }

          // Apply Z errors
          for (uint8_t qubit : z_qubits) {
            sampleSim->z_table[qubit][last_column_touched + shot] ^= 1;
            CUDAQ_INFO("  Shot {}: Applied Z error to qubit {}", shot, qubit);
          }
        }
        last_column_touched += num_shots;
        // Move to the next error entry
        noise_application_index++;
        error_log_vec_index++;
        CUDAQ_INFO("Finished replaying errors for noise operation ID {}",
                   error_log_vec_index - 1);
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

    auto exe_ctx = getExecutionContext();
    if (exe_ctx && exe_ctx->randomSeed != 0) {
      setRandomSeed(exe_ctx->randomSeed);
      CUDAQ_INFO("Setting random seed to {} in Stim simulator",
                 exe_ctx->randomSeed);
    }
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

  std::unique_ptr<cudaq::SimulationState> getSimulationState() override {
    flushGateQueue();
    StimData data = serialize_frame_simulator(sampleSim.get());
    return std::make_unique<StimState>(data);
  }

  std::unique_ptr<cudaq::SimulationState> getCurrentSimulationState() override {
    CUDAQ_INFO("Getting current simulation state from stim simulator");
    flushGateQueue();
    StimData data = serialize_frame_simulator(sampleSim.get());
    return std::make_unique<StimState>(data);
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
