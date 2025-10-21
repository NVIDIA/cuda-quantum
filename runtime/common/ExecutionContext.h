/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "Future.h"
#include "NoiseModel.h"
#include "SampleResult.h"
#include "SimulationState.h"
#include "Trace.h"
#include "cudaq/algorithms/optimizer.h"
#include "cudaq/operators.h"
#include <iostream>
#include <optional>
#include <string_view>

#include "nvqir/stim/StimState.h"

namespace cudaq {
using ErrorByShotLogEntry = std::pair<std::vector<std::vector<size_t>>,
                                      std::vector<std::vector<size_t>>>;
using ErrorLogType = std::vector<std::tuple<size_t, ErrorByShotLogEntry>>;

struct RecordStorage {

  size_t memory_limit;
  size_t current_memory;
  ErrorLogType error_data;
  RecordStorage(size_t limit = 1e9) : memory_limit(limit), current_memory(0) {}

  std::vector<std::unique_ptr<SimulationState>> recordedStates;

  void save_state(SimulationState *state) {
    recordedStates.push_back(clone_state(state));
  }
  const std::vector<std::unique_ptr<SimulationState>> &
  get_recorded_states() const {
    return recordedStates;
  }

  void clear() { recordedStates.clear(); }
  void dump_recorded_states() const {
    for (std::size_t i = 0; i < recordedStates.size(); i++) {
      recordedStates[i]->dump(std::cout);
    }
  }

  void dump_error_data() const {
    printf("=== Error Data Dump ===\n");
    if (error_data.empty()) {
      printf("(no error data)\n");
      return;
    }

    for (const auto &[index, entry] : error_data) {
      const auto &[x_errors, z_errors] =
          entry; // both are vector<vector<size_t>>

      printf("\n---------------------------------------\n");
      printf(" Error Index: %zu\n", index);
      printf("---------------------------------------\n");

      // X error Shots
      printf("  X Error Shots (%zu):\n", x_errors.size());
      if (x_errors.empty()) {
        printf("    (none)\n");
      } else {
        for (std::size_t i = 0; i < x_errors.size(); ++i) {
          printf("    - Set %zu (%zu elements): ", i, x_errors[i].size());
          for (const auto &q : x_errors[i])
            printf("%zu ", q);
          printf("\n");
        }
      }

      // Z error Shots
      printf("  Z Error Shots (%zu):\n", z_errors.size());
      if (z_errors.empty()) {
        printf("    (none)\n");
      } else {
        for (std::size_t i = 0; i < z_errors.size(); ++i) {
          printf("    - Set %zu (%zu elements): ", i, z_errors[i].size());
          for (const auto &q : z_errors[i])
            printf("%zu ", q);
          printf("\n");
        }
      }
    }

    printf("\n=== End of Error Data ===\n");
  }

  void record_error_data(const size_t index, const ErrorByShotLogEntry &entry) {
    error_data.emplace_back(index, entry);
  }
  ~RecordStorage() {
    std::cout << "Destroying RecordStorage with " << recordedStates.size()
              << " recorded states.\n";
  }

private:
  std::unique_ptr<SimulationState> clone_state(SimulationState *state) {
    if (state->isArrayLike()) {
      // Handle array-like states (CusvState, etc.)
      return clone_array_like_state(state);
    } else {
      // Handle specialized states (CuDensityMatState, StimState, etc.)
      return clone_specialized_state(state);
    }
  }

  std::unique_ptr<SimulationState>
  clone_array_like_state(SimulationState *state) {
    auto numQubits = state->getNumQubits();
    if (numQubits > 20) { // Prevent exponential explosion
      throw std::runtime_error("State too large to clone via amplitudes");
    }

    // Generate all basis states
    auto totalStates = 1ULL << numQubits;
    std::vector<std::vector<int>> basisStates;
    for (size_t i = 0; i < totalStates; ++i) {
      std::vector<int> basis(numQubits);
      for (size_t j = 0; j < numQubits; ++j) {
        basis[j] = (i >> j) & 1;
      }
      basisStates.push_back(basis);
    }

    auto amplitudes = state->getAmplitudes(basisStates);

    // Create new state with appropriate precision
    if (state->getPrecision() == SimulationState::precision::fp32) {
      std::vector<std::complex<float>> floatAmps;
      for (const auto &amp : amplitudes) {
        floatAmps.emplace_back(static_cast<float>(amp.real()),
                               static_cast<float>(amp.imag()));
      }
      return state->createFromData(floatAmps);
    } else {
      return state->createFromData(amplitudes);
    }
  }

  std::unique_ptr<SimulationState>
  clone_specialized_state(SimulationState *state) {
    // Try dynamic_cast to known types that have clone methods
    // this triggerd fatal error: library_types.h: No such file or directory
    // if (auto* densityState = dynamic_cast<const CuDensityMatState*>(state)) {
    //    return CuDensityMatState::clone(*densityState);
    //}

    if (auto *cloneable = dynamic_cast<ClonableState *>(state)) {
      return cloneable->clone();
    }

    // Fallback for non-cloneable specialized states
    throw std::runtime_error("Specialized state type does not support cloning");
    // For unknown specialized types, try createFromSizeAndPtr as fallback
    // This might work for some specialized states
    // auto tensor = state->getTensor(0);
    // return state->createFromSizeAndPtr(tensor.get_num_elements(),
    // tensor.data, 1);
  }
};

/// The ExecutionContext is an abstraction to indicate how a CUDA-Q kernel
/// should be executed.
class ExecutionContext {

  ///@brief record storage for the states saved during execution
  RecordStorage recordStorage;

public:
  /// @brief The Constructor, takes the name of the context
  /// @param n The name of the context
  ExecutionContext(const std::string &n) : name(n) {}

  /// @brief The constructor, takes the name and the number of shots.
  /// @param n The name of the context
  /// @param shots_ The number of shots
  ExecutionContext(const std::string &n, std::size_t shots_)
      : name(n), shots(shots_) {}

  ~ExecutionContext() = default;

  /// @brief The name of the context ({basic, sampling, observe})
  const std::string name;

  /// @brief The number of execution shots
  std::size_t shots = 0;

  /// @brief An optional spin operator
  std::optional<cudaq::spin_op> spin;

  /// @brief Measurement counts for a CUDA-Q kernel invocation
  sample_result result;

  /// @brief A computed expectation value
  std::optional<double> expectationValue = std::nullopt;

  /// @brief An optimization result
  std::optional<cudaq::optimization_result> optResult = std::nullopt;

  /// @brief The kernel being executed in this context has conditional
  /// statements on measure results.
  bool hasConditionalsOnMeasureResults = false;

  /// @brief Noise model to apply to the current execution.
  const noise_model *noiseModel = nullptr;

  /// @brief Flag to indicate if backend can handle spin_op observe task under
  /// this ExecutionContext.
  bool canHandleObserve = false;

  /// @brief Flag indicating that the current execution should occur
  /// asynchronously
  bool asyncExec = false;

  /// @brief When execution asynchronously, store the expected results as a
  /// cudaq::future here.
  details::future futureResult;

  /// @brief Construct a `async_sample_result` so as to pass across Python
  /// boundary
  async_result<sample_result> asyncResult;

  /// @brief Pointer to simulation-specific simulation data.
  std::unique_ptr<SimulationState> simulationState;

  /// @brief A map of basis-state amplitudes
  // The list of basis state is set before kernel launch and the map is filled
  // by the executor platform.
  std::optional<std::map<std::vector<int>, std::complex<double>>>
      amplitudeMaps = std::nullopt;

  /// @brief List of pairs of states to compute the overlap
  std::optional<std::pair<const SimulationState *, const SimulationState *>>
      overlapComputeStates = std::nullopt;

  /// @brief Overlap results
  std::optional<std::complex<double>> overlapResult = std::nullopt;

  /// @brief When run under the tracer context, persist the traced quantum
  /// resources here.
  Trace kernelTrace;

  /// @brief The name of the kernel being executed.
  std::string kernelName;

  /// @brief The current iteration for a batch execution, used by `observe_n`
  /// and `sample_n`.
  std::size_t batchIteration = 0;

  /// @brief For batch execution, the total number of batch iterations.
  std::size_t totalIterations = 0;

  /// @brief For mid-circuit measurements in library mode keep track of the
  /// register names.
  std::vector<std::string> registerNames;

  /// @brief A vector containing information about how to reorder the global
  /// register after execution. Empty means no reordering.
  std::vector<std::size_t> reorderIdx;

  /// @brief A buffer containing the return value of a kernel invocation.
  /// Note: this is only needed for invocation not able to return a
  /// `sample_result`.
  std::vector<char> invocationResultBuffer;

  /// @brief The number of trajectories to be used for an expectation
  /// calculation on simulation backends that support trajectory simulation.
  std::optional<std::size_t> numberTrajectories = std::nullopt;

  /// @brief Whether or not to simply concatenate measurements in execution
  /// order.
  bool explicitMeasurements = false;

  /// @brief Probability of occurrence of each error mechanism (column) in
  /// Measurement Syndrome Matrix (0-1 range).
  std::optional<std::vector<double>> msm_probabilities;

  /// @brief Error mechanism ID. From a probability perspective, each error
  /// mechanism ID is independent of all other error mechanism ID. For all
  /// errors with the *same* ID, only one of them can happen. That is - the
  /// errors containing the same ID are correlated with each other.
  std::optional<std::vector<std::size_t>> msm_prob_err_id;

  /// @brief The number of rows and columns of a Measurement Syndrome Matrix.
  /// Note: Measurement Syndrome Matrix is defined in
  /// https://arxiv.org/pdf/2407.13826.
  std::optional<std::pair<std::size_t, std::size_t>> msm_dimensions;

  std::size_t randomSeed = 0;

  std::size_t replay_columns = 0;

  /// @brief Save the current simulation state in the recorded states storage.
  void save_state(SimulationState *state) { recordStorage.save_state(state); }

  /// @brief Get the recorded states saved during execution.
  const std::vector<std::unique_ptr<SimulationState>> &
  get_recorded_states() const {
    return recordStorage.get_recorded_states();
  }

  /// @brief Clear the recorded states saved during execution.
  void clear_recorded_states() { recordStorage.clear(); }

  /// @brief Dump the recorded states saved during execution.
  void dump_recorded_states() const { recordStorage.dump_recorded_states(); }

  void dump_error_data() const { recordStorage.dump_error_data(); }

  void record_error_data(const size_t index, const ErrorByShotLogEntry &entry) {
    recordStorage.record_error_data(index, entry);
  }

  const auto &get_error_data() const { return recordStorage.error_data; }
  void set_error_data(const ErrorLogType &data) {
    recordStorage.error_data = data;
  }
  void update_replay_columns(std::size_t cols) { replay_columns = cols; }
  std::size_t get_replay_columns() const { return replay_columns; }

  void set_seed(std::size_t seed) { randomSeed = seed; }
};
} // namespace cudaq
