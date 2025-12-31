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
#include <optional>
#include <string_view>

namespace cudaq {

/// The ExecutionContext is an abstraction to indicate how a CUDA-Q kernel
/// should be executed.
class ExecutionContext {
public:
  /// @brief The Constructor, takes the name of the context
  /// @param n The name of the context
  ExecutionContext(const std::string &n) : name(n) {}

  /// @brief The constructor, takes the name and the number of shots.
  /// @param n The name of the context
  /// @param shots_ The number of shots
  /// @param qpu_id The ID of the QPU that this execution context is running on.
  ExecutionContext(const std::string &n, std::size_t shots_,
                   std::size_t qpu_id = 0)
      : name(n), shots(shots_), qpuId(qpu_id) {}

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

  /// @brief The ID of the QPU that this execution context is running on.
  std::size_t qpuId = 0;

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
};
} // namespace cudaq
