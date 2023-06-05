/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "Future.h"
#include "MeasureCounts.h"
#include "NoiseModel.h"
#include "Resources.h"
#include <optional>
#include <string_view>

namespace cudaq {
class spin_op;

// A State is the vector of data for the density matrix or state vector,
// as well as the array shape (n,n) or (n)
using State =
    std::tuple<std::vector<std::size_t>, std::vector<std::complex<double>>>;

/// @brief The ExecutionContext is an abstraction to indicate
/// how a CUDA Quantum kernel should be executed.
class ExecutionContext {
public:
  /// @brief The name of the context ({basic, sampling, observe})
  const std::string name;

  /// @brief The number of execution shots
  std::size_t shots = 0;

  /// @brief An optional spin operator
  std::optional<cudaq::spin_op *> spin;

  /// @brief Measurement counts for a CUDA Quantum kernel invocation
  sample_result result;

  /// @brief A computed expectation value
  std::optional<double> expectationValue = std::nullopt;

  /// @brief The kernel being executed in this context
  /// has conditional statements on measure results.
  bool hasConditionalsOnMeasureResults = false;

  /// @brief Noise model to apply to the
  /// current execution.
  noise_model *noiseModel = nullptr;

  /// @brief Flag to indicate if backend can
  /// handle spin_op observe task under this ExecutionContext.
  bool canHandleObserve = false;

  /// @brief Flag indicating that the current
  /// execution should occur asynchronously
  bool asyncExec = false;

  /// @brief When execution asynchronously, store
  /// the expected results as a cudaq::future here.
  details::future futureResult;

  /// @brief simulationData provides a mechanism for
  /// simulation clients to extract the underlying simulation data.
  State simulationData;

  /// @brief When run under the tracer context, persist the
  /// traced quantum resources here.
  Resources kernelResources;

  /// @brief The name of the kernel being executed.
  std::string kernelName = "";

  /// @brief The current iteration for a batch execution,
  /// used by observe_n and sample_n.
  std::size_t batchIteration = 0;

  /// @brief For batch execution, the total number of
  /// batch iterations.
  std::size_t totalIterations = 0;

  /// @brief For mid-circuit measurements in library mode
  /// keep track of the register names.
  std::vector<std::string> registerNames;

  /// @brief The Constructor, takes the name of the context
  /// @param n The name of the context
  ExecutionContext(const std::string n) : name(n) {}

  /// @brief The constructor, takes the name and the number of shots.
  /// @param n The name of the context
  /// @param shots_ The number of shots
  ExecutionContext(const std::string n, std::size_t shots_)
      : name(n), shots(shots_) {}
  ~ExecutionContext() = default;
};
} // namespace cudaq
