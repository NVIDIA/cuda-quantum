/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "KrausTrajectory.h"
#include <complex>
#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace cudaq::ptsbe {

/// @brief Noise point information extracted from circuit analysis
///
/// Represents a single location in the circuit where noise can be applied,
/// along with the Kraus operators and their probabilities for that noise
/// channel.
struct NoisePoint {
  /// @brief Location in the circuit (gate index)
  std::size_t circuit_location;

  /// @brief Qubits affected by this noise
  std::vector<std::size_t> qubits;

  /// @brief Gate operation name ("h", `"cx"`)
  std::string op_name;

  /// @brief Kraus operator matrices for this noise channel
  std::vector<std::vector<std::complex<double>>> kraus_operators;

  /// @brief Probabilities for each Kraus operator
  std::vector<double> probabilities;

  /// @brief Check if this is a unitary mixture channel
  /// @return true if all probabilities sum to ~1.0
  [[nodiscard]] bool isUnitaryMixture() const {
    double sum = 0.0;
    for (auto p : probabilities)
      sum += p;
    return std::abs(sum - 1.0) < 1e-9;
  }
};

/// @brief Base class for trajectory sampling strategies
/// The sampling strategy receives processed noise information from the engine
/// and returns a list of unique trajectories to execute.
class PTSSamplingStrategy {
public:
  /// @brief Virtual destructor for polymorphic behavior
  virtual ~PTSSamplingStrategy() = default;

  /// @brief Generate unique trajectories from the noise space
  /// @param noise_points Processed noise information from circuit analysis
  /// @param max_trajectories Maximum number of unique trajectories to generate
  /// @return Vector of unique generated trajectories
  [[nodiscard]] virtual std::vector<cudaq::KrausTrajectory>
  generateTrajectories(std::span<const NoisePoint> noise_points,
                       std::size_t max_trajectories) const = 0;

  /// @brief Get a name for this strategy
  /// @return Strategy name (e.g., "Probabilistic", "Exhaustive")
  [[nodiscard]] virtual const char *name() const = 0;

  /// @brief Clone this strategy
  /// @return Unique pointer to a copy of this strategy
  [[nodiscard]] virtual std::unique_ptr<PTSSamplingStrategy> clone() const = 0;

protected:
  /// @brief Protected default constructor
  PTSSamplingStrategy() = default;

  /// @brief Protected copy constructor
  PTSSamplingStrategy(const PTSSamplingStrategy &) = default;

  /// @brief Protected move constructor
  PTSSamplingStrategy(PTSSamplingStrategy &&) = default;

  /// @brief Protected copy assignment
  PTSSamplingStrategy &operator=(const PTSSamplingStrategy &) = default;

  /// @brief Protected move assignment
  PTSSamplingStrategy &operator=(PTSSamplingStrategy &&) = default;
};

} // namespace cudaq::ptsbe
