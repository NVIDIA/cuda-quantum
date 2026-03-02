/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "KrausTrajectory.h"
#include "common/NoiseModel.h"
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace cudaq::ptsbe::detail {

/// @brief Noise point information extracted from circuit analysis
///
/// Represents a single location in the circuit where noise can be applied.
/// Stores the validated kraus_channel which contains the Kraus operators and
/// their probabilities for that noise channel.
struct NoisePoint {
  /// @brief Index into a noise location in the PTSBE instruction sequence.
  /// Valid for both gate and measurement noise.
  std::size_t circuit_location;

  /// @brief Qubits affected by this noise
  std::vector<std::size_t> qubits;

  /// @brief Gate operation name ("h", `"cx"`)
  std::string op_name;

  /// @brief Validated noise channel containing Kraus operators and
  /// probabilities
  cudaq::kraus_channel channel;
};

/// @brief Compute total trajectory space with overflow protection
///
/// Calculates the `combinatoric` product of operator counts across all noise
/// points. For N noise points with k_i operators each: total = k_1 × k_2 × ...
/// × k_N
///
/// @param noise_points Noise information from circuit analysis
/// @return Total number of unique trajectories, capped at 2^40 (~1 trillion
/// cap) to prevent overflow
inline std::size_t
computeTotalTrajectories(std::span<const NoisePoint> noise_points) {
  constexpr std::size_t MAX_SAFE = std::size_t(1) << 40;
  std::size_t total = 1;

  for (const auto &np : noise_points) {
    std::size_t count = np.channel.size();
    if (count == 0)
      continue;
    if (total > MAX_SAFE / count)
      return MAX_SAFE;
    total *= count;
  }

  return total;
}

} // namespace cudaq::ptsbe::detail

namespace cudaq::ptsbe {

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
  generateTrajectories(std::span<const detail::NoisePoint> noise_points,
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
