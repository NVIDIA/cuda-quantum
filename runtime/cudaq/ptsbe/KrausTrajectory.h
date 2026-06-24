/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "KrausSelection.h"
#include "common/SampleResult.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudaq {

inline constexpr double PROBABILITY_EPSILON = 1e-9;

// Forward declaration
class KrausTrajectoryBuilder;

/// @brief Trajectory container for PTSBE execution
/// This struct represents one complete path through the space of possible noise
/// realizations and serves as a container across three execution phases
struct KrausTrajectory {
  /// @brief Unique identifier for this trajectory
  std::size_t trajectory_id = 0;

  /// @brief Complete specification of which Kraus operators to apply at each
  /// noise point. This vector must be ordered by circuit_location in ascending
  /// order. This tracks only the injected noise operators.
  std::vector<KrausSelection> kraus_selections;

  /// @brief Computed probability of this trajectory occurring
  /// This is the product of individual Kraus operator probabilities
  double probability = 0.0;

  /// @brief Number of measurement shots allocated to this trajectory
  std::size_t num_shots = 0;

  /// @brief Number of times this trajectory was drawn in Monte Carlo sampling.
  /// For exhaustive strategies, this is 1 as each trajectory is enumerated
  /// once.
  std::size_t multiplicity = 1;

  /// @brief Allocation weight for shot distribution. PROPORTIONAL and biased
  /// allocation strategies distribute shots proportional to this value.
  /// For Monte Carlo strategies this equals the multiplicity (sample count).
  /// For exhaustive strategies this equals the trajectory probability.
  double weight = 0.0;

  /// @brief The measurement results for this specific trajectory
  CountsDictionary measurement_counts;

  /// @brief Default constructor
  KrausTrajectory() = default;

  /// @brief Constructor for trajectory creation
  /// @param id Unique identifier for this trajectory
  /// @param selections Complete specification of Kraus operators at each noise
  /// point
  /// @param prob Computed probability of this trajectory occurring
  /// @param shots Number of measurement shots allocated to this trajectory
  /// (default: 0)
  KrausTrajectory(std::size_t id, std::vector<KrausSelection> selections,
                  double prob, std::size_t shots = 0)
      : trajectory_id(id), kraus_selections(std::move(selections)),
        probability(prob), num_shots(shots), weight(prob) {}

  /// @brief Create a KrausTrajectoryBuilder
  /// @return KrausTrajectoryBuilder
  [[nodiscard]] static KrausTrajectoryBuilder builder();

  /// @brief Equality comparison for testing
  /// @param other KrausTrajectory to compare with
  /// @return true if trajectory_id, selections, probability, num_shots, and
  /// multiplicity match
  constexpr bool operator==(const KrausTrajectory &other) const {
    return trajectory_id == other.trajectory_id &&
           kraus_selections == other.kraus_selections &&
           std::abs(probability - other.probability) < PROBABILITY_EPSILON &&
           num_shots == other.num_shots && multiplicity == other.multiplicity;
  }

  /// @brief Count non-identity errors in this trajectory (error weight)
  /// @return Number of selections with is_error == true
  [[nodiscard]] constexpr std::size_t countErrors() const {
    return std::ranges::count_if(kraus_selections,
                                 [](const auto &sel) { return sel.is_error; });
  }

  /// @brief Verify that kraus_selections are ordered by circuit_location
  /// @return true if selections are properly ordered (or empty)
  [[nodiscard]] bool isOrdered() const {
    return std::ranges::is_sorted(
        kraus_selections, [](const auto &a, const auto &b) {
          return a.circuit_location < b.circuit_location;
        });
  }
};

/// @brief Builder for Phase 1 trajectory construction
class KrausTrajectoryBuilder {
private:
  std::size_t id_ = 0;
  std::vector<KrausSelection> selections_;
  double probability_ = 0.0;

public:
  /// @brief Set the trajectory identifier
  /// @param id Unique identifier for this trajectory
  /// @return Reference to this builder for chaining
  KrausTrajectoryBuilder &setId(std::size_t id) {
    id_ = id;
    return *this;
  }

  /// @brief Set the Kraus operator selections
  /// @param selections Complete specification of noise operators
  /// @return Reference to this builder for chaining
  KrausTrajectoryBuilder &
  setSelections(std::vector<KrausSelection> selections) {
    selections_ = std::move(selections);
    return *this;
  }

  /// @brief Set the trajectory probability
  /// @param prob Computed probability of this trajectory occurring
  /// @return Reference to this builder for chaining
  KrausTrajectoryBuilder &setProbability(double prob) {
    probability_ = prob;
    return *this;
  }

  /// @brief Build the KrausTrajectory
  /// @return Constructed KrausTrajectory with num_shots = 0
  /// @throws std::logic_error if probability is invalid
  [[nodiscard]] KrausTrajectory build() const {
    // Validate probability
    if (probability_ < 0.0 || probability_ > 1.0) {
      throw std::logic_error("Trajectory probability must be in range [0, 1]");
    }

    return KrausTrajectory(id_, std::vector<KrausSelection>(selections_),
                           probability_, 0);
  }
};

inline KrausTrajectoryBuilder KrausTrajectory::builder() {
  return KrausTrajectoryBuilder();
}

} // namespace cudaq
