/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "ShotAllocationStrategy.h"
#include "cudaq/algorithms/broadcast.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>

namespace cudaq::ptsbe {

/// @brief Multinomial shot allocation: draw total_shots samples from
/// trajectories weighted by the given weights, incrementing num_shots for
/// each draw.
static void multinomialAllocate(std::span<cudaq::KrausTrajectory> trajectories,
                                const std::vector<double> &weights,
                                std::size_t total_shots, std::uint64_t seed) {
  std::uint64_t resolved_seed =
      seed != 0 ? seed
                : (cudaq::get_random_seed() != 0 ? cudaq::get_random_seed()
                                                 : std::random_device{}());
  std::mt19937_64 rng(resolved_seed);
  std::discrete_distribution<std::size_t> dist(weights.begin(), weights.end());

  for (std::size_t i = 0; i < total_shots; ++i)
    trajectories[dist(rng)].num_shots++;
}

static void allocateUniform(std::span<cudaq::KrausTrajectory> trajectories,
                            std::size_t total_shots);
static void
allocateLowWeightBias(std::span<cudaq::KrausTrajectory> trajectories,
                      std::size_t total_shots, double bias_strength,
                      std::uint64_t seed);
static void
allocateHighWeightBias(std::span<cudaq::KrausTrajectory> trajectories,
                       std::size_t total_shots, double bias_strength,
                       std::uint64_t seed);

void allocateProportional(std::span<cudaq::KrausTrajectory> trajectories,
                          std::size_t total_shots, std::uint64_t seed) {
  std::vector<double> weights;
  weights.reserve(trajectories.size());
  for (const auto &traj : trajectories)
    weights.push_back(traj.probability);

  multinomialAllocate(trajectories, weights, total_shots, seed);
}

void allocateUniform(std::span<cudaq::KrausTrajectory> trajectories,
                     std::size_t total_shots) {
  // Equal shots per trajectory
  std::size_t shots_per_traj = total_shots / trajectories.size();
  std::size_t remainder = total_shots % trajectories.size();

  for (std::size_t i = 0; i < trajectories.size(); ++i) {
    // Base allocation to all, plus 1 extra to first 'remainder' trajectories
    trajectories[i].num_shots = shots_per_traj + (i < remainder ? 1 : 0);
  }
}

void allocateLowWeightBias(std::span<cudaq::KrausTrajectory> trajectories,
                           std::size_t total_shots, double bias_strength,
                           std::uint64_t seed) {
  // Bias toward trajectories with fewer errors
  std::vector<double> weights;
  weights.reserve(trajectories.size());

  for (const auto &traj : trajectories) {
    std::size_t error_count = traj.countErrors();
    // weight = (1 + error_count)^(-bias_strength) * probability
    // Lower error_count → higher weight
    double weight =
        std::pow(1.0 + error_count, -bias_strength) * traj.probability;
    weights.push_back(weight);
  }

  double total_weight = std::accumulate(weights.begin(), weights.end(), 0.0);
  if (total_weight <= 0.0) {
    throw std::invalid_argument(
        "Total weight must be positive for biased allocation");
  }

  multinomialAllocate(trajectories, weights, total_shots, seed);
}

void allocateHighWeightBias(std::span<cudaq::KrausTrajectory> trajectories,
                            std::size_t total_shots, double bias_strength,
                            std::uint64_t seed) {
  // Bias toward trajectories with more errors
  std::vector<double> weights;
  weights.reserve(trajectories.size());

  for (const auto &traj : trajectories) {
    std::size_t error_count = traj.countErrors();
    // weight = (1 + error_count)^(+bias_strength) * probability
    // Higher error_count → higher weight
    double weight =
        std::pow(1.0 + error_count, bias_strength) * traj.probability;
    weights.push_back(weight);
  }

  double total_weight = std::accumulate(weights.begin(), weights.end(), 0.0);
  if (total_weight <= 0.0) {
    throw std::invalid_argument(
        "Total weight must be positive for biased allocation");
  }

  multinomialAllocate(trajectories, weights, total_shots, seed);
}

void allocateMultiplicityWeighted(
    std::span<cudaq::KrausTrajectory> trajectories, std::size_t total_shots,
    std::uint64_t seed) {
  std::vector<double> weights;
  weights.reserve(trajectories.size());
  for (const auto &traj : trajectories)
    weights.push_back(static_cast<double>(traj.multiplicity));

  multinomialAllocate(trajectories, weights, total_shots, seed);
}

void allocateShots(std::span<cudaq::KrausTrajectory> trajectories,
                   std::size_t total_shots,
                   const ShotAllocationStrategy &strategy) {

  if (trajectories.empty()) {
    throw std::invalid_argument(
        "Cannot allocate shots to empty trajectory list");
  }

  if (total_shots == 0) {
    throw std::invalid_argument(
        "Cannot allocate zero shots - total_shots must be positive");
  }

  switch (strategy.type) {
  case ShotAllocationStrategy::Type::PROPORTIONAL:
    allocateProportional(trajectories, total_shots, strategy.seed);
    return;

  case ShotAllocationStrategy::Type::UNIFORM:
    allocateUniform(trajectories, total_shots);
    return;

  case ShotAllocationStrategy::Type::LOW_WEIGHT_BIAS:
    allocateLowWeightBias(trajectories, total_shots, strategy.bias_strength,
                          strategy.seed);
    return;

  case ShotAllocationStrategy::Type::HIGH_WEIGHT_BIAS:
    allocateHighWeightBias(trajectories, total_shots, strategy.bias_strength,
                           strategy.seed);
    return;

  case ShotAllocationStrategy::Type::MULTIPLICITY_WEIGHTED:
    allocateMultiplicityWeighted(trajectories, total_shots, strategy.seed);
    return;
  }
}

} // namespace cudaq::ptsbe
