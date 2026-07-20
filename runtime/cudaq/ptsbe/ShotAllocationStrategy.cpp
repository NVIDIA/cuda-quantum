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

namespace {

/// @brief Resolve an optional seed to a concrete value.
std::uint64_t resolveSeed(const std::optional<std::uint64_t> &seed) {
  if (seed.has_value())
    return seed.value();
  auto global = cudaq::get_random_seed();
  return global != 0 ? global : std::random_device{}();
}

/// @brief Multinomial shot allocation: draw total_shots samples from
/// trajectories weighted by the given weights, incrementing num_shots for
/// each draw.
void multinomialAllocate(std::span<cudaq::KrausTrajectory> trajectories,
                         const std::vector<double> &weights,
                         std::size_t total_shots,
                         const std::optional<std::uint64_t> &seed) {
  std::mt19937_64 rng(resolveSeed(seed));
  std::discrete_distribution<std::size_t> dist(weights.begin(), weights.end());

  for (std::size_t i = 0; i < total_shots; ++i)
    trajectories[dist(rng)].num_shots++;
}

void allocateUniform(std::span<cudaq::KrausTrajectory> trajectories,
                     std::size_t total_shots);
void allocateLowWeightBias(std::span<cudaq::KrausTrajectory> trajectories,
                           std::size_t total_shots, double bias_strength,
                           const std::optional<std::uint64_t> &seed);
void allocateHighWeightBias(std::span<cudaq::KrausTrajectory> trajectories,
                            std::size_t total_shots, double bias_strength,
                            const std::optional<std::uint64_t> &seed);

void allocateProportional(std::span<cudaq::KrausTrajectory> trajectories,
                          std::size_t total_shots,
                          const std::optional<std::uint64_t> &seed) {
  std::vector<double> weights;
  weights.reserve(trajectories.size());
  for (const auto &traj : trajectories)
    weights.push_back(traj.weight);

  double total_weight = std::accumulate(weights.begin(), weights.end(), 0.0);
  if (total_weight <= 0.0) {
    throw std::invalid_argument(
        "Total weight must be positive for proportional allocation. "
        "Ensure trajectories have weight set (e.g. by a sampling strategy).");
  }

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
                           const std::optional<std::uint64_t> &seed) {
  // Bias toward trajectories with fewer errors
  std::vector<double> weights;
  weights.reserve(trajectories.size());

  for (const auto &traj : trajectories) {
    std::size_t error_count = traj.countErrors();
    // alloc_weight = (1 + error_count)^(-bias_strength) * trajectory weight
    // Lower error_count -> higher alloc_weight
    double alloc_weight =
        std::pow(1.0 + error_count, -bias_strength) * traj.weight;
    weights.push_back(alloc_weight);
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
                            const std::optional<std::uint64_t> &seed) {
  // Bias toward trajectories with more errors
  std::vector<double> weights;
  weights.reserve(trajectories.size());

  for (const auto &traj : trajectories) {
    std::size_t error_count = traj.countErrors();
    // alloc_weight = (1 + error_count)^(+bias_strength) * trajectory weight
    // Higher error_count -> higher alloc_weight
    double alloc_weight =
        std::pow(1.0 + error_count, bias_strength) * traj.weight;
    weights.push_back(alloc_weight);
  }

  double total_weight = std::accumulate(weights.begin(), weights.end(), 0.0);
  if (total_weight <= 0.0) {
    throw std::invalid_argument(
        "Total weight must be positive for biased allocation");
  }

  multinomialAllocate(trajectories, weights, total_shots, seed);
}

} // namespace

void allocateShots(std::span<cudaq::KrausTrajectory> trajectories,
                   std::size_t total_shots,
                   const ShotAllocationStrategy &strategy) {

  for (auto &traj : trajectories)
    traj.num_shots = 0;

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
  }
}

} // namespace cudaq::ptsbe
