/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.  *
 ******************************************************************************/

#include "TrajectoryDeduplication.h"
#include "KrausSelection.h"
#include <cstddef>
#include <functional>
#include <unordered_map>
#include <vector>

namespace cudaq::ptsbe {

namespace {

// Referred from `runtime/cudaq/operators/helpers.cpp`
inline void hashCombine(std::size_t &seed, std::size_t value) {
  seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

std::size_t hashKrausSelection(const cudaq::KrausSelection &sel) {
  std::size_t h = std::hash<std::size_t>{}(sel.circuit_location);
  for (std::size_t q : sel.qubits)
    hashCombine(h, std::hash<std::size_t>{}(q));
  hashCombine(h, std::hash<std::string>{}(sel.op_name));
  hashCombine(h, static_cast<std::size_t>(sel.kraus_operator_index));
  return h;
}

} // namespace

std::size_t hashTrajectoryContent(const cudaq::KrausTrajectory &trajectory) {
  std::size_t h = 0;
  for (const auto &sel : trajectory.kraus_selections)
    hashCombine(h, hashKrausSelection(sel));
  return h;
}

std::vector<cudaq::KrausTrajectory>
deduplicateTrajectories(std::span<const cudaq::KrausTrajectory> trajectories) {
  if (trajectories.empty())
    return {};

  std::unordered_map<std::size_t, std::vector<std::size_t>> hash_to_indices;
  std::vector<cudaq::KrausTrajectory> result;
  result.reserve(trajectories.size());

  for (const auto &trajectory : trajectories) {
    std::size_t h = hashTrajectoryContent(trajectory);
    auto &indices = hash_to_indices[h];
    bool found = false;
    for (std::size_t i : indices) {
      if (result[i].kraus_selections == trajectory.kraus_selections) {
        result[i].multiplicity++;
        found = true;
        break;
      }
    }
    if (!found) {
      cudaq::KrausTrajectory rep = trajectory;
      rep.multiplicity = 1;
      indices.push_back(result.size());
      result.push_back(std::move(rep));
    }
  }

  return result;
}

} // namespace cudaq::ptsbe
