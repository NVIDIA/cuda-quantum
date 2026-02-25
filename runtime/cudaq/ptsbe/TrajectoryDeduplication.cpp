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
  hashCombine(h, sel.kraus_operator_index);
  hashCombine(h, std::hash<bool>{}(sel.is_error));
  return h;
}

struct ContentHash {
  std::size_t operator()(const cudaq::KrausTrajectory &t) const {
    std::size_t h = 0;
    for (const auto &sel : t.kraus_selections)
      hashCombine(h, hashKrausSelection(sel));
    return h;
  }
};

struct ContentEqual {
  bool operator()(const cudaq::KrausTrajectory &a,
                  const cudaq::KrausTrajectory &b) const {
    return a.kraus_selections == b.kraus_selections;
  }
};

} // namespace

std::size_t hashTrajectoryContent(const cudaq::KrausTrajectory &trajectory) {
  return ContentHash{}(trajectory);
}

std::vector<cudaq::KrausTrajectory>
deduplicateTrajectories(std::span<const cudaq::KrausTrajectory> trajectories) {
  if (trajectories.empty())
    return {};

  std::unordered_map<cudaq::KrausTrajectory, std::size_t, ContentHash,
                     ContentEqual>
      content_to_index;
  std::vector<cudaq::KrausTrajectory> result;
  result.reserve(trajectories.size());

  for (const auto &trajectory : trajectories) {
    auto it = content_to_index.find(trajectory);
    if (it != content_to_index.end()) {
      result[it->second].multiplicity += trajectory.multiplicity;
      result[it->second].weight += trajectory.weight;
      result[it->second].num_shots += trajectory.num_shots;
    } else {
      cudaq::KrausTrajectory rep = trajectory;
      rep.multiplicity = trajectory.multiplicity;
      rep.weight = trajectory.weight;
      content_to_index[rep] = result.size();
      result.push_back(std::move(rep));
    }
  }

  return result;
}

} // namespace cudaq::ptsbe
