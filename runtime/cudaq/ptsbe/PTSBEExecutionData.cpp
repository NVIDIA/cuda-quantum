/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PTSBEExecutionData.h"
#include <algorithm>

namespace cudaq::ptsbe {

std::size_t numQubits(std::span<const TraceInstruction> trace) {
  std::size_t maxId = 0;
  bool found = false;
  for (const auto &inst : trace) {
    for (auto id : inst.targets) {
      maxId = found ? std::max(maxId, id) : id;
      found = true;
    }
    for (auto id : inst.controls) {
      maxId = found ? std::max(maxId, id) : id;
      found = true;
    }
  }
  return found ? maxId + 1 : 0;
}

std::size_t countInstructions(std::span<const TraceInstruction> trace,
                              TraceInstructionType type,
                              std::optional<std::string> name) {
  std::size_t count = 0;
  for (const auto &inst : trace)
    if (inst.type == type && (!name || inst.name == *name))
      ++count;
  return count;
}

std::size_t
PTSBEExecutionData::count_instructions(TraceInstructionType type,
                                       std::optional<std::string> name) const {
  return countInstructions(instructions, type, std::move(name));
}

std::optional<std::reference_wrapper<const cudaq::KrausTrajectory>>
PTSBEExecutionData::get_trajectory(std::size_t trajectoryId) const {
  for (const auto &traj : trajectories) {
    if (traj.trajectory_id == trajectoryId)
      return std::cref(traj);
  }
  return std::nullopt;
}

} // namespace cudaq::ptsbe
