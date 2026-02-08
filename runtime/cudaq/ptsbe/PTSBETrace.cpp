/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PTSBETrace.h"

namespace cudaq::ptsbe {

std::size_t
PTSBETrace::count_instructions(TraceInstructionType type,
                               std::optional<std::string> name) const {
  std::size_t count = 0;
  for (const auto &inst : instructions)
    if (inst.type == type && (!name || inst.name == *name))
      ++count;
  return count;
}

std::optional<std::reference_wrapper<const cudaq::KrausTrajectory>>
PTSBETrace::get_trajectory(std::size_t trajectoryId) const {
  for (const auto &traj : trajectories) {
    if (traj.trajectory_id == trajectoryId)
      return std::cref(traj);
  }
  return std::nullopt;
}

} // namespace cudaq::ptsbe
