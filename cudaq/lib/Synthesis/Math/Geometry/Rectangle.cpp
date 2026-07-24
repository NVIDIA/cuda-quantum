/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Math/Geometry/Rectangle.h"
#include "cudaq/Synthesis/Math/Real.h"

#include <algorithm>

namespace cudaq::synth {

std::optional<std::pair<Real, Real>>
Rectangle::intersect(const DOmega &u0, const DOmega &v) const {
  // Rectangle ray intersection is currently unimplemented. The TDGP pipeline
  // does not reach this path -- the existing convex sets in play are the
  // EpsilonRegion and the UnitDisk, both of which override intersect()
  // directly. This stub aborts so accidental callers fail loudly rather
  // than silently returning a wrong answer.
  //
  // A working implementation should walk both axis intervals and intersect
  // the per-axis t-ranges; see the commented-out sketch in the file history
  // for one possible structure.
  exit(1);
}

} // namespace cudaq::synth
