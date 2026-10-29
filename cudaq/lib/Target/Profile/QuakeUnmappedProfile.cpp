/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Target/Profile/QuakeUnmappedProfile.h"
#include "cudaq/Support/Hash.h"

std::size_t std::hash<cudaq::config::QuakeUnmappedProfile>::operator()(
    const cudaq::config::QuakeUnmappedProfile &profile) const noexcept {
  std::size_t seed = 0;
  for (const auto &gate : profile.basisGates)
    cudaq::detail::hashCombine(seed, gate);
  cudaq::detail::hashCombine(seed, profile.approximationTolerance);
  cudaq::detail::hashCombine(seed, profile.costModel);
  return seed;
}
