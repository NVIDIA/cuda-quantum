/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/edem/owned.hpp"
#include "cudaq/gadgets/gadget/gadget.h"
#include "cudaq/gadgets/noise/noise.h"

namespace cudaq::gadgets {
inline namespace v0 {

using EDEM = cudaq::edem::PackedEDEM;

enum class VerificationResult : u32 {
  Proven,
  Counterexample,
  SampledPass,
  Inconclusive,
  Unsupported
};

struct VerificationReport {
  std::string claim;
  std::string method;
  std::string domain;
  std::vector<std::string> assumptions;
  VerificationResult result = VerificationResult::Unsupported;
  std::string evidence;
};

/// Checks the fault-free realization against the logical action.
VerificationReport verify(const Gadget &gadget);

/// The gadget's EDEM under noise, using its profile.
EDEM get_edem(const Gadget &gadget, const NoiseModel &noise);

} // namespace v0
} // namespace cudaq::gadgets
