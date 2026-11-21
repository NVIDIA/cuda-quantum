/******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 *****************************************************************************/

#include "cudaq/gadgets/analysis/analysis.h"

namespace cudaq::gadgets {
inline namespace v0 {

VerificationReport verify(const Gadget &) {
  // TODO: Implement verification.
  return {};
}

EDEM get_edem(const Gadget &, const NoiseModel &) {
  // TODO: Implement EDEM generation.
  return {};
}

} // namespace v0
} // namespace cudaq::gadgets
