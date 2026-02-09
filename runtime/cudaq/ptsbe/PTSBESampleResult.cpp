/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PTSBESampleResult.h"
#include <stdexcept>

namespace cudaq::ptsbe {

sample_result::sample_result(cudaq::sample_result &&base)
    : cudaq::sample_result(std::move(base)) {}

sample_result::sample_result(cudaq::sample_result &&base, PTSBETrace trace)
    : cudaq::sample_result(std::move(base)), trace_(std::move(trace)) {}

bool sample_result::has_trace() const { return trace_.has_value(); }

const PTSBETrace &sample_result::trace() const {
  if (!trace_.has_value())
    throw std::runtime_error("PTSBE trace not available. Enable trace output "
                             "with return_trace=true.");
  return trace_.value();
}

void sample_result::set_trace(PTSBETrace trace) { trace_ = std::move(trace); }

} // namespace cudaq::ptsbe
