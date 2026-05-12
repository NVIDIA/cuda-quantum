/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PTSBESampleResult.h"
#include "cudaq/runtime/logger/logger.h"
#include <mutex>
#include <stdexcept>

namespace cudaq::ptsbe {
namespace {
std::once_flag ptsbeExecutionDataWarningOnce;

void warnExperimentalExecutionData() {
  std::call_once(ptsbeExecutionDataWarningOnce, []() {
    CUDAQ_WARN(
        "PTSBE execution data API is experimental and may change in a future "
        "release.");
  });
}
} // namespace

sample_result::sample_result(cudaq::sample_result &&base)
    : cudaq::sample_result(std::move(base)) {}

sample_result::sample_result(cudaq::sample_result &&base,
                             PTSBEExecutionData executionData)
    : cudaq::sample_result(std::move(base)),
      executionData_(std::move(executionData)) {}

bool sample_result::has_execution_data() const {
  return executionData_.has_value();
}

const PTSBEExecutionData &sample_result::execution_data() const {
  if (!executionData_.has_value())
    throw std::runtime_error("PTSBE execution data not available. Enable with "
                             "return_execution_data=true.");
  warnExperimentalExecutionData();
  return executionData_.value();
}

void sample_result::set_execution_data(PTSBEExecutionData executionData) {
  executionData_ = std::move(executionData);
}

} // namespace cudaq::ptsbe
