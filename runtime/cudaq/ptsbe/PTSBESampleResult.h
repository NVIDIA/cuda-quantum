/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "PTSBETrace.h"
#include "common/SampleResult.h"
#include <optional>

namespace cudaq::ptsbe {

/// @brief PTSBE-specific result type returned by `ptsbe::sample()`
///    which may contain trace data.
class SampleResult : public cudaq::sample_result {
private:
  std::optional<PTSBETrace> trace_;

public:
  SampleResult() = default;

  /// @brief Construct from a base sample_result (move)
  explicit SampleResult(cudaq::sample_result &&base);

  /// @brief Construct from a base sample_result with trace data
  SampleResult(cudaq::sample_result &&base, PTSBETrace trace);

  /// @brief Check if trace data is available
  bool has_trace() const;

  /// @brief Get trace data
  /// @throws std::runtime_error if trace not available
  const PTSBETrace &trace() const;

  /// @brief Attach trace data to this result
  void set_trace(PTSBETrace trace);
};

} // namespace cudaq::ptsbe
