/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "PTSBEExecutionData.h"
#include "common/SampleResult.h"
#include <optional>

namespace cudaq::ptsbe {

/// @brief PTSBE-specific result type returned by `ptsbe::sample()`
///    which may contain execution data.
class sample_result : public cudaq::sample_result {
private:
  std::optional<PTSBEExecutionData> executionData_;

public:
  sample_result() = default;

  /// @brief Construct from a base sample_result (move)
  explicit sample_result(cudaq::sample_result &&base);

  /// @brief Construct from a base sample_result with execution data
  sample_result(cudaq::sample_result &&base, PTSBEExecutionData executionData);

  /// @brief Check if execution data is available
  bool has_execution_data() const;

  /// @brief Get execution data
  /// @throws std::runtime_error if execution data not available
  const PTSBEExecutionData &execution_data() const;

  /// @brief Attach execution data to this result
  void set_execution_data(PTSBEExecutionData executionData);
};

} // namespace cudaq::ptsbe
