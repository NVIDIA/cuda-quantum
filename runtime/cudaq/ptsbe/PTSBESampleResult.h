/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/SampleResult.h"

namespace cudaq::ptsbe {

/// @brief PTSBE-specific result type returned by `ptsbe::sample()`.
class sample_result : public cudaq::sample_result {
public:
  sample_result() = default;

  /// @brief Construct from a base sample_result (move)
  explicit sample_result(cudaq::sample_result &&base);
};

} // namespace cudaq::ptsbe
