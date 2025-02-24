/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "cudaq/qis/state.h"

namespace cudaq {
class TimeStepper {
public:
  virtual ~TimeStepper() = default;

  virtual state
  compute(const state &inputState, double t, double step_size,
          const std::unordered_map<std::string, std::complex<double>>
              &parameters) = 0;
};
} // namespace cudaq