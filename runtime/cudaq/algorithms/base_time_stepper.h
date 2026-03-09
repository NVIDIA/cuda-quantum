/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "cudaq/qis/state.h"
#include <unordered_map>

namespace cudaq {
class base_time_stepper {
public:
  virtual ~base_time_stepper() = default;

  virtual state
  compute(const state &inputState, double t,
          const std::unordered_map<std::string, std::complex<double>>
              &parameters) = 0;
};
} // namespace cudaq
