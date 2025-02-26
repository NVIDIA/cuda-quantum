/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "CuDensityMatState.h"
#include "cudaq/BaseTimeStepper.h"
#include <cudensitymat.h>

namespace cudaq {
class CuDensityMatTimeStepper : public BaseTimeStepper {
public:
  explicit CuDensityMatTimeStepper(cudensitymatHandle_t handle,
                                   cudensitymatOperator_t liouvillian);

  state compute(const state &inputState, double t, double step_size,
                const std::unordered_map<std::string, std::complex<double>>
                    &parameters) override;

private:
  cudensitymatHandle_t m_handle;
  cudensitymatOperator_t m_liouvillian;
};
} // namespace cudaq