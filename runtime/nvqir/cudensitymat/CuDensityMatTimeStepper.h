/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "CuDensityMatState.h"
#include "cudaq/algorithms/base_time_stepper.h"
#include <cudensitymat.h>

namespace cudaq {
class CuDensityMatTimeStepper : public base_time_stepper {
public:
  explicit CuDensityMatTimeStepper(cudensitymatHandle_t handle,
                                   cudensitymatOperator_t liouvillian);

  state compute(const state &inputState, double t,
                const std::unordered_map<std::string, std::complex<double>>
                    &parameters) override;
  void computeImpl(
      cudensitymatState_t inState, cudensitymatState_t outState, double t,
      const std::unordered_map<std::string, std::complex<double>> &parameters,
      int64_t batchSize);

private:
  cudensitymatHandle_t m_handle;
  cudensitymatOperator_t m_liouvillian;
};
} // namespace cudaq
