/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/base_time_stepper.h"
#include "cudaq/cudm_state.h"
#include <cudensitymat.h>

namespace cudaq {
class cudm_time_stepper : public BaseTimeStepper<cudm_state> {
public:
  explicit cudm_time_stepper(cudensitymatHandle_t handle,
                             cudensitymatOperator_t liouvillian);

  cudm_state compute(cudm_state &state, double t, double step_size);

private:
  cudensitymatHandle_t handle_;
  cudensitymatOperator_t liouvillian_;
};
} // namespace cudaq