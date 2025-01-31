/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cmath>
#include <complex>
#include <cudaq/cudm_error_handling.h>
#include <cudensitymat.h>
#include <iostream>

// Mock Liouvillian operator creation
inline cudensitymatOperator_t mock_liouvillian(cudensitymatHandle_t handle) {
  cudensitymatOperator_t liouvillian;
  std::vector<int64_t> dimensions = {2, 2};
  HANDLE_CUDM_ERROR(cudensitymatCreateOperator(
      handle, static_cast<int32_t>(dimensions.size()), dimensions.data(),
      &liouvillian));
  return liouvillian;
}

// Mock Hilbert space dimensions
inline std::vector<std::complex<double>> mock_initial_state_data() {
  return {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
}

// Mock initial raw state data
inline std::vector<int64_t> mock_hilbert_space_dims() { return {2, 2}; }
