/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/operators.h"
#include "cudaq/utils/matrix.h"
#include <CuDensityMatErrorHandling.h>
#include <cmath>
#include <complex>
#include <cudensitymat.h>
#include <iostream>

// Mock cudensitymatHandle_t
inline cudensitymatHandle_t mock_handle() {
  cudensitymatHandle_t handle;
  HANDLE_CUDM_ERROR(cudensitymatCreate(&handle));
  return handle;
}

// Mock Liouvillian operator creation
inline cudensitymatOperator_t mock_liouvillian(cudensitymatHandle_t handle) {
  cudensitymatOperator_t liouvillian = nullptr;
  std::vector<int64_t> dimensions = {2, 2};
  HANDLE_CUDM_ERROR(cudensitymatCreateOperator(
      handle, static_cast<int32_t>(dimensions.size()), dimensions.data(),
      &liouvillian));

  if (!liouvillian)
    throw std::runtime_error("Failed to create mock Liouvillian!");

  return liouvillian;
}

// Mock Hilbert space dimensions
inline std::vector<std::complex<double>> mock_initial_state_data() {
  std::vector<std::complex<double>> data = {
      {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};

  if (data.size() != 4)
    throw std::runtime_error("Mock initial state data has incorrect size!");

  return data;
}

// Mock initial raw state data
inline std::vector<int64_t> mock_hilbert_space_dims() {
  std::vector<int64_t> dims = {2, 2};

  if (dims.empty())
    throw std::runtime_error("Mock Hilbert space dimensions are empty!");

  return dims;
}
