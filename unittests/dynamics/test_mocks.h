/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/operators.h"
#include "cudaq/utils/tensor.h"
#include <cmath>
#include <complex>
#include <cudaq/cudm_error_handling.h>
#include <cudensitymat.h>
#include <iostream>

// Mock matrix_operator
inline cudaq::matrix_operator mock_matrix_operator(const std::string &op_id,
                                                   int qubit_index) {
  try {
    auto callback = [](std::vector<int> dimensions,
                       std::map<std::string, std::complex<double>>) {
      if (dimensions.size() != 1 || dimensions[0] != 2) {
        throw std::invalid_argument("Invalid dimensions for operator.");
      }

      cudaq::matrix_2 matrix(2, 2);
      matrix[{0, 1}] = 1.0;
      matrix[{1, 0}] = 1.0;
      return matrix;
    };

    cudaq::matrix_operator::define(op_id, {-1}, callback);
  } catch (...) {
  }

  return cudaq::matrix_operator(op_id, {qubit_index});
}

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

  if (!liouvillian) {
    throw std::runtime_error("Failed to create mock Liouvillian!");
  }

  return liouvillian;
}

// Mock Hilbert space dimensions
inline std::vector<std::complex<double>> mock_initial_state_data() {
  std::vector<std::complex<double>> data = {
      {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};

  if (data.size() != 4) {
    throw std::runtime_error("Mock initial state data has incorrect size!");
  }

  return data;
}

// Mock initial raw state data
inline std::vector<int64_t> mock_hilbert_space_dims() {
  std::vector<int64_t> dims = {2, 2};

  if (dims.empty()) {
    throw std::runtime_error("Mock Hilbert space dimensions are empty!");
  }

  return dims;
}
