/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace cudaq {

/// @brief Sparse binary matrix mapping detectors (rows) to measurements
/// (columns). Returned alongside `M2OSparseMatrix` when
/// `return_measurement_matrices=True` is passed to `dem_from_kernel` (Python),
/// or via the `m2d_out` / `m2o_out` reference overloads (C++).
///
/// `rows[d]` lists the chronological measurement indices that contribute to
/// detector `d` (i.e. are XOR-ed together to form its syndrome bit).
/// `num_measurements` gives the total column count (shape is
/// `rows.size() × num_measurements`).
struct M2DSparseMatrix {
  std::size_t num_measurements = 0;
  std::vector<std::vector<std::size_t>> rows;
};

/// @brief Sparse binary matrix mapping observables (rows) to measurements
/// (columns). Returned alongside `M2DSparseMatrix` when
/// `return_measurement_matrices=True` is passed to `dem_from_kernel` (Python),
/// or via the `m2d_out` / `m2o_out` reference overloads (C++).
///
/// `rows[k]` lists the chronological measurement indices that contribute to
/// observable `k`. `num_measurements` gives the total column count (shape is
/// `rows.size() × num_measurements`).
struct M2OSparseMatrix {
  std::size_t num_measurements = 0;
  std::vector<std::vector<std::size_t>> rows;
};

struct dem_result {
  /// @brief The Detector Error Model (DEM) string.
  std::string dem;

  /// @brief The measurement-to-detector sparse matrix.
  M2DSparseMatrix m2d;

  /// @brief The measurement-to-observable sparse matrix.
  M2OSparseMatrix m2o;
};

} // namespace cudaq
