/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/cudaq_json.h"
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

class dem_result {
public:
  dem_result() = default;

  dem_result(std::string dem, M2DSparseMatrix m2d, M2OSparseMatrix m2o,
             std::size_t num_detectors, std::size_t num_observables,
             std::size_t num_measurements, bool matrices_computed,
             cudaq_json annotations = {})
      : dem(std::move(dem)), m2d(std::move(m2d)), m2o(std::move(m2o)),
        num_detectors(num_detectors), num_observables(num_observables),
        num_measurements(num_measurements),
        matrices_computed(matrices_computed),
        annotations(std::move(annotations)) {}

  // Accessors
  const std::string &get_dem() const { return dem; }
  M2DSparseMatrix &&get_m2d() { return std::move(m2d); }
  M2OSparseMatrix &&get_m2o() { return std::move(m2o); }
  std::size_t get_num_detectors() const { return num_detectors; }
  std::size_t get_num_observables() const { return num_observables; }
  std::size_t get_num_measurements() const { return num_measurements; }
  bool get_matrices_computed() const { return matrices_computed; }
  const cudaq_json &get_annotations() const { return annotations; }
  cudaq_json &get_annotations() { return annotations; }

private:
  /// @brief The Detector Error Model (DEM) string.
  std::string dem;

  /// @brief The measurement-to-detector sparse matrix.
  M2DSparseMatrix m2d;

  /// @brief The measurement-to-observable sparse matrix.
  M2OSparseMatrix m2o;

  std::size_t num_detectors = 0;
  std::size_t num_observables = 0;
  std::size_t num_measurements = 0;

  /// @brief True when m2d / m2o were populated; false when the caller opted
  /// out via return_measurement_matrices=False. Distinguishes "not computed"
  /// from "computed but empty (zero-detector circuit)".
  bool matrices_computed = false;

  /// @brief Extensible endpoint metadata. Empty by default; runtime endpoints
  /// may attach information the contract does not otherwise model.
  cudaq_json annotations;
};

} // namespace cudaq
