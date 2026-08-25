/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

namespace cudaq {

/// @brief Options forwarded to
/// `stim::ErrorAnalyzer::circuit_to_detector_error_model` when generating a
/// Detector Error Model (DEM) from a kernel.
struct dem_options {
  /// Decompose hyper-edge error mechanisms into pairs of two-detector edges.
  bool decompose_errors = false;
  /// Fold loop bodies in the circuit for a more compact DEM.
  bool fold_loops = false;
  /// Allow detectors whose parity is not determined by the circuit.
  bool allow_gauge_detectors = false;
  /// Threshold (in [0,1]) for approximating disjoint-error products.
  /// Set to 0 to disable approximation.
  double approximate_disjoint_errors_threshold = 0.0;
  /// When decomposition fails for an error mechanism, insert it into the DEM
  /// undecomposed (as a hyper-edge) instead of raising an exception. Only
  /// relevant when decompose_errors is true.
  bool ignore_decomposition_failures = false;
  /// Prevent the decomposer from introducing remnant edges that would otherwise
  /// be needed to satisfy the decomposition.
  bool block_decomposition_from_introducing_remnant_edges = false;
  /// When true, also compute the measurement-to-detector and
  /// measurement-to-observable matrices.
  bool return_measurement_matrices = false;
};

} // namespace cudaq
