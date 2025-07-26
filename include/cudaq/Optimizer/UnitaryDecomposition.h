/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/EigenDense.h"
#include <complex>
#include <tuple>

namespace cudaq::detail {

/// Tolerance for numerical comparisons
constexpr double TOL = 1e-7;

/// Result structure for 1-q Euler decomposition in ZYZ basis, including global
/// phase
struct ZYZComponents {
  double alpha;
  double beta;
  double gamma;
  double phase;
};

/**
 * Decompose a single-qubit unitary matrix into ZYZ Euler angles.
 * This decomposes a 2x2 unitary matrix U into:
 * U = exp(i*phase) * Rz(alpha) * Ry(beta) * Rz(gamma)
 *
 * @param matrix A 2x2 unitary matrix to decompose
 * @return ZYZComponents containing alpha, beta, gamma and phase
 */
ZYZComponents decomposeZYZ(const Eigen::Matrix2cd &matrix);

/// Result structure for 2-q KAK decomposition, including global phase
struct KAKComponents {
  Eigen::Matrix2cd a0;
  Eigen::Matrix2cd a1;
  Eigen::Matrix2cd b0;
  Eigen::Matrix2cd b1;
  double x;
  double y;
  double z;
  std::complex<double> phase;
};

/**
 * Decompose a two-qubit unitary matrix using KAK decomposition.
 * This decomposes a 4x4 unitary matrix U into:
 * U = (a1 ⊗ a0) x exp(i(xXX + yYY + zZZ)) x (b1 ⊗ b0)
 *
 * @param matrix A 4x4 unitary matrix to decompose
 * @return KAKComponents containing local operations a0, a1, b0, b1,
 *         interaction terms x, y, z, and global phase
 */
KAKComponents decomposeKAK(const Eigen::Matrix4cd &matrix);

} // namespace cudaq::detail
