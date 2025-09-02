/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/UnitaryDecomposition.h"
#include <unsupported/Eigen/KroneckerProduct>
#include <unsupported/Eigen/MatrixFunctions>

using namespace std::complex_literals;

namespace cudaq::detail {

/// This logic is based on https://arxiv.org/pdf/quant-ph/9503016 and its
/// corresponding explanation in https://threeplusone.com/pubs/on_gates.pdf,
/// Section 4.
EulerAngles decomposeZYZ(const Eigen::Matrix2cd &matrix) {
  EulerAngles angles;
  /// Rescale the input unitary matrix, `u`, to be special unitary.
  /// Extract a phase factor, `phase`, so that
  /// `determinant(inverse_phase * unitary) = 1`
  auto det = matrix.determinant();
  angles.phase = 0.5 * std::arg(det);
  Eigen::Matrix2cd specialUnitary = std::exp(-1i * angles.phase) * matrix;
  auto abs00 = std::abs(specialUnitary(0, 0));
  auto abs01 = std::abs(specialUnitary(0, 1));
  if (abs00 >= abs01)
    angles.beta = 2.0 * std::acos(abs00);
  else
    angles.beta = 2.0 * std::asin(abs01);
  auto sum =
      std::atan2(specialUnitary(1, 1).imag(), specialUnitary(1, 1).real());
  auto diff =
      std::atan2(specialUnitary(1, 0).imag(), specialUnitary(1, 0).real());
  angles.alpha = sum + diff;
  angles.gamma = sum - diff;

  return angles;
}

/// Helper function to convert a matrix into 'magic' basis
/// M = 1 / sqrt(2) *  1  0  0  i
///                    0  i  1  0
///                    0  i −1  0
///                    1  0  0 −i
const Eigen::Matrix4cd &MagicBasisMatrix() {
  static Eigen::Matrix4cd MagicBasisMatrix;
  MagicBasisMatrix << 1.0, 0.0, 0.0, 1i, 0.0, 1i, 1.0, 0, 0, 1i, -1.0, 0, 1.0,
      0, 0, -1i;
  MagicBasisMatrix = MagicBasisMatrix * M_SQRT1_2;
  return MagicBasisMatrix;
}

/// Helper function to convert a matrix into 'magic' basis
const Eigen::Matrix4cd &MagicBasisMatrixAdj() {
  static Eigen::Matrix4cd MagicBasisMatrixAdj = MagicBasisMatrix().adjoint();
  return MagicBasisMatrixAdj;
}

/// Helper function to extract the coefficients of canonical vector
/// Gamma matrix = +1 +1 −1 +1
///                +1 +1 +1 −1
///                +1 −1 −1 −1
///                +1 −1 +1 +1
const Eigen::Matrix4cd &GammaFactor() {

  static Eigen::Matrix4cd GammaT;
  GammaT << 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, 1;
  GammaT /= 4;
  return GammaT;
}

/// Given an input matrix which is unitary, find two orthogonal matrices, 'left'
/// and 'right', and a diagonal unitary matrix, 'diagonal', such that
/// `input_matrix = left * diagonal * right.transpose()`. This function uses QZ
/// decomposition for this purpose.
/// NOTE: This function may not generate accurate diagonal matrix in some corner
/// cases like degenerate matrices.
std::tuple<Eigen::Matrix4d, Eigen::Matrix4cd, Eigen::Matrix4d>
bidiagonalize(const Eigen::Matrix4cd &matrix) {
  Eigen::Matrix4d real = matrix.real();
  Eigen::Matrix4d imag = matrix.imag();
  Eigen::RealQZ<Eigen::Matrix4d> qz(4);
  qz.compute(real, imag);
  Eigen::Matrix4d left = qz.matrixQ();
  Eigen::Matrix4d right = qz.matrixZ();
  if (left.determinant() < 0.0)
    left.col(0) *= -1.0;
  if (right.determinant() < 0.0)
    right.row(0) *= -1.0;
  Eigen::Matrix4cd diagonal = left.transpose() * matrix * right.transpose();
  assert(diagonal.isDiagonal(TOL));
  return std::make_tuple(left, diagonal, right);
}

/// Separate input matrix into local operations. The input matrix must be
/// special orthogonal. Given a map, SU(2) × SU(2) -> SO(4),
/// map(A, B) = M.adjoint() (A ⊗ B∗) M, find A and B.
std::tuple<Eigen::Matrix2cd, Eigen::Matrix2cd, std::complex<double>>
extractSU2FromSO4(const Eigen::Matrix4cd &matrix) {
  /// Verify input matrix is special orthogonal
  assert(std::abs(std::abs(matrix.determinant()) - 1.0) < TOL);
  assert((matrix * matrix.transpose() - Eigen::Matrix4cd::Identity()).norm() <
         TOL);
  Eigen::Matrix4cd mb = MagicBasisMatrix() * matrix * MagicBasisMatrixAdj();
  /// Use Kronecker factorization
  size_t r = 0;
  size_t c = 0;
  double largest = std::abs(mb(r, c));
  for (size_t i = 0; i < 4; i++)
    for (size_t j = 0; j < 4; j++) {
      if (std::abs(mb(i, j)) >= largest) {
        largest = std::abs(mb(i, j));
        r = i;
        c = j;
      }
    }
  Eigen::Matrix2cd part1 = Eigen::Matrix2cd::Zero();
  Eigen::Matrix2cd part2 = Eigen::Matrix2cd::Zero();
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      part1((r >> 1) ^ i, (c >> 1) ^ j) = mb(r ^ (i << 1), c ^ (j << 1));
      part2((r & 1) ^ i, (c & 1) ^ j) = mb(r ^ i, c ^ j);
    }
  }
  auto det1 = part1.determinant();
  if (std::abs(det1) > TOL)
    part1 /= (std::sqrt(det1));
  auto det2 = part2.determinant();
  if (std::abs(det2) > TOL)
    part2 /= (std::sqrt(det2));
  std::complex<double> phase =
      mb(r, c) / (part1(r >> 1, c >> 1) * part2(r & 1, c & 1));
  if (phase.real() < 0.0) {
    part1 *= -1;
    phase = -phase;
  }
  assert(mb.isApprox(phase * Eigen::kroneckerProduct(part1, part2), TOL));
  assert(part1.isUnitary(TOL) && part2.isUnitary(TOL));
  return std::make_tuple(part1, part2, phase);
}

/// Compute exp(i(x XX + y YY + z ZZ)) matrix for verification
Eigen::Matrix4cd canonicalVecToMatrix(double x, double y, double z) {
  Eigen::Matrix2cd X{Eigen::Matrix2cd::Zero()};
  Eigen::Matrix2cd Y{Eigen::Matrix2cd::Zero()};
  Eigen::Matrix2cd Z{Eigen::Matrix2cd::Zero()};
  X << 0, 1, 1, 0;
  Y << 0, -1i, 1i, 0;
  Z << 1, 0, 0, -1;
  auto XX = Eigen::kroneckerProduct(X, X);
  auto YY = Eigen::kroneckerProduct(Y, Y);
  auto ZZ = Eigen::kroneckerProduct(Z, Z);
  return (1i * (x * XX + y * YY + z * ZZ)).exp();
}

/// This logic is based on the Cartan's KAK decomposition.
/// Ref: https://arxiv.org/pdf/quant-ph/0507171
/// Ref: https://arxiv.org/pdf/0806.4015
KAKComponents decomposeKAK(const Eigen::Matrix4cd &matrix) {
  KAKComponents components;
  /// Step0: Convert to special unitary
  components.phase = std::pow(matrix.determinant(), 0.25);
  auto specialUnitary = matrix / components.phase;
  /// Step1: Convert into magic basis
  Eigen::Matrix4cd matrixMagicBasis =
      MagicBasisMatrixAdj() * specialUnitary * MagicBasisMatrix();
  /// Step2: Diagonalize
  auto [left, diagonal, right] = bidiagonalize(matrixMagicBasis);
  /// Step3: Get the KAK components
  auto [a1, a0, aPh] = extractSU2FromSO4(left);
  components.a0 = a0;
  components.a1 = a1;
  components.phase *= aPh;
  auto [b1, b0, bPh] = extractSU2FromSO4(right);
  components.b0 = b0;
  components.b1 = b1;
  components.phase *= bPh;
  /// Step4: Get the coefficients of canonical class vector
  if (diagonal.determinant().real() < 0.0)
    diagonal(0, 0) *= 1.0;
  Eigen::Vector4cd diagonalPhases;
  for (size_t i = 0; i < 4; i++)
    diagonalPhases(i) = std::arg(diagonal(i, i));
  auto coefficients = GammaFactor() * diagonalPhases;
  components.x = coefficients(1).real();
  components.y = coefficients(2).real();
  components.z = coefficients(3).real();
  components.phase *= std::exp(1i * coefficients(0));
  /// Final check to verify results
  auto canVecToMat =
      canonicalVecToMatrix(components.x, components.y, components.z);
  assert(matrix.isApprox(components.phase * Eigen::kroneckerProduct(a1, a0) *
                             canVecToMat * Eigen::kroneckerProduct(b1, b0),
                         TOL));
  return components;
}

} // namespace cudaq::detail
