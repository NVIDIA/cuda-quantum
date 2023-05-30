/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/matrix.h"
#include "common/EigenDense.h"
#include "common/FmtCore.h"
#include <iostream>

namespace cudaq {

/// @brief Hash function for an Eigen::MatrixXcd
struct complex_matrix_hash {
  std::size_t operator()(const Eigen::MatrixXcd &matrix) const {
    size_t seed = 0;
    for (Eigen::Index i = 0; i < matrix.size(); ++i) {
      auto elem = *(matrix.data() + i);
      seed ^= std::hash<double>()(elem.real()) +
              std::hash<double>()(elem.imag()) + 0x9e3779b9 + (seed << 6) +
              (seed >> 2);
    }
    return seed;
  }
};

/// @brief Store eigen solvers for the same matrix so that
/// we don't recompute every time.
std::unordered_map<Eigen::MatrixXcd,
                   Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd>,
                   complex_matrix_hash>
    selfAdjointEigenSolvers;

std::unordered_map<Eigen::MatrixXcd,
                   Eigen::ComplexEigenSolver<Eigen::MatrixXcd>,
                   complex_matrix_hash>
    generalEigenSolvers;

complex_matrix::complex_matrix(const std::size_t rows, const std::size_t cols)
    : internalOwnedData(std::unique_ptr<complex_matrix::value_type>(
          new complex_matrix::value_type[rows * cols])),
      nRows(rows), nCols(cols) {
  internalData = internalOwnedData.get();
}

complex_matrix::complex_matrix(complex_matrix::value_type *rawData,
                               const std::size_t rows, const std::size_t cols)
    : nRows(rows), nCols(cols) {
  internalData = rawData;
}

complex_matrix::value_type *complex_matrix::data() const {
  return internalData;
}

void complex_matrix::dump() { dump(std::cout); }
void complex_matrix::dump(std::ostream &os) {
  Eigen::Map<Eigen::MatrixXcd> map(internalData, nRows, nCols);
  os << map << "\n";
}

void complex_matrix::set_zero() {
  Eigen::Map<Eigen::MatrixXcd> map(internalData, nRows, nCols);
  map.setZero();
}

complex_matrix complex_matrix::operator*(complex_matrix &other) const {
  Eigen::Map<Eigen::MatrixXcd> map(internalData, nRows, nCols);
  Eigen::Map<Eigen::MatrixXcd> otherMap(other.data(), other.nRows, other.nCols);
  Eigen::MatrixXcd ret = map * otherMap;
  complex_matrix copy(ret.rows(), ret.cols());
  std::memcpy(copy.data(), ret.data(), sizeof(value_type) * ret.size());
  return copy;
}

complex_matrix complex_matrix::operator*(std::vector<value_type> &other) const {
  if (nCols != other.size())
    throw std::runtime_error(
        fmt::format("Invalid vector<T> size for complex_matrix matrix-vector "
                    "product ({} != {}).",
                    nCols, other.size()));

  Eigen::Map<Eigen::MatrixXcd> map(internalData, nRows, nCols);
  Eigen::Map<Eigen::VectorXcd> otherMap(other.data(), other.size());

  Eigen::MatrixXcd ret = map * otherMap;
  complex_matrix copy(ret.rows(), ret.cols());
  std::memcpy(copy.data(), ret.data(), sizeof(value_type) * ret.size());
  return copy;
}

complex_matrix::value_type &complex_matrix::operator()(std::size_t i,
                                                       std::size_t j) const {
  Eigen::Map<Eigen::MatrixXcd> map(internalData, nRows, nCols);
  return map(i, j);
}

std::vector<complex_matrix::value_type> complex_matrix::eigenvalues() const {
  Eigen::Map<Eigen::MatrixXcd> map(internalData, nRows, nCols);
  if (map.isApprox(map.adjoint())) {
    auto iter = selfAdjointEigenSolvers.find(map);
    if (iter == selfAdjointEigenSolvers.end())
      selfAdjointEigenSolvers.emplace(
          map, Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd>(map));

    auto eigs = selfAdjointEigenSolvers[map].eigenvalues();
    std::vector<complex_matrix::value_type> ret(eigs.size());
    Eigen::VectorXcd::Map(&ret[0], eigs.size()) = eigs;
    return ret;
  }

  // This matrix is not self adjoint, use the ComplexEigenSolver
  auto iter = generalEigenSolvers.find(map);
  if (iter == generalEigenSolvers.end())
    generalEigenSolvers.emplace(
        map, Eigen::ComplexEigenSolver<Eigen::MatrixXcd>(map));

  auto eigs = generalEigenSolvers[map].eigenvalues();
  std::vector<complex_matrix::value_type> ret(eigs.size());
  Eigen::VectorXcd::Map(&ret[0], eigs.size()) = eigs;
  return ret;
}

complex_matrix complex_matrix::eigenvectors() const {
  Eigen::Map<Eigen::MatrixXcd> map(internalData, nRows, nCols);

  if (map.isApprox(map.adjoint())) {
    auto iter = selfAdjointEigenSolvers.find(map);
    if (iter == selfAdjointEigenSolvers.end())
      selfAdjointEigenSolvers.emplace(
          map, Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd>(map));

    auto eigv = selfAdjointEigenSolvers[map].eigenvectors();
    complex_matrix copy(eigv.rows(), eigv.cols());
    std::memcpy(copy.data(), eigv.data(), sizeof(value_type) * eigv.size());
    return copy;
  }

  // This matrix is not self adjoint, use the ComplexEigenSolver
  auto iter = generalEigenSolvers.find(map);
  if (iter == generalEigenSolvers.end())
    generalEigenSolvers.emplace(
        map, Eigen::ComplexEigenSolver<Eigen::MatrixXcd>(map));

  auto eigv = generalEigenSolvers[map].eigenvectors();
  complex_matrix copy(eigv.rows(), eigv.cols());
  std::memcpy(copy.data(), eigv.data(), sizeof(value_type) * eigv.size());
  return copy;
}

complex_matrix::value_type complex_matrix::minimal_eigenvalue() const {
  return eigenvalues()[0];
}

} // namespace cudaq