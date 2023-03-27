/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

#include "cudaq/matrix.h"
#include <Eigen/Dense>
#include <iostream>

namespace cudaq {
complex_matrix::complex_matrix(const std::size_t rows, const std::size_t cols)
    : internalData(std::unique_ptr<complex_matrix::value_type>(
          new complex_matrix::value_type[rows * cols])),
      nRows(rows), nCols(cols) {}

complex_matrix::complex_matrix(complex_matrix::value_type *rawData,
                               const std::size_t rows, const std::size_t cols)
    : internalData(std::unique_ptr<complex_matrix::value_type>(rawData)),
      nRows(rows), nCols(cols) {}

void complex_matrix::dump() { dump(std::cout); }
void complex_matrix::dump(std::ostream &os) {
  Eigen::Map<Eigen::MatrixXcd> map(internalData.get(), nRows, nCols);
  os << map << "\n";
}

void complex_matrix::set_zero() {
  Eigen::Map<Eigen::MatrixXcd> map(internalData.get(), nRows, nCols);
  map.setZero();
}

complex_matrix complex_matrix::operator*(complex_matrix &other) {
  Eigen::Map<Eigen::MatrixXcd> map(internalData.get(), nRows, nCols);
  Eigen::Map<Eigen::MatrixXcd> otherMap(other.data(), other.nRows, other.nCols);
  Eigen::MatrixXcd ret = map * otherMap;
  complex_matrix copy(ret.rows(), ret.cols());
  std::memcpy(copy.data(), ret.data(), sizeof(value_type) * ret.size());
  return copy;
}

complex_matrix::value_type &complex_matrix::operator()(std::size_t i,
                                                       std::size_t j) {
  Eigen::Map<Eigen::MatrixXcd> map(internalData.get(), nRows, nCols);
  return map(i, j);
}

std::vector<complex_matrix::value_type> complex_matrix::eigenvalues() {
  Eigen::Map<Eigen::MatrixXcd> map(internalData.get(), nRows, nCols);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> tmp(map);
  auto eigs = tmp.eigenvalues();

  std::vector<complex_matrix::value_type> ret(eigs.size());
  Eigen::VectorXcd::Map(&ret[0], eigs.size()) = eigs;
  return ret;
}

complex_matrix::value_type complex_matrix::minimal_eigenvalue() {
  return eigenvalues()[0];
}

} // namespace cudaq