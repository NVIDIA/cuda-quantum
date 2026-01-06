/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/utils/matrix.h"
#include <cmath>
#include <iostream>
#include <sstream>

#include <Eigen/Dense>

// tools for caching eigensolvers

/// @brief Hash function for an cudaq::complex_matrix::EigenMatrix
struct complex_matrix_hash {
  std::size_t
  operator()(const cudaq::complex_matrix::EigenMatrix &matrix) const {
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

std::unordered_map<
    cudaq::complex_matrix::EigenMatrix,
    Eigen::SelfAdjointEigenSolver<cudaq::complex_matrix::EigenMatrix>,
    complex_matrix_hash>
    selfAdjointEigenSolvers;

std::unordered_map<
    cudaq::complex_matrix::EigenMatrix,
    Eigen::ComplexEigenSolver<cudaq::complex_matrix::EigenMatrix>,
    complex_matrix_hash>
    generalEigenSolvers;

// matrix implementation

inline cudaq::complex_matrix::value_type &
access(cudaq::complex_matrix::value_type *p,
       cudaq::complex_matrix::Dimensions sizes, std::size_t row,
       std::size_t col, cudaq::complex_matrix::order internal_order) {
  return internal_order == cudaq::complex_matrix::order::row_major
             ? p[row * sizes.second + col]
             : p[col * sizes.first + row];
}

cudaq::complex_matrix::complex_matrix(const cudaq::complex_matrix &other,
                                      order order)
    : dimensions{other.dimensions},
      data{new value_type[get_size(other.dimensions)]}, internal_order(order) {
  if (order == other.internal_order)
    std::copy(other.data, other.data + get_size(dimensions), data);
  else {
    for (std::size_t i = 0; i < this->rows(); i++)
      for (std::size_t j = 0; j < this->cols(); j++)
        access(data, this->dimensions, i, j, order) =
            access(other.data, other.dimensions, i, j, other.internal_order);
  }
}

cudaq::complex_matrix::value_type *
cudaq::complex_matrix::get_data(cudaq::complex_matrix::order order) {
  if (order != this->internal_order) {
    auto new_data =
        new cudaq::complex_matrix::value_type[this->rows() * this->cols()];
    for (std::size_t i = 0; i < this->rows(); i++)
      for (std::size_t j = 0; j < this->cols(); j++)
        access(new_data, this->dimensions, i, j, order) =
            access(this->data, this->dimensions, i, j, this->internal_order);
    swap(new_data);
    this->internal_order = order;
  }
  return this->data;
}

const cudaq::complex_matrix::EigenMatrix
cudaq::complex_matrix::as_eigen() const {
  if (this->internal_order == cudaq::complex_matrix::order::row_major)
    return Eigen::Map<
        Eigen::Matrix<value_type, -1, -1, Eigen::RowMajor, -1, -1>>(
        this->data, this->dimensions.first, this->dimensions.second);
  else {
    assert(this->internal_order == cudaq::complex_matrix::order::column_major);
    return Eigen::Map<
        Eigen::Matrix<value_type, -1, -1, Eigen::ColMajor, -1, -1>>(
        this->data, this->dimensions.first, this->dimensions.second);
  }
}

cudaq::complex_matrix::value_type
cudaq::complex_matrix::minimal_eigenvalue() const {
  return eigenvalues()[0];
}

std::vector<cudaq::complex_matrix::value_type>
cudaq::complex_matrix::eigenvalues() const {
  auto map = this->as_eigen();
  if (map.isApprox(map.adjoint())) {
    auto iter = selfAdjointEigenSolvers.find(map);
    if (iter == selfAdjointEigenSolvers.end())
      selfAdjointEigenSolvers.emplace(
          map,
          Eigen::SelfAdjointEigenSolver<cudaq::complex_matrix::EigenMatrix>(
              map));

    auto eigs = selfAdjointEigenSolvers[map].eigenvalues();
    std::vector<cudaq::complex_matrix::value_type> ret(eigs.size());
    Eigen::VectorXcd::Map(&ret[0], eigs.size()) = eigs;
    return ret;
  }

  // This matrix is not self adjoint, use the ComplexEigenSolver
  auto iter = generalEigenSolvers.find(map);
  if (iter == generalEigenSolvers.end())
    generalEigenSolvers.emplace(
        map,
        Eigen::ComplexEigenSolver<cudaq::complex_matrix::EigenMatrix>(map));

  auto eigs = generalEigenSolvers[map].eigenvalues();
  std::vector<cudaq::complex_matrix::value_type> ret(eigs.size());
  Eigen::VectorXcd::Map(&ret[0], eigs.size()) = eigs;
  return ret;
}

cudaq::complex_matrix cudaq::complex_matrix::eigenvectors() const {
  auto map = this->as_eigen();

  if (map.isApprox(map.adjoint())) {
    auto iter = selfAdjointEigenSolvers.find(map);
    if (iter == selfAdjointEigenSolvers.end())
      selfAdjointEigenSolvers.emplace(
          map,
          Eigen::SelfAdjointEigenSolver<cudaq::complex_matrix::EigenMatrix>(
              map));

    auto eigv = selfAdjointEigenSolvers[map].eigenvectors();
    cudaq::complex_matrix copy(eigv.rows(), eigv.cols(), false);
    std::memcpy(copy.data, eigv.data(),
                sizeof(cudaq::complex_matrix::value_type) * eigv.size());
    return copy;
  }

  // This matrix is not self adjoint, use the ComplexEigenSolver
  auto iter = generalEigenSolvers.find(map);
  if (iter == generalEigenSolvers.end())
    generalEigenSolvers.emplace(
        map,
        Eigen::ComplexEigenSolver<cudaq::complex_matrix::EigenMatrix>(map));

  auto eigv = generalEigenSolvers[map].eigenvectors();
  cudaq::complex_matrix copy(eigv.rows(), eigv.cols(), false);
  std::memcpy(copy.data, eigv.data(),
              sizeof(cudaq::complex_matrix::value_type) * eigv.size());
  return copy;
}

cudaq::complex_matrix &
cudaq::complex_matrix::operator*=(const cudaq::complex_matrix &right) {
  if (cols() != right.rows())
    throw std::runtime_error("matrix dimensions mismatch in operator*=");

  auto new_data = new cudaq::complex_matrix::value_type[rows() * right.cols()];
  cudaq::complex_matrix::Dimensions new_dims = {rows(), right.cols()};
  for (std::size_t i = 0; i < rows(); i++)
    for (std::size_t j = 0; j < right.cols(); j++)
      for (std::size_t k = 0; k < cols(); k++)
        access(new_data, new_dims, i, j, this->internal_order) +=
            access(data, dimensions, i, k, this->internal_order) *
            access(right.data, right.dimensions, k, j, right.internal_order);
  swap(new_data);
  dimensions = new_dims;
  return *this;
}

std::vector<cudaq::complex_matrix::value_type>
cudaq::operator*(const cudaq::complex_matrix &matrix,
                 const std::vector<cudaq::complex_matrix::value_type> &vect) {
  if (matrix.cols() != vect.size())
    throw std::runtime_error("size mismatch for vector multiplication - "
                             "expecting a vector of length " +
                             std::to_string(matrix.cols()));
  // Initialized res to the correct size and set all elements to 0
  std::vector<cudaq::complex_matrix::value_type> res(matrix.rows());
  for (std::size_t i = 0; i < matrix.rows(); i++)
    for (std::size_t j = 0; j < matrix.cols(); j++)
      res[i] += matrix(i, j) * vect[j];
  return res;
}

cudaq::complex_matrix cudaq::operator*(cudaq::complex_matrix::value_type scalar,
                                       const cudaq::complex_matrix &right) {
  auto new_data =
      new cudaq::complex_matrix::value_type[right.rows() * right.cols()];
  for (std::size_t i = 0; i < right.rows(); i++)
    for (std::size_t j = 0; j < right.cols(); j++)
      access(new_data, right.dimensions, i, j, right.internal_order) =
          scalar *
          access(right.data, right.dimensions, i, j, right.internal_order);
  return {new_data, right.dimensions, right.internal_order};
}

cudaq::complex_matrix &
cudaq::complex_matrix::operator+=(const cudaq::complex_matrix &right) {
  if (!(rows() == right.rows() && cols() == right.cols()))
    throw std::runtime_error("matrix dimensions mismatch in operator+=");

  for (std::size_t i = 0; i < rows(); i++)
    for (std::size_t j = 0; j < cols(); j++)
      access(data, dimensions, i, j, this->internal_order) +=
          access(right.data, right.dimensions, i, j, right.internal_order);
  return *this;
}

cudaq::complex_matrix &
cudaq::complex_matrix::operator-=(const cudaq::complex_matrix &right) {
  if (!(rows() == right.rows() && cols() == right.cols()))
    throw std::runtime_error("matrix dimensions mismatch in operator-=");

  for (std::size_t i = 0; i < rows(); i++)
    for (std::size_t j = 0; j < cols(); j++)
      access(data, dimensions, i, j, this->internal_order) -=
          access(right.data, right.dimensions, i, j, right.internal_order);
  return *this;
}

bool cudaq::operator==(const cudaq::complex_matrix &lhs,
                       const cudaq::complex_matrix &rhs) {
  if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols())
    return false;
  for (std::size_t i = 0; i < lhs.rows(); i++) {
    for (std::size_t j = 0; j < lhs.cols(); j++) {
      if (lhs[{i, j}] != rhs[{i, j}])
        return false;
    }
  }
  return true;
}

cudaq::complex_matrix &
cudaq::complex_matrix::kronecker_inplace(const cudaq::complex_matrix &right) {
  Dimensions new_dim{rows() * right.rows(), cols() * right.cols()};
  auto new_data = new cudaq::complex_matrix::value_type[rows() * right.rows() *
                                                        cols() * right.cols()];
  for (std::size_t i = 0; i < rows(); i++)
    for (std::size_t k = 0; k < right.rows(); k++)
      for (std::size_t j = 0; j < cols(); j++)
        for (std::size_t m = 0; m < right.cols(); m++)
          access(new_data, new_dim, right.rows() * i + k, right.cols() * j + m,
                 this->internal_order) =
              access(data, dimensions, i, j, this->internal_order) *
              access(right.data, right.dimensions, k, m, right.internal_order);
  swap(new_data);
  dimensions = new_dim;
  return *this;
}

void cudaq::complex_matrix::check_size(std::size_t size,
                                       const Dimensions &dim) {
  if (size != get_size(dim))
    throw std::runtime_error("mismatch between data and dimensions");
}

cudaq::complex_matrix::value_type
cudaq::complex_matrix::operator[](const std::vector<std::size_t> &at) const {
  if (at.size() != 2)
    throw std::runtime_error("Invalid access: indices must have length of 2");

  if (at[0] >= rows() || at[1] >= cols())
    throw std::runtime_error(
        "Invalid access: indices {" + std::to_string(at[0]) + ", " +
        std::to_string(at[1]) + "} are larger than matrix dimensions: {" +
        std::to_string(dimensions.first) + ", " +
        std::to_string(dimensions.second) + "}");
  return access(data, dimensions, at[0], at[1], this->internal_order);
}

cudaq::complex_matrix::value_type &
cudaq::complex_matrix::operator[](const std::vector<std::size_t> &at) {
  if (at.size() != 2)
    throw std::runtime_error("Invalid access: indices must have length of 2");

  if (at[0] >= rows() || at[1] >= cols())
    throw std::runtime_error(
        "Invalid access: indices {" + std::to_string(at[0]) + ", " +
        std::to_string(at[1]) + "} are larger than matrix dimensions: {" +
        std::to_string(dimensions.first) + ", " +
        std::to_string(dimensions.second) + "}");
  return access(data, dimensions, at[0], at[1], this->internal_order);
}

cudaq::complex_matrix::value_type
cudaq::complex_matrix::operator()(std::size_t i, std::size_t j) const {
  if (i >= rows() || j >= cols())
    throw std::runtime_error("Invalid access: indices {" + std::to_string(i) +
                             ", " + std::to_string(j) +
                             "} are larger than matrix dimensions: {" +
                             std::to_string(dimensions.first) + ", " +
                             std::to_string(dimensions.second) + "}");
  return access(data, dimensions, i, j, this->internal_order);
}

cudaq::complex_matrix::value_type &
cudaq::complex_matrix::operator()(std::size_t i, std::size_t j) {
  if (i >= rows() || j >= cols())
    throw std::runtime_error("Invalid access: indices {" + std::to_string(i) +
                             ", " + std::to_string(j) +
                             "} are larger than matrix dimensions: {" +
                             std::to_string(dimensions.first) + ", " +
                             std::to_string(dimensions.second) + "}");
  return access(data, dimensions, i, j, this->internal_order);
}

std::string cudaq::complex_matrix::to_string() const {
  std::stringstream out;
  dump(out);
  return out.str();
}

void cudaq::complex_matrix::dump() const { dump(std::cout); }

void cudaq::complex_matrix::dump(std::ostream &os) const {
  auto map = this->as_eigen();
  os << map << "\n";
}

// Calculate the power of a given matrix, `powers` times.
cudaq::complex_matrix cudaq::complex_matrix::power(int powers) {
  // Initialize as identity.
  if (rows() != cols())
    throw std::runtime_error("Matrix power expects a square matrix.");
  auto result = cudaq::complex_matrix::identity(rows());

  // Calculate the matrix power iteratively.
  for (std::size_t i = 0; i < (std::size_t)powers; i++)
    result = result * *this;
  return result;
}

void cudaq::complex_matrix::set_zero() {
  auto size = rows() * cols();
  for (std::size_t i = 0; i < size; ++i)
    data[i] = 0.;
}

// Calculate the Taylor approximation to the exponential of the given matrix.
cudaq::complex_matrix cudaq::complex_matrix::exponential() {
  auto factorial = [](std::size_t value) {
    std::size_t res = 1;
    for (std::size_t factor = 2; factor <= value; ++factor)
      res *= factor;
    return (double)res;
  };

  std::size_t rows = this->rows();
  std::size_t columns = this->cols();
  if (rows != columns)
    throw std::runtime_error("Matrix exponential expects a square matrix.");
  auto result = cudaq::complex_matrix(rows, columns, false);
  // Taylor Series Approximation, fixed at 20 steps.
  std::size_t taylor_steps = 20;
  for (std::size_t step = 0; step < taylor_steps; step++) {
    auto term = this->power(step);
    for (std::size_t i = 0; i < rows; i++) {
      for (std::size_t j = 0; j < columns; j++) {
        result[{i, j}] += term[{i, j}] / factorial(step);
      }
    }
  }
  return result;
}

cudaq::complex_matrix cudaq::complex_matrix::identity(const std::size_t rows) {
  auto result = cudaq::complex_matrix(rows, rows);
  for (std::size_t i = 0; i < rows; i++)
    result[{i, i}] = 1.;
  return result;
}

// Transpose + Conjugate
cudaq::complex_matrix cudaq::complex_matrix::adjoint() {
  std::size_t rows = this->rows();
  std::size_t cols = this->cols();
  cudaq::complex_matrix result(cols, rows, false);

  for (std::size_t i = 0; i < rows; i++) {
    for (std::size_t j = 0; j < cols; j++) {
      result[{j, i}] = std::conj((*this)[{i, j}]);
    }
  }

  return result;
}

std::vector<cudaq::complex_matrix::value_type>
cudaq::complex_matrix::diagonal_elements(int index) const {
  auto map = this->as_eigen();
  auto diag = map.diagonal(index);
  std::vector<cudaq::complex_matrix::value_type> result(diag.begin(),
                                                        diag.end());
  return result;
}
