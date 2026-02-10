/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <array>
#include <cassert>
#include <complex>
#include <iterator>
#include <optional>
#include <vector>

namespace Eigen {
// forward declared here so that this header can be used even if the Eigen is
// not used/found
template <typename Scalar_, int Rows_, int Cols_, int Options_, int MaxRows_,
          int MaxCols_>
class Matrix;
} // namespace Eigen

namespace cudaq {

class complex_matrix;

complex_matrix operator*(const complex_matrix &, const complex_matrix &);
std::vector<std::complex<double>>
operator*(const complex_matrix &, const std::vector<std::complex<double>> &);
complex_matrix operator*(std::complex<double>, const complex_matrix &);
complex_matrix operator+(const complex_matrix &, const complex_matrix &);
complex_matrix operator-(const complex_matrix &, const complex_matrix &);
bool operator==(const complex_matrix &, const complex_matrix &);
complex_matrix kronecker(const complex_matrix &, const complex_matrix &);
template <typename Iterable,
          typename T = typename std::iterator_traits<Iterable>::value_type>
complex_matrix kronecker(Iterable begin, Iterable end);

//===----------------------------------------------------------------------===//

/// This is a minimalist matrix container. It is two-dimensional. It owns its
/// data. Elements are of type `complex<double>`. Typically, it will contain a
/// two-by-two set of values.
class complex_matrix {
public:
  using value_type = std::complex<double>;
  using Dimensions = std::pair<std::size_t, std::size_t>;
  using EigenMatrix =
      Eigen::Matrix<value_type, -1, -1, 0x1, -1, -1>; // row major

  enum class order { row_major, column_major };

  complex_matrix() = default;

  // Instantiates a matrix of the given size.
  // All entries are set to zero by default.
  complex_matrix(std::size_t rows, std::size_t cols, bool set_zero = true,
                 order order = order::row_major)
      : dimensions(std::make_pair(rows, cols)),
        data{new value_type[rows * cols]}, internal_order(order) {
    if (set_zero)
      this->set_zero();
  }

  complex_matrix(const complex_matrix &other)
      : dimensions{other.dimensions},
        data{new value_type[get_size(other.dimensions)]},
        internal_order(other.internal_order) {
    std::copy(other.data, other.data + get_size(dimensions), data);
  }

  complex_matrix(const complex_matrix &other, order order);

  complex_matrix(complex_matrix &&other)
      : dimensions{other.dimensions}, data{other.data},
        internal_order(other.internal_order) {
    other.data = nullptr;
  }

  complex_matrix(const std::vector<value_type> &v,
                 const Dimensions &dim = {2, 2}, order order = order::row_major)
      : dimensions{dim}, data{new value_type[get_size(dim)]},
        internal_order(order) {
    check_size(v.size(), dimensions);
    std::copy(v.begin(), v.begin() + get_size(dimensions), data);
  }

  complex_matrix &operator=(const complex_matrix &other) {
    dimensions = other.dimensions;
    data = new value_type[get_size(other.dimensions)];
    std::copy(other.data, other.data + get_size(dimensions), data);
    internal_order = other.internal_order;
    return *this;
  }

  complex_matrix &operator=(complex_matrix &&other) {
    dimensions = other.dimensions;
    data = other.data;
    other.data = nullptr;
    internal_order = other.internal_order;
    return *this;
  }

  /// @brief Return the minimal eigenvalue for this matrix.
  value_type minimal_eigenvalue() const;

  /// @brief Return this matrix's eigenvalues.
  std::vector<value_type> eigenvalues() const;

  /// @brief Return the eigenvectors of this matrix.
  /// They are returned as the rows of a new matrix.
  complex_matrix eigenvectors() const;

  ~complex_matrix() {
    if (data)
      delete[] data;
    data = nullptr;
  }

  //===--------------------------------------------------------------------===//
  // Primitive operations on `complex_matrix` objects.
  //===--------------------------------------------------------------------===//

  /// Multiplication (cross-product) of two matrices.
  friend complex_matrix operator*(const complex_matrix &,
                                  const complex_matrix &);
  complex_matrix &operator*=(const complex_matrix &);

  /// Right-side multiplication with a vector
  friend std::vector<complex_matrix::value_type>
  operator*(const complex_matrix &,
            const std::vector<complex_matrix::value_type> &);

  /// Scalar Multiplication with matrices.
  friend complex_matrix operator*(complex_matrix::value_type,
                                  const complex_matrix &);

  /// Addition of two matrices.
  friend complex_matrix operator+(const complex_matrix &,
                                  const complex_matrix &);
  complex_matrix &operator+=(const complex_matrix &);

  /// Subtraction of two matrices.
  friend complex_matrix operator-(const complex_matrix &,
                                  const complex_matrix &);
  complex_matrix &operator-=(const complex_matrix &);

  /// Kronecker of two matrices.
  friend complex_matrix kronecker(const complex_matrix &,
                                  const complex_matrix &);
  complex_matrix &kronecker_inplace(const complex_matrix &);

  /// Resets the matrix to all zero entries.
  /// Not needed after construction since the matrix will be initialized to
  /// zero.
  void set_zero();

  /// Matrix exponential, uses 20 terms of Taylor Series approximation.
  complex_matrix exponential();

  /// Matrix power.
  complex_matrix power(int powers);

  /// Returns the conjugate transpose of a matrix.
  complex_matrix adjoint();

  /// Returns diagonal elements
  // Index can be used to get super/sub diagonal elements
  std::vector<value_type> diagonal_elements(int index = 0) const;

  /// Return a square identity matrix for the given size.
  static complex_matrix identity(const std::size_t rows);

  /// Kronecker a list of matrices. The list can be any container that has
  /// iterators defined.
  template <typename Iterable, typename T>
  friend complex_matrix kronecker(Iterable begin, Iterable end);

  /// Operator to get the value at a particular index in the matrix.
  complex_matrix::value_type
  operator[](const std::vector<std::size_t> &at) const;

  /// Operator to get the value at a particular index in the matrix.
  complex_matrix::value_type &operator[](const std::vector<std::size_t> &at);

  /// Operator to get the value at a particular index in the matrix.
  complex_matrix::value_type operator()(std::size_t i, std::size_t j) const;

  /// Operator to get the value at a particular index in the matrix.
  complex_matrix::value_type &operator()(std::size_t i, std::size_t j);

  /// @brief Returns a string representation of the matrix
  std::string to_string() const;

  /// @brief Print this matrix to the standard output stream
  void dump() const;

  /// @brief Print this matrix to the given output stream
  void dump(std::ostream &os) const;

  std::size_t get_rank() const { return 2; }
  std::size_t rows() const { return dimensions.first; }
  std::size_t cols() const { return dimensions.second; }
  std::size_t size() const { return get_size(dimensions); }

  const EigenMatrix as_eigen() const;

  complex_matrix::value_type *get_data(order order);

private:
  complex_matrix(const complex_matrix::value_type *v, const Dimensions &dim,
                 order order)
      : dimensions{dim}, data{new complex_matrix::value_type[get_size(dim)]},
        internal_order(order) {
    auto size = get_size(dimensions);
    std::copy(v, v + size, data);
  }

  static std::size_t get_size(const Dimensions &dim) {
    return dim.first * dim.second;
  }

  static void check_size(std::size_t size, const Dimensions &dim);

  void swap(complex_matrix::value_type *new_data) {
    if (data)
      delete[] data;
    data = new_data;
  }

  void clear() {
    if (data)
      delete[] data;
    data = nullptr;
    dimensions = {};
  }

  Dimensions dimensions = {};
  complex_matrix::value_type *data = nullptr;
  complex_matrix::order internal_order = complex_matrix::order::row_major;
};

//===----------------------------------------------------------------------===//

template <typename Iterable, typename T>
complex_matrix kronecker(Iterable begin, Iterable end) {
  complex_matrix result;
  if (begin == end)
    return result;
  result = *begin;
  for (auto i = std::next(begin); i != end; i = std::next(i))
    result.kronecker_inplace(*i);
  return result;
}

inline complex_matrix operator*(const complex_matrix &left,
                                const complex_matrix &right) {
  complex_matrix result = left;
  result *= right;
  return result;
}

inline complex_matrix operator+(const complex_matrix &left,
                                const complex_matrix &right) {
  complex_matrix result = left;
  result += right;
  return result;
}

inline complex_matrix operator-(const complex_matrix &left,
                                const complex_matrix &right) {
  complex_matrix result = left;
  result -= right;
  return result;
}

inline complex_matrix kronecker(const complex_matrix &left,
                                const complex_matrix &right) {
  complex_matrix result = left;
  result.kronecker_inplace(right);
  return result;
}

} // namespace cudaq
