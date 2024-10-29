/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
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
#include <vector>

namespace cudaq {

class matrix_2;

matrix_2 operator*(const matrix_2 &, const matrix_2 &);
matrix_2 operator*(std::complex<double>, const matrix_2 &);
matrix_2 operator+(const matrix_2 &, const matrix_2 &);
matrix_2 operator-(const matrix_2 &, const matrix_2 &);
matrix_2 kronecker(const matrix_2 &, const matrix_2 &);
template <typename Iterable,
          typename T = typename std::iterator_traits<Iterable>::value_type>
matrix_2 kronecker(Iterable begin, Iterable end);

//===----------------------------------------------------------------------===//

/// This is a minimalist matrix container. It is two-dimensional. It owns its
/// data. Elements are of type `complex<double>`. Typically, it will contain a
/// two-by-two set of values.
class matrix_2 {
public:
  using Dimensions = std::pair<std::size_t, std::size_t>;

  matrix_2() = default;
  matrix_2(const matrix_2 &other)
      : dimensions{other.dimensions},
        data{new std::complex<double>[get_size(other.dimensions)]} {
    std::copy(other.data, other.data + get_size(dimensions), data);
  }
  matrix_2(matrix_2 &&other) : dimensions{other.dimensions}, data{other.data} {
    other.data = nullptr;
  }
  matrix_2(const std::vector<std::complex<double>> &v,
           const Dimensions &dim = {2, 2})
      : dimensions{dim}, data{new std::complex<double>[get_size(dim)]} {
    assert(v.size() >= get_size(dimensions) &&
           "vector must have enough elements");
    std::copy(v.begin(), v.begin() + get_size(dimensions), data);
  }
  matrix_2(const std::complex<double> *v, const Dimensions &dim = {2, 2})
      : dimensions{dim}, data{new std::complex<double>[get_size(dim)]} {
    auto size = get_size(dimensions);
    std::copy(v, v + size, data);
  }

  matrix_2 &operator=(const matrix_2 &other) {
    dimensions = other.dimensions;
    data = new std::complex<double>[get_size(other.dimensions)];
    std::copy(other.data, other.data + get_size(dimensions), data);
    return *this;
  }
  matrix_2 &operator=(matrix_2 &&other) {
    dimensions = other.dimensions;
    data = other.data;
    other.data = nullptr;
    return *this;
  }

  ~matrix_2() {
    if (data)
      delete[] data;
    data = nullptr;
  }

  //===--------------------------------------------------------------------===//
  // Primitive operations on `matrix_2` objects.
  //===--------------------------------------------------------------------===//

  /// Multiplication (cross-product) of two matrices.
  friend matrix_2 operator*(const matrix_2 &, const matrix_2 &);
  matrix_2 &operator*=(const matrix_2 &);

  /// Scalar Multiplication with matrices.
  friend matrix_2 operator*(std::complex<double>, const matrix_2 &);

  /// Addition of two matrices.
  friend matrix_2 operator+(const matrix_2 &, const matrix_2 &);
  matrix_2 &operator+=(const matrix_2 &);

  /// Subtraction of two matrices.
  friend matrix_2 operator-(const matrix_2 &, const matrix_2 &);
  matrix_2 &operator-=(const matrix_2 &);

  /// Kronecker of two matrices.
  friend matrix_2 kronecker(const matrix_2 &, const matrix_2 &);
  matrix_2 &kronecker_inplace(const matrix_2 &);

  /// Kronecker a list of matrices. The list can be any container that has
  /// iterators defined.
  template <typename Iterable, typename T>
  friend matrix_2 kronecker(Iterable begin, Iterable end);

  std::string dump() const;

  std::size_t get_rank() const { return 2; }
  std::size_t get_rows() const { return dimensions.first; }
  std::size_t get_columns() const { return dimensions.second; }
  std::size_t get_size() const { return get_size(dimensions); }

private:
  static std::size_t get_size(const Dimensions &dim) {
    return dim.first * dim.second;
  }

  void swap(std::complex<double> *new_data) {
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
  std::complex<double> *data = nullptr;
};

//===----------------------------------------------------------------------===//

template <typename Iterable, typename T>
matrix_2 kronecker(Iterable begin, Iterable end) {
  matrix_2 result;
  if (begin == end)
    return result;
  result = *begin;
  for (auto i = std::next(begin); i != end; i = std::next(i))
    result.kronecker_inplace(*i);
  return result;
}

inline matrix_2 operator*(const matrix_2 &left, const matrix_2 &right) {
  matrix_2 result = left;
  result *= right;
  return result;
}

inline matrix_2 operator+(const matrix_2 &left, const matrix_2 &right) {
  matrix_2 result = left;
  result += right;
  return result;
}

inline matrix_2 operator-(const matrix_2 &left, const matrix_2 &right) {
  matrix_2 result = left;
  result -= right;
  return result;
}

inline matrix_2 kronecker(const matrix_2 &left, const matrix_2 &right) {
  matrix_2 result = left;
  result.kronecker_inplace(right);
  return result;
}

} // namespace cudaq
