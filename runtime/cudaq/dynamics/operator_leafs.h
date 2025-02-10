/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <complex>
#include <map>
#include <variant>
#include <vector>

#include "callback.h"
#include "cudaq/utils/tensor.h"

namespace cudaq {

class scalar_operator {

private:
  // If someone gave us a constant value, we will just return that
  // directly to them when they call `evaluate`.
  std::variant<std::complex<double>, ScalarCallbackFunction> value;

public:

  // constructors and destructors

  scalar_operator(double value);

  /// @brief Constructor that just takes and returns a complex double value.
  /// @NOTE: This replicates the behavior of the python `scalar_operator::const`
  /// without the need for an extra member function.
  scalar_operator(std::complex<double> value);

  scalar_operator(const ScalarCallbackFunction &create);

  /// @brief Constructor that just takes a callback function with no
  /// arguments.
  scalar_operator(ScalarCallbackFunction &&create);

  // copy constructor
  scalar_operator(const scalar_operator &other);

  // move constructor
  scalar_operator(scalar_operator &&other);

  ~scalar_operator() = default;

  // assignments

  // assignment operator
  scalar_operator& operator=(const scalar_operator &other);

  // move assignment operator
  scalar_operator& operator=(scalar_operator &&other);

  // evaluations

  /// @brief Return the scalar operator as a concrete complex value.
  std::complex<double>
  evaluate(const std::map<std::string, std::complex<double>> parameters = {}) const;

  // Return the scalar operator as a 1x1 matrix. This is needed for
  // compatibility with the other inherited classes.
  matrix_2 to_matrix(const std::map<int, int> dimensions = {},
                     const std::map<std::string, std::complex<double>> parameters = {}) const;

  // comparisons

  bool operator==(scalar_operator other);

  // unary operators

  scalar_operator operator-() const;
  scalar_operator operator+() const;

  // right-hand arithmetics

  scalar_operator operator*(double other) const;
  scalar_operator operator/(double other) const;
  scalar_operator operator+(double other) const;
  scalar_operator operator-(double other) const;
  scalar_operator& operator*=(double other);
  scalar_operator& operator/=(double other);
  scalar_operator& operator+=(double other);
  scalar_operator& operator-=(double other);
  scalar_operator operator*(std::complex<double> other) const;
  scalar_operator operator/(std::complex<double> other) const;
  scalar_operator operator+(std::complex<double> other) const;
  scalar_operator operator-(std::complex<double> other) const;
  scalar_operator& operator*=(std::complex<double> other);
  scalar_operator& operator/=(std::complex<double> other);
  scalar_operator& operator+=(std::complex<double> other);
  scalar_operator& operator-=(std::complex<double> other);
  scalar_operator operator*(const scalar_operator &other) const;
  scalar_operator operator/(const scalar_operator &other) const;
  scalar_operator operator+(const scalar_operator &other) const;
  scalar_operator operator-(const scalar_operator &other) const;
  scalar_operator& operator*=(const scalar_operator &other);
  scalar_operator& operator/=(const scalar_operator &other);
  scalar_operator& operator+=(const scalar_operator &other);
  scalar_operator& operator-=(const scalar_operator &other);

  friend scalar_operator operator*(scalar_operator &&self, double other);
  friend scalar_operator operator/(scalar_operator &&self, double other);
  friend scalar_operator operator+(scalar_operator &&self, double other);
  friend scalar_operator operator-(scalar_operator &&self, double other);
  friend scalar_operator operator+(scalar_operator &&self, std::complex<double> other);
  friend scalar_operator operator/(scalar_operator &&self, std::complex<double> other);
  friend scalar_operator operator+(scalar_operator &&self, std::complex<double> other);
  friend scalar_operator operator-(scalar_operator &&self, std::complex<double> other);

  // left-hand arithmetics

  friend scalar_operator operator*(double other, const scalar_operator &self);
  friend scalar_operator operator/(double other, const scalar_operator &self);
  friend scalar_operator operator+(double other, const scalar_operator &self);
  friend scalar_operator operator-(double other, const scalar_operator &self);
  friend scalar_operator operator*(std::complex<double> other, const scalar_operator &self);
  friend scalar_operator operator/(std::complex<double> other, const scalar_operator &self);
  friend scalar_operator operator+(std::complex<double> other, const scalar_operator &self);
  friend scalar_operator operator-(std::complex<double> other, const scalar_operator &self);
};

class operator_handler {
public:
  virtual ~operator_handler() = default;

  virtual std::vector<int> degrees() const = 0;

  virtual bool is_identity() const = 0;

  /// @brief Return the `matrix_operator` as a matrix.
  /// @arg  `dimensions` : A map specifying the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0 : 2, 1 : 2}`.
  virtual matrix_2 to_matrix(std::map<int, int> &dimensions,
                             std::map<std::string, std::complex<double>> parameters = {}) const = 0;

  virtual std::string to_string(bool include_degrees = true) const = 0;
};

} // namespace cudaq