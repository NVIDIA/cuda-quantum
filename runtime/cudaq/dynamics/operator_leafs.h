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
#include <type_traits>
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

  bool is_constant() const;

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
  scalar_operator &operator=(const scalar_operator &other);

  // move assignment operator
  scalar_operator &operator=(scalar_operator &&other);

  // evaluations

  /// @brief Return the scalar operator as a concrete complex value.
  std::complex<double>
  evaluate(const std::unordered_map<std::string, std::complex<double>>
               &parameters = {}) const;

  // Return the scalar operator as a 1x1 matrix. This is needed for
  // compatibility with the other inherited classes.
  matrix_2 to_matrix(const std::unordered_map<std::string, std::complex<double>>
                         &parameters = {}) const;

  // comparisons

  bool operator==(scalar_operator other) const;

  // unary operators

  scalar_operator operator-() const;
  scalar_operator operator+() const;

  // right-hand arithmetics

  scalar_operator operator*(double other) const;
  scalar_operator operator/(double other) const;
  scalar_operator operator+(double other) const;
  scalar_operator operator-(double other) const;
  scalar_operator &operator*=(double other);
  scalar_operator &operator/=(double other);
  scalar_operator &operator+=(double other);
  scalar_operator &operator-=(double other);
  scalar_operator operator*(std::complex<double> other) const;
  scalar_operator operator/(std::complex<double> other) const;
  scalar_operator operator+(std::complex<double> other) const;
  scalar_operator operator-(std::complex<double> other) const;
  scalar_operator &operator*=(std::complex<double> other);
  scalar_operator &operator/=(std::complex<double> other);
  scalar_operator &operator+=(std::complex<double> other);
  scalar_operator &operator-=(std::complex<double> other);
  scalar_operator operator*(const scalar_operator &other) const;
  scalar_operator operator/(const scalar_operator &other) const;
  scalar_operator operator+(const scalar_operator &other) const;
  scalar_operator operator-(const scalar_operator &other) const;
  scalar_operator &operator*=(const scalar_operator &other);
  scalar_operator &operator/=(const scalar_operator &other);
  scalar_operator &operator+=(const scalar_operator &other);
  scalar_operator &operator-=(const scalar_operator &other);

  friend scalar_operator operator*(scalar_operator &&self, double other);
  friend scalar_operator operator/(scalar_operator &&self, double other);
  friend scalar_operator operator+(scalar_operator &&self, double other);
  friend scalar_operator operator-(scalar_operator &&self, double other);
  friend scalar_operator operator+(scalar_operator &&self,
                                   std::complex<double> other);
  friend scalar_operator operator/(scalar_operator &&self,
                                   std::complex<double> other);
  friend scalar_operator operator+(scalar_operator &&self,
                                   std::complex<double> other);
  friend scalar_operator operator-(scalar_operator &&self,
                                   std::complex<double> other);

  // left-hand arithmetics

  friend scalar_operator operator*(double other, const scalar_operator &self);
  friend scalar_operator operator/(double other, const scalar_operator &self);
  friend scalar_operator operator+(double other, const scalar_operator &self);
  friend scalar_operator operator-(double other, const scalar_operator &self);
  friend scalar_operator operator*(std::complex<double> other,
                                   const scalar_operator &self);
  friend scalar_operator operator/(std::complex<double> other,
                                   const scalar_operator &self);
  friend scalar_operator operator+(std::complex<double> other,
                                   const scalar_operator &self);
  friend scalar_operator operator-(std::complex<double> other,
                                   const scalar_operator &self);
};

template <typename HandlerTy>
class product_operator;

template <typename HandlerTy>
class operator_sum;

class operator_handler {
public:
#if !defined(NDEBUG)
  static bool can_be_canonicalized; // whether a canonical order can be defined
                                    // for operator expressions
#endif

  virtual ~operator_handler() = default;

  virtual const std::string &unique_id() const = 0;

  virtual std::vector<int> degrees() const = 0;

  /// @brief Return the `matrix_operator` as a matrix.
  /// @arg  `dimensions` : A map specifying the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0 : 2, 1 : 2}`.
  virtual matrix_2
  to_matrix(std::unordered_map<int, int> &dimensions,
            const std::unordered_map<std::string, std::complex<double>>
                &parameters = {}) const = 0;

  virtual std::string to_string(bool include_degrees = true) const = 0;

  template <typename HandlerTy>
  static operator_sum<HandlerTy> empty();

  template <
      typename HandlerTy, typename... Args,
      std::enable_if_t<std::conjunction<std::is_same<int, Args>...>::value,
                       bool> = true>
  static product_operator<HandlerTy> identity(Args... targets);
};

} // namespace cudaq