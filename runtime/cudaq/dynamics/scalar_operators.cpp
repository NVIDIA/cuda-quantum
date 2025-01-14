/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/EigenDense.h"
#include "cudaq/operators.h"

#include <iostream>
#include <set>

namespace cudaq {

/// @brief Constructor that just takes and returns a complex double value.
scalar_operator::scalar_operator(std::complex<double> value) {
  m_constant_value = value;
  auto func = [&](std::map<std::string, std::complex<double>> _none) {
    return m_constant_value;
  };
  generator = ScalarCallbackFunction(func);
}

/// @brief Constructor that just takes a double and returns a complex double.
scalar_operator::scalar_operator(double value) {
  std::complex<double> castValue(value, 0.0);
  m_constant_value = castValue;
  auto func = [&](std::map<std::string, std::complex<double>> _none) {
    return m_constant_value;
  };
  generator = ScalarCallbackFunction(func);
}

std::complex<double> scalar_operator::evaluate(
    const std::map<std::string, std::complex<double>> parameters) const {
  return generator(parameters);
}

matrix_2 scalar_operator::to_matrix(
    const std::map<int, int> dimensions,
    const std::map<std::string, std::complex<double>> parameters) const {
  auto returnOperator = matrix_2(1, 1);
  returnOperator[{0, 0}] = evaluate(parameters);
  return returnOperator;
}

#define ARITHMETIC_OPERATIONS_SCALAR_OPS(op)                                   \
  scalar_operator operator op(const scalar_operator &self,                     \
                              const scalar_operator other) {                   \
    auto newGenerator =                                                        \
        [&](std::map<std::string, std::complex<double>> parameters) {          \
          return self                                                          \
              .evaluate(parameters) op other                                   \
              .evaluate(parameters);                                           \
        };                                                                     \
    return scalar_operator(ScalarCallbackFunction(newGenerator));              \
  }

ARITHMETIC_OPERATIONS_SCALAR_OPS(+);
ARITHMETIC_OPERATIONS_SCALAR_OPS(-);
ARITHMETIC_OPERATIONS_SCALAR_OPS(*);
ARITHMETIC_OPERATIONS_SCALAR_OPS(/);

#define ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES(op)                              \
  scalar_operator operator op(const scalar_operator &self,                     \
                              const std::complex<double> other) {              \
    auto newGenerator =                                                        \
        [&](std::map<std::string, std::complex<double>> parameters) {          \
          return self.evaluate(parameters) op other;                           \
        };                                                                     \
    return scalar_operator(ScalarCallbackFunction(newGenerator));              \
  }

ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES(+);
ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES(-);
ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES(*);
ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES(/);

#define ARITHMETIC_OPERATIONS_DOUBLES(op)                                      \
  scalar_operator operator op(const scalar_operator &self, double other) {     \
    auto newGenerator =                                                        \
        [&](std::map<std::string, std::complex<double>> parameters) {          \
          return self                                                          \
              .evaluate(parameters) op other;                                  \
        };                                                                     \
    return scalar_operator(ScalarCallbackFunction(newGenerator));              \
  }

ARITHMETIC_OPERATIONS_DOUBLES(+);
ARITHMETIC_OPERATIONS_DOUBLES(-);
ARITHMETIC_OPERATIONS_DOUBLES(*);
ARITHMETIC_OPERATIONS_DOUBLES(/);

#define ARITHMETIC_OPERATIONS_SCALAR_OPS_ASSIGNMENT(op)                        \
  void operator op(scalar_operator &self, const scalar_operator other) {       \
    /* Need to move the existing generating function to a new operator so      \
     * that we can modify the generator in `self` in-place. */                 \
    scalar_operator prevSelf(self);                                            \
    auto newGenerator =                                                        \
        [&](std::map<std::string, std::complex<double>> parameters) {          \
          return prevSelf                                                      \
              .evaluate(parameters) op other                                   \
              .evaluate(parameters);                                           \
        };                                                                     \
    self.generator = ScalarCallbackFunction(newGenerator);                     \
  }

ARITHMETIC_OPERATIONS_SCALAR_OPS_ASSIGNMENT(+=);
ARITHMETIC_OPERATIONS_SCALAR_OPS_ASSIGNMENT(-=);
ARITHMETIC_OPERATIONS_SCALAR_OPS_ASSIGNMENT(*=);
ARITHMETIC_OPERATIONS_SCALAR_OPS_ASSIGNMENT(/=);

#define ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES_ASSIGNMENT(op)                   \
  void operator op(scalar_operator &self, const std::complex<double> other) {  \
    /* Need to move the existing generating function to a new operator so that \
     * we can modify the generator in `self` in-place. */                      \
    scalar_operator prevSelf(self);                                            \
    auto newGenerator =                                                        \
        [&](std::map<std::string, std::complex<double>> parameters) {          \
          return prevSelf                                                      \
              .evaluate(parameters) op other;                                  \
        };                                                                     \
    self.generator = ScalarCallbackFunction(newGenerator);                     \
  }

ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES_ASSIGNMENT(+=);
ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES_ASSIGNMENT(-=);
ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES_ASSIGNMENT(*=);
ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES_ASSIGNMENT(/=);

#define ARITHMETIC_OPERATIONS_DOUBLES_ASSIGNMENT(op)                           \
  void operator op(scalar_operator &self, double other) {                      \
    /* Need to move the existing generating function to a new operator so that \
     * we can modify the generator in `self` in-place. */                      \
    scalar_operator prevSelf(self);                                            \
    auto newGenerator =                                                        \
        [&](std::map<std::string, std::complex<double>> parameters) {          \
          return prevSelf                                                      \
              .evaluate(parameters) op other;                                  \
        };                                                                     \
    self.generator = ScalarCallbackFunction(newGenerator);                     \
  }

ARITHMETIC_OPERATIONS_DOUBLES_ASSIGNMENT(+=);
ARITHMETIC_OPERATIONS_DOUBLES_ASSIGNMENT(-=);
ARITHMETIC_OPERATIONS_DOUBLES_ASSIGNMENT(*=);
ARITHMETIC_OPERATIONS_DOUBLES_ASSIGNMENT(/=);

#define ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES_REVERSE(op)                      \
  scalar_operator operator op(const std::complex<double> other,                \
                              const scalar_operator self) {                    \
    auto newGenerator =                                                        \
        [&](std::map<std::string, std::complex<double>> parameters) {          \
          return other op self.evaluate(parameters);                           \
        };                                                                     \
    return scalar_operator(ScalarCallbackFunction(newGenerator));              \
  }

ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES_REVERSE(+);
ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES_REVERSE(-);
ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES_REVERSE(*);
ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES_REVERSE(/);

#define ARITHMETIC_OPERATIONS_DOUBLES_REVERSE(op)                              \
  scalar_operator operator op(double other, const scalar_operator self) {      \
    auto newGenerator =                                                        \
        [&](std::map<std::string, std::complex<double>> parameters) {          \
          return other op self.evaluate(parameters);                           \
        };                                                                     \
    return scalar_operator(ScalarCallbackFunction(newGenerator));              \
  }

ARITHMETIC_OPERATIONS_DOUBLES_REVERSE(+);
ARITHMETIC_OPERATIONS_DOUBLES_REVERSE(-);
ARITHMETIC_OPERATIONS_DOUBLES_REVERSE(*);
ARITHMETIC_OPERATIONS_DOUBLES_REVERSE(/);

template <typename HandlerTy>
operator_sum<HandlerTy> scalar_operator::operator+(const HandlerTy other) const {
  // Operator sum is composed of product operators, so we must convert
  // both underlying types to `product_operators` to perform the arithmetic.
  return operator_sum({product_operator({*this}), product_operator({other})});
}

template <typename HandlerTy>
operator_sum<HandlerTy> scalar_operator::operator-(const HandlerTy other) const {
  // Operator sum is composed of product operators, so we must convert
  // both underlying types to `product_operators` to perform the arithmetic.
  return operator_sum(
      {product_operator({*this}), product_operator({-1. * other})});
}

template <typename HandlerTy>
product_operator<HandlerTy> scalar_operator::operator*(const HandlerTy other) const {
  return product_operator({*this, other});
}

template <typename HandlerTy>
operator_sum<HandlerTy> scalar_operator::operator+(const product_operator<HandlerTy> other) const {
  return operator_sum({product_operator({*this}), other});
}

template <typename HandlerTy>
operator_sum<HandlerTy> scalar_operator::operator-(const product_operator<HandlerTy> other) const {
  return operator_sum({product_operator({*this}), (-1. * other)});
}

template <typename HandlerTy>
product_operator<HandlerTy> scalar_operator::operator*(const product_operator<HandlerTy> other) const {
  std::vector<std::variant<scalar_operator, HandlerTy>> other_terms =
      other.get_terms();
  /// Insert this scalar operator to the front of the terms list.
  other_terms.insert(other_terms.begin(), *this);
  return product_operator(other_terms);
}

template <typename HandlerTy>
operator_sum<HandlerTy> scalar_operator::operator+(const operator_sum<HandlerTy> other) const {
  std::vector<product_operator<HandlerTy>> other_terms = other.get_terms();
  other_terms.insert(other_terms.begin(), *this);
  return operator_sum(other_terms);
}

template <typename HandlerTy>
operator_sum<HandlerTy> scalar_operator::operator-(const operator_sum<HandlerTy> other) const {
  auto negative_other = (-1. * other);
  std::vector<product_operator<HandlerTy>> other_terms = negative_other.get_terms();
  other_terms.insert(other_terms.begin(), *this);
  return operator_sum(other_terms);
}

template <typename HandlerTy>
operator_sum<HandlerTy> scalar_operator::operator*(const operator_sum<HandlerTy> other) const {
  std::vector<product_operator<HandlerTy>> other_terms = other.get_terms();
  for (auto &term : other_terms)
    term = *this * term;
  return operator_sum(other_terms);
}

} // namespace cudaq