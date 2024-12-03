/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/EigenDense.h"
#include "operators.h"

#include <iostream>
#include <set>

namespace cudaq {

/// Constructors.
scalar_operator::scalar_operator(const scalar_operator &other)
    : generator(other.generator), m_constant_value(other.m_constant_value) {}
scalar_operator::scalar_operator(scalar_operator &other)
    : generator(other.generator), m_constant_value(other.m_constant_value) {}

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

complex_matrix scalar_operator::to_matrix(
    const std::map<int, int> dimensions,
    const std::map<std::string, std::complex<double>> parameters) const {
  complex_matrix returnOperator(1, 1);
  returnOperator.set_zero();
  returnOperator(0, 0) = evaluate(parameters);
  return returnOperator;
}

template <typename TEval>
TEval elementary_operator::_evaluate(
    operator_arithmetics<TEval> &arithmetics) const {
  std::cout << "In ScalarOp _evaluate" << std::endl;
  return arithmetics.evaluate(*this);
}

#define ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES(op)                              \
  scalar_operator operator op(std::complex<double> other,                      \
                              scalar_operator self) {                          \
    /* Create an operator for the complex double value. */                     \
    auto otherOperator = scalar_operator(other);                               \
    /* Create an operator that we will store the result in and return to the   \
     * user. */                                                                \
    scalar_operator returnOperator;                                            \
    /* Store the previous generator functions in the new operator. This is     \
     * needed as the old generator functions would effectively be lost once we \
     * leave this function scope. */                                           \
    returnOperator._operators_to_compose.push_back(self);                      \
    returnOperator._operators_to_compose.push_back(otherOperator);             \
    auto newGenerator =                                                        \
        [&](std::map<std::string, std::complex<double>> parameters) {          \
          /* FIXME: I have to use this hacky `.get_val()` on the newly created \
           * operator for the given complex double -- because calling the      \
           * evaluate function returns 0.0 . I have no clue why??? */          \
          return returnOperator._operators_to_compose[0]                       \
              .evaluate(parameters) op returnOperator._operators_to_compose[1] \
              .get_val();                                                      \
        };                                                                     \
    returnOperator.generator = ScalarCallbackFunction(newGenerator);           \
    return returnOperator;                                                     \
  }

#define ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES_REVERSE(op)                      \
  scalar_operator operator op(scalar_operator self,                            \
                              std::complex<double> other) {                    \
    /* Create an operator for the complex double value. */                     \
    auto otherOperator = scalar_operator(other);                               \
    /* Create an operator that we will store the result in and return to the   \
     * user. */                                                                \
    scalar_operator returnOperator;                                            \
    /* Store the previous generator functions in the new operator. This is     \
     * needed as the old generator functions would effectively be lost once we \
     * leave this function scope. */                                           \
    returnOperator._operators_to_compose.push_back(self);                      \
    returnOperator._operators_to_compose.push_back(otherOperator);             \
    auto newGenerator =                                                        \
        [&](std::map<std::string, std::complex<double>> parameters) {          \
          /* FIXME: I have to use this hacky `.get_val()` on the newly created \
           * operator for the given complex double -- because calling the      \
           * evaluate function returns 0.0 . I have no clue why??? */          \
          return returnOperator._operators_to_compose[1]                       \
              .get_val() op returnOperator._operators_to_compose[0]            \
              .evaluate(parameters);                                           \
        };                                                                     \
    returnOperator.generator = ScalarCallbackFunction(newGenerator);           \
    return returnOperator;                                                     \
  }

#define ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES_ASSIGNMENT(op)                   \
  void operator op(scalar_operator &self, std::complex<double> other) {        \
    /* Create an operator for the complex double value. */                     \
    auto otherOperator = scalar_operator(other);                               \
    /* Need to move the existing generating function to a new operator so that \
     * we can modify the generator in `self` in-place. */                      \
    scalar_operator copy(self);                                                \
    /* Store the previous generator functions in the new operator. This is     \
     * needed as the old generator functions would effectively be lost once we \
     * leave this function scope. */                                           \
    self._operators_to_compose.push_back(copy);                                \
    self._operators_to_compose.push_back(otherOperator);                       \
    auto newGenerator =                                                        \
        [&](std::map<std::string, std::complex<double>> parameters) {          \
          /* FIXME: I have to use this hacky `.get_val()` on the newly created \
           * operator for the given complex double -- because calling the      \
           * evaluate function returns 0.0 . I have no clue why??? */          \
          return self._operators_to_compose[0]                                 \
              .evaluate(parameters) op self._operators_to_compose[1]           \
              .get_val();                                                      \
        };                                                                     \
    self.generator = ScalarCallbackFunction(newGenerator);                     \
  }

#define ARITHMETIC_OPERATIONS_DOUBLES(op)                                      \
  scalar_operator operator op(double other, scalar_operator self) {            \
    std::complex<double> value(other, 0.0);                                    \
    return self op value;                                                      \
  }

#define ARITHMETIC_OPERATIONS_DOUBLES_REVERSE(op)                              \
  scalar_operator operator op(scalar_operator self, double other) {            \
    std::complex<double> value(other, 0.0);                                    \
    return value op self;                                                      \
  }

#define ARITHMETIC_OPERATIONS_DOUBLES_ASSIGNMENT(op)                           \
  void operator op(scalar_operator &self, double other) {                      \
    std::complex<double> value(other, 0.0);                                    \
    self op value;                                                             \
  }

ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES(+);
ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES(-);
ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES(*);
ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES(/);
ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES_REVERSE(+);
ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES_REVERSE(-);
ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES_REVERSE(*);
ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES_REVERSE(/);
ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES_ASSIGNMENT(+=);
ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES_ASSIGNMENT(-=);
ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES_ASSIGNMENT(*=);
ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES_ASSIGNMENT(/=);
ARITHMETIC_OPERATIONS_DOUBLES(+);
ARITHMETIC_OPERATIONS_DOUBLES(-);
ARITHMETIC_OPERATIONS_DOUBLES(*);
ARITHMETIC_OPERATIONS_DOUBLES(/);
ARITHMETIC_OPERATIONS_DOUBLES_REVERSE(+);
ARITHMETIC_OPERATIONS_DOUBLES_REVERSE(-);
ARITHMETIC_OPERATIONS_DOUBLES_REVERSE(*);
ARITHMETIC_OPERATIONS_DOUBLES_REVERSE(/);
ARITHMETIC_OPERATIONS_DOUBLES_ASSIGNMENT(+=);
ARITHMETIC_OPERATIONS_DOUBLES_ASSIGNMENT(-=);
ARITHMETIC_OPERATIONS_DOUBLES_ASSIGNMENT(*=);
ARITHMETIC_OPERATIONS_DOUBLES_ASSIGNMENT(/=);

#define ARITHMETIC_OPERATIONS_SCALAR_OPS(op)                                   \
  scalar_operator scalar_operator::operator op(scalar_operator other) {        \
    /* Create an operator that we will store the result in and return to the   \
     * user. */                                                                \
    scalar_operator returnOperator;                                            \
    /* Store the previous generator functions in the new operator. This is     \
     * needed as the old generator functions would effectively be lost once we \
     * leave this function scope. */                                           \
    returnOperator._operators_to_compose.push_back(*this);                     \
    returnOperator._operators_to_compose.push_back(other);                     \
    auto newGenerator =                                                        \
        [&](std::map<std::string, std::complex<double>> parameters) {          \
          return returnOperator._operators_to_compose[0]                       \
              .evaluate(parameters) op returnOperator._operators_to_compose[1] \
              .evaluate(parameters);                                           \
        };                                                                     \
    returnOperator.generator = ScalarCallbackFunction(newGenerator);           \
    return returnOperator;                                                     \
  }

/// FIXME: Broken implementation
#define ARITHMETIC_OPERATIONS_SCALAR_OPS_ASSIGNMENT(op)                        \
  void operator op(scalar_operator &self, scalar_operator other) {             \
    /* Need to move the existing generating function to a new operator so      \
     * that we can modify the generator in `self` in-place. */                 \
    scalar_operator selfCopy(self);                                            \
    /* Store the previous generator functions in the new operator. This is     \
     * needed as the old generator functions would effectively be lost once we \
     * leave this function scope. */                                           \
    self._operators_to_compose.push_back(selfCopy);                            \
    self._operators_to_compose.push_back(other);                               \
    auto newGenerator =                                                        \
        [&](std::map<std::string, std::complex<double>> parameters) {          \
          return self._operators_to_compose[0]                                 \
              .evaluate(parameters) op self._operators_to_compose[1]           \
              .evaluate(parameters);                                           \
        };                                                                     \
    self.generator = ScalarCallbackFunction(newGenerator);                     \
  }

ARITHMETIC_OPERATIONS_SCALAR_OPS(+);
ARITHMETIC_OPERATIONS_SCALAR_OPS(-);
ARITHMETIC_OPERATIONS_SCALAR_OPS(*);
ARITHMETIC_OPERATIONS_SCALAR_OPS(/);
ARITHMETIC_OPERATIONS_SCALAR_OPS_ASSIGNMENT(+=);
ARITHMETIC_OPERATIONS_SCALAR_OPS_ASSIGNMENT(-=);
ARITHMETIC_OPERATIONS_SCALAR_OPS_ASSIGNMENT(*=);
ARITHMETIC_OPERATIONS_SCALAR_OPS_ASSIGNMENT(/=);

operator_sum scalar_operator::operator+(elementary_operator other) {
  // Operator sum is composed of product operators, so we must convert
  // both underlying types to `product_operators` to perform the arithmetic.
  return operator_sum({product_operator({*this}), product_operator({other})});
}

operator_sum scalar_operator::operator-(elementary_operator other) {
  // Operator sum is composed of product operators, so we must convert
  // both underlying types to `product_operators` to perform the arithmetic.
  return operator_sum(
      {product_operator({*this}), product_operator({-1. * other})});
}

product_operator scalar_operator::operator*(elementary_operator other) {
  return product_operator({*this, other});
}

/// FIXME: division on elementary op needed
// product_operator scalar_operator::operator/(elementary_operator other) {
//   return product_operator({*this, (1./other)});
// }

} // namespace cudaq