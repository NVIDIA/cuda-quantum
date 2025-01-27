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

// evaluations

std::complex<double> scalar_operator::evaluate(
    const std::map<std::string, std::complex<double>> parameters) const {
  if (m_constant_value.has_value()) return m_constant_value.value();
  else return generator(parameters);
}

matrix_2 scalar_operator::to_matrix(
    const std::map<int, int> dimensions,
    const std::map<std::string, std::complex<double>> parameters) const {
  auto returnOperator = matrix_2(1, 1);
  returnOperator[{0, 0}] = evaluate(parameters);
  return returnOperator;
}

// left-hand arithmetics

#define ARITHMETIC_OPERATIONS_DOUBLES_REVERSE(op)                              \
  scalar_operator operator op(double other, const scalar_operator &self) {     \
    auto newGenerator =                                                        \
        [=](std::map<std::string, std::complex<double>> parameters) {          \
          return other op self.evaluate(parameters);                           \
        };                                                                     \
    return scalar_operator(newGenerator);                                      \
  }

ARITHMETIC_OPERATIONS_DOUBLES_REVERSE(*);
ARITHMETIC_OPERATIONS_DOUBLES_REVERSE(/);
ARITHMETIC_OPERATIONS_DOUBLES_REVERSE(+);
ARITHMETIC_OPERATIONS_DOUBLES_REVERSE(-);

#define ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES_REVERSE(op)                      \
  scalar_operator operator op(std::complex<double> other,                      \
                              const scalar_operator &self) {                   \
    auto newGenerator =                                                        \
        [=](std::map<std::string, std::complex<double>> parameters) {          \
          return other op self.evaluate(parameters);                           \
        };                                                                     \
    return scalar_operator(newGenerator);                                      \
  }

ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES_REVERSE(*);
ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES_REVERSE(/);
ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES_REVERSE(+);
ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES_REVERSE(-);

// right-hand arithmetics

#define ARITHMETIC_OPERATIONS_DOUBLES(op)                                      \
  scalar_operator scalar_operator::operator op(double other) const {           \
    auto newGenerator =                                                        \
        [=, *this](std::map<std::string, std::complex<double>> parameters) {   \
          return this->evaluate(parameters) op other;                          \
        };                                                                     \
    return scalar_operator(newGenerator);                                      \
  }

ARITHMETIC_OPERATIONS_DOUBLES(*);
ARITHMETIC_OPERATIONS_DOUBLES(/);
ARITHMETIC_OPERATIONS_DOUBLES(+);
ARITHMETIC_OPERATIONS_DOUBLES(-);

#define ARITHMETIC_OPERATIONS_DOUBLES_ASSIGNMENT(op)                           \
  scalar_operator& scalar_operator::operator op(double other) {                \
    if (this->m_constant_value.has_value()) {                                  \
        this->m_constant_value.value() op other;                               \
        return *this;                                                          \
    }                                                                          \
    /* Need to move the existing generating function to a new operator so that \
     * we can modify the generator in-place. */                                \
    scalar_operator prevSelf(*this);                                           \
    auto newGenerator =                                                        \
        [=](std::map<std::string, std::complex<double>> parameters) {          \
          return prevSelf.evaluate(parameters) op other;                       \
        };                                                                     \
    this->generator = newGenerator;                                            \
    return *this;                                                              \
  }

ARITHMETIC_OPERATIONS_DOUBLES_ASSIGNMENT(*=);
ARITHMETIC_OPERATIONS_DOUBLES_ASSIGNMENT(/=);
ARITHMETIC_OPERATIONS_DOUBLES_ASSIGNMENT(+=);
ARITHMETIC_OPERATIONS_DOUBLES_ASSIGNMENT(-=);

#define ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES(op)                              \
  scalar_operator scalar_operator::operator op(                                \
                                    std::complex<double> other) const{         \
    auto newGenerator =                                                        \
        [=, *this](std::map<std::string, std::complex<double>> parameters) {   \
          return this->evaluate(parameters) op other;                          \
        };                                                                     \
    return scalar_operator(newGenerator);                                      \
  }

ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES(*);
ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES(/);
ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES(+);
ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES(-);

#define ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES_ASSIGNMENT(op)                   \
  scalar_operator& scalar_operator::operator op(                               \
                               std::complex<double> other) {                   \
    if (this->m_constant_value.has_value()) {                                  \
        this->m_constant_value.value() op other;                               \
        return *this;                                                          \
    }                                                                          \
    /* Need to move the existing generating function to a new operator so that \
     * we can modify the generator in-place. */                                \
    scalar_operator prevSelf(*this);                                           \
    auto newGenerator =                                                        \
        [=](std::map<std::string, std::complex<double>> parameters) {          \
          return prevSelf.evaluate(parameters) op other;                       \
        };                                                                     \
    this->generator = newGenerator;                                            \
    return *this;                                                              \
  }

ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES_ASSIGNMENT(*=);
ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES_ASSIGNMENT(/=);
ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES_ASSIGNMENT(+=);
ARITHMETIC_OPERATIONS_COMPLEX_DOUBLES_ASSIGNMENT(-=);

#define ARITHMETIC_OPERATIONS_SCALAR_OPS(op)                                   \
  scalar_operator scalar_operator::operator op(                                \
                              const scalar_operator &other) const {            \
    auto newGenerator =                                                        \
        [=, *this](std::map<std::string, std::complex<double>> parameters) {   \
          return this->evaluate(parameters) op other.evaluate(parameters);     \
        };                                                                     \
    return scalar_operator(newGenerator);                                      \
  }

ARITHMETIC_OPERATIONS_SCALAR_OPS(*);
ARITHMETIC_OPERATIONS_SCALAR_OPS(/);
ARITHMETIC_OPERATIONS_SCALAR_OPS(+);
ARITHMETIC_OPERATIONS_SCALAR_OPS(-);

#define ARITHMETIC_OPERATIONS_SCALAR_OPS_ASSIGNMENT(op)                        \
  scalar_operator& scalar_operator::operator op(                               \
                               const scalar_operator &other) {                 \
    if (this->m_constant_value.has_value() &&                                  \
          other.m_constant_value.has_value()) {                                \
        this->m_constant_value.value() op other.m_constant_value.value();      \
        return *this;                                                          \
    }                                                                          \
    /* Need to move the existing generating function to a new operator so      \
     * that we can modify the generator in-place. */                           \
    scalar_operator prevSelf(*this);                                           \
    auto newGenerator =                                                        \
        [=](std::map<std::string, std::complex<double>> parameters) {          \
          return prevSelf.evaluate(parameters) op other.evaluate(parameters);  \
        };                                                                     \
    this->generator = newGenerator;                                            \
    return *this;                                                              \
  }

ARITHMETIC_OPERATIONS_SCALAR_OPS_ASSIGNMENT(*=);
ARITHMETIC_OPERATIONS_SCALAR_OPS_ASSIGNMENT(/=);
ARITHMETIC_OPERATIONS_SCALAR_OPS_ASSIGNMENT(+=);
ARITHMETIC_OPERATIONS_SCALAR_OPS_ASSIGNMENT(-=);

} // namespace cudaq