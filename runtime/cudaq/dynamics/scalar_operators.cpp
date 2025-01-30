/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/operators.h"

#include <iostream>
#include <set>

namespace cudaq {

// constructors and destructors

scalar_operator::scalar_operator(double value) 
  : constant_value(value), generator() {}

scalar_operator::scalar_operator(std::complex<double> value) 
  : constant_value(value), generator() {}

scalar_operator::scalar_operator(const ScalarCallbackFunction &create) 
  : constant_value(), generator(create) {}

scalar_operator::scalar_operator(ScalarCallbackFunction &&create)
  : constant_value() {
    generator = std::move(create);
}

scalar_operator::scalar_operator(const scalar_operator &other) 
  : constant_value(other.constant_value), generator(other.generator) {}

scalar_operator::scalar_operator(scalar_operator &&other) 
  : constant_value(other.constant_value) {
    generator = std::move(other.generator);
}

// assignments

scalar_operator& scalar_operator::operator=(const scalar_operator &other) {
  if (this != &other) {
    constant_value = other.constant_value;
    generator = other.generator;
  }
  return *this;
}

scalar_operator& scalar_operator::operator=(scalar_operator &&other) {
  if (this != &other) {
    constant_value = other.constant_value;
    generator = std::move(other.generator);
  }
  return *this;
}

// evaluations

std::complex<double> scalar_operator::evaluate(
    const std::map<std::string, std::complex<double>> parameters) const {
  if (constant_value.has_value()) return constant_value.value();
  else return generator(parameters);
}

matrix_2 scalar_operator::to_matrix(
    const std::map<int, int> dimensions,
    const std::map<std::string, std::complex<double>> parameters) const {
  auto returnOperator = matrix_2(1, 1);
  returnOperator[{0, 0}] = evaluate(parameters);
  return returnOperator;
}

// comparison

bool scalar_operator::operator==(scalar_operator other) {
  if (this->constant_value.has_value() && other.constant_value.has_value()) {
    return this->constant_value == other.constant_value;
  } else {
    throw std::runtime_error("not implemented");
  }
}

// unary operators

scalar_operator scalar_operator::operator-() const {
  return *this * (-1.);
}

scalar_operator scalar_operator::operator+() const {
  return *this;
}

// right-hand arithmetics

#define ARITHMETIC_OPERATIONS(op, otherTy)                                     \
  scalar_operator scalar_operator::operator op(otherTy other) const {          \
    if (this->constant_value.has_value()) {                                    \
      return scalar_operator(this->constant_value.value() op other);           \
    }                                                                          \
    auto newGenerator =                                                        \
      [other, generator = this->generator](                                    \
          std::map<std::string, std::complex<double>> parameters) {            \
        return generator(parameters) op other;                                 \
      };                                                                       \
    return scalar_operator(newGenerator);                                      \
  }

ARITHMETIC_OPERATIONS(*, double);
ARITHMETIC_OPERATIONS(/, double);
ARITHMETIC_OPERATIONS(+, double);
ARITHMETIC_OPERATIONS(-, double);
ARITHMETIC_OPERATIONS(*, std::complex<double>);
ARITHMETIC_OPERATIONS(/, std::complex<double>);
ARITHMETIC_OPERATIONS(+, std::complex<double>);
ARITHMETIC_OPERATIONS(-, std::complex<double>);

#define ARITHMETIC_OPERATIONS_SCALAR_OPS(op)                                   \
  scalar_operator scalar_operator::operator op(                                \
                              const scalar_operator &other) const {            \
    if (this->constant_value.has_value() &&                                    \
        other.constant_value.has_value()) {                                    \
      auto res = this->constant_value.value() op other.constant_value.value(); \
      return scalar_operator(res);                                             \
    }                                                                          \
    auto newGenerator =                                                        \
        [other, *this](                                                        \
            std::map<std::string, std::complex<double>> parameters) {          \
          return this->evaluate(parameters) op other.evaluate(parameters);     \
        };                                                                     \
    return scalar_operator(newGenerator);                                      \
  }

ARITHMETIC_OPERATIONS_SCALAR_OPS(*);
ARITHMETIC_OPERATIONS_SCALAR_OPS(/);
ARITHMETIC_OPERATIONS_SCALAR_OPS(+);
ARITHMETIC_OPERATIONS_SCALAR_OPS(-);

#define ARITHMETIC_OPERATIONS_ASSIGNMENT(op, otherTy)                          \
  scalar_operator& scalar_operator::operator op(otherTy other) {               \
    if (this->constant_value.has_value()) {                                    \
      this->constant_value.value() op other;                                   \
      return *this;                                                            \
    }                                                                          \
    auto newGenerator =                                                        \
      [other, generator = std::move(this->generator)](                         \
          std::map<std::string, std::complex<double>> parameters) {            \
        return generator(parameters) op other;                                 \
      };                                                                       \
    this->generator = newGenerator;                                            \
    return *this;                                                              \
  }

ARITHMETIC_OPERATIONS_ASSIGNMENT(*=, double);
ARITHMETIC_OPERATIONS_ASSIGNMENT(/=, double);
ARITHMETIC_OPERATIONS_ASSIGNMENT(+=, double);
ARITHMETIC_OPERATIONS_ASSIGNMENT(-=, double);
ARITHMETIC_OPERATIONS_ASSIGNMENT(*=, std::complex<double>);
ARITHMETIC_OPERATIONS_ASSIGNMENT(/=, std::complex<double>);
ARITHMETIC_OPERATIONS_ASSIGNMENT(+=, std::complex<double>);
ARITHMETIC_OPERATIONS_ASSIGNMENT(-=, std::complex<double>);

#define ARITHMETIC_OPERATIONS_SCALAR_OPS_ASSIGNMENT(op)                        \
  scalar_operator& scalar_operator::operator op(                               \
                               const scalar_operator &other) {                 \
    if (this->constant_value.has_value() &&                                    \
        other.constant_value.has_value()) {                                    \
      this->constant_value.value() op other.constant_value.value();            \
      return *this;                                                            \
    }                                                                          \
    auto newGenerator =                                                        \
        [other, *this](                                                        \
            std::map<std::string, std::complex<double>> parameters) {          \
          return this->evaluate(parameters) op other.evaluate(parameters);     \
        };                                                                     \
    this->generator = newGenerator;                                            \
    return *this;                                                              \
  }

ARITHMETIC_OPERATIONS_SCALAR_OPS_ASSIGNMENT(*=);
ARITHMETIC_OPERATIONS_SCALAR_OPS_ASSIGNMENT(/=);
ARITHMETIC_OPERATIONS_SCALAR_OPS_ASSIGNMENT(+=);
ARITHMETIC_OPERATIONS_SCALAR_OPS_ASSIGNMENT(-=);

#define ARITHMETIC_OPERATIONS_RVALUE(op, otherTy)                              \
  scalar_operator operator op(scalar_operator &&self, otherTy other) {         \
    return std::move(self op##= other);                                        \
  }

ARITHMETIC_OPERATIONS_RVALUE(*, double);
ARITHMETIC_OPERATIONS_RVALUE(/, double);
ARITHMETIC_OPERATIONS_RVALUE(+, double);
ARITHMETIC_OPERATIONS_RVALUE(-, double);
ARITHMETIC_OPERATIONS_RVALUE(*, std::complex<double>);
ARITHMETIC_OPERATIONS_RVALUE(/, std::complex<double>);
ARITHMETIC_OPERATIONS_RVALUE(+, std::complex<double>);
ARITHMETIC_OPERATIONS_RVALUE(-, std::complex<double>);

// left-hand arithmetics

#define ARITHMETIC_OPERATIONS_REVERSE(op, otherTy)                             \
  scalar_operator operator op(otherTy other, const scalar_operator &self) {    \
    if (self.constant_value.has_value()) {                                     \
      return scalar_operator(other op self.constant_value.value());            \
    }                                                                          \
    auto newGenerator =                                                        \
      [other, generator = self.generator](                                     \
          std::map<std::string, std::complex<double>> parameters) {            \
        return other op generator(parameters);                                 \
      };                                                                       \
    return scalar_operator(newGenerator);                                      \
  }

ARITHMETIC_OPERATIONS_REVERSE(*, double);
ARITHMETIC_OPERATIONS_REVERSE(/, double);
ARITHMETIC_OPERATIONS_REVERSE(+, double);
ARITHMETIC_OPERATIONS_REVERSE(-, double);
ARITHMETIC_OPERATIONS_REVERSE(*, std::complex<double>);
ARITHMETIC_OPERATIONS_REVERSE(/, std::complex<double>);
ARITHMETIC_OPERATIONS_REVERSE(+, std::complex<double>);
ARITHMETIC_OPERATIONS_REVERSE(-, std::complex<double>);

} // namespace cudaq