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
    : value(std::variant<std::complex<double>, scalar_callback>(
          std::complex<double>(value))) {}

scalar_operator::scalar_operator(std::complex<double> value)
    : value(std::variant<std::complex<double>, scalar_callback>(value)) {}

scalar_operator::scalar_operator(const scalar_callback &create)
    : value(std::variant<std::complex<double>, scalar_callback>(create)) {}

scalar_operator::scalar_operator(scalar_callback &&create)
    : value(std::variant<std::complex<double>, scalar_callback>(
          std::move(create))) {}

scalar_operator::scalar_operator(const scalar_operator &other)
    : value(other.value) {}

scalar_operator::scalar_operator(scalar_operator &&other)
    : value(std::move(other.value)) {}

bool scalar_operator::is_constant() const {
  return std::holds_alternative<std::complex<double>>(value);
}

// assignments

scalar_operator &scalar_operator::operator=(const scalar_operator &other) {
  if (this != &other)
    this->value = other.value;
  return *this;
}

scalar_operator &scalar_operator::operator=(scalar_operator &&other) {
  if (this != &other)
    this->value = std::move(other.value);
  return *this;
}

// evaluations

std::complex<double> scalar_operator::evaluate(
    const std::unordered_map<std::string, std::complex<double>> &parameters)
    const {
  if (std::holds_alternative<scalar_callback>(this->value))
    return std::get<scalar_callback>(this->value)(parameters);
  return std::get<std::complex<double>>(this->value);
}

matrix_2 scalar_operator::to_matrix(
    const std::unordered_map<std::string, std::complex<double>> &parameters)
    const {
  auto returnOperator = matrix_2(1, 1);
  returnOperator[{0, 0}] = evaluate(parameters);
  return returnOperator;
}

std::string scalar_operator::to_string() const {
  if (std::holds_alternative<std::complex<double>>(this->value)) {
    auto value = std::get<std::complex<double>>(this->value);
    return "(" + std::to_string(value.real()) + "+" +
           std::to_string(value.imag()) + "i)";
  }
  return "scalar";
}

// comparison

bool scalar_operator::operator==(scalar_operator other) const {
  if (std::holds_alternative<scalar_callback>(this->value)) {
    return std::holds_alternative<scalar_callback>(other.value) &&
           &std::get<scalar_callback>(this->value) ==
               &std::get<scalar_callback>(other.value);
  } else {
    return std::holds_alternative<std::complex<double>>(this->value) &&
           std::get<std::complex<double>>(this->value) ==
               std::get<std::complex<double>>(other.value);
  }
}

// unary operators

scalar_operator scalar_operator::operator-() const & { return *this * (-1.); }

scalar_operator scalar_operator::operator-() && {
  *this *= -1.;
  return std::move(*this);
}

scalar_operator scalar_operator::operator+() const & { return *this; }

scalar_operator scalar_operator::operator+() && { return std::move(*this); }

// right-hand arithmetics

#define ARITHMETIC_OPERATIONS(op, otherTy)                                     \
  scalar_operator scalar_operator::operator op(otherTy other) const & {        \
    if (std::holds_alternative<std::complex<double>>(this->value)) {           \
      return scalar_operator(std::get<std::complex<double>>(this->value)       \
                                 op other);                                    \
    }                                                                          \
    auto newGenerator =                                                        \
        [other, generator = std::get<scalar_callback>(this->value)](           \
            const std::unordered_map<std::string, std::complex<double>>        \
                &parameters) { return generator(parameters) op other; };       \
    return scalar_operator(std::move(newGenerator));                           \
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
  scalar_operator scalar_operator::operator op(const scalar_operator &other)   \
      const & {                                                                \
    if (std::holds_alternative<std::complex<double>>(this->value) &&           \
        std::holds_alternative<std::complex<double>>(other.value)) {           \
      return scalar_operator(std::get<std::complex<double>>(                   \
          this->value) op std::get<std::complex<double>>(other.value));        \
    }                                                                          \
    auto newGenerator =                                                        \
        [other,                                                                \
         *this](const std::unordered_map<std::string, std::complex<double>>    \
                    &parameters) {                                             \
          return this->evaluate(parameters) op other.evaluate(parameters);     \
        };                                                                     \
    return scalar_operator(std::move(newGenerator));                           \
  }

ARITHMETIC_OPERATIONS_SCALAR_OPS(*);
ARITHMETIC_OPERATIONS_SCALAR_OPS(/);
ARITHMETIC_OPERATIONS_SCALAR_OPS(+);
ARITHMETIC_OPERATIONS_SCALAR_OPS(-);

#define ARITHMETIC_OPERATIONS_ASSIGNMENT(op, otherTy)                          \
  scalar_operator &scalar_operator::operator op##=(otherTy other) {            \
    if (std::holds_alternative<std::complex<double>>(this->value)) {           \
      this->value = std::get<std::complex<double>>(this->value) op other;      \
      return *this;                                                            \
    }                                                                          \
    auto newGenerator =                                                        \
        [other,                                                                \
         generator = std::move(std::get<scalar_callback>(this->value))](       \
            const std::unordered_map<std::string, std::complex<double>>        \
                &parameters) { return generator(parameters) op## = other; };   \
    this->value = std::move(newGenerator);                                     \
    return *this;                                                              \
  }

ARITHMETIC_OPERATIONS_ASSIGNMENT(*, double);
ARITHMETIC_OPERATIONS_ASSIGNMENT(/, double);
ARITHMETIC_OPERATIONS_ASSIGNMENT(+, double);
ARITHMETIC_OPERATIONS_ASSIGNMENT(-, double);
ARITHMETIC_OPERATIONS_ASSIGNMENT(*, std::complex<double>);
ARITHMETIC_OPERATIONS_ASSIGNMENT(/, std::complex<double>);
ARITHMETIC_OPERATIONS_ASSIGNMENT(+, std::complex<double>);
ARITHMETIC_OPERATIONS_ASSIGNMENT(-, std::complex<double>);

#define ARITHMETIC_OPERATIONS_SCALAR_OPS_ASSIGNMENT(op)                        \
  scalar_operator &scalar_operator::operator op##=(                            \
      const scalar_operator &other) {                                          \
    if (std::holds_alternative<std::complex<double>>(this->value) &&           \
        std::holds_alternative<std::complex<double>>(other.value)) {           \
      this->value = std::get<std::complex<double>>(this->value)                \
          op std::get<std::complex<double>>(other.value);                      \
      return *this;                                                            \
    }                                                                          \
    auto newGenerator =                                                        \
        [other,                                                                \
         *this](const std::unordered_map<std::string, std::complex<double>>    \
                    &parameters) {                                             \
          return this->evaluate(parameters) op## = other.evaluate(parameters); \
        };                                                                     \
    this->value = std::move(newGenerator);                                     \
    return *this;                                                              \
  }

ARITHMETIC_OPERATIONS_SCALAR_OPS_ASSIGNMENT(*);
ARITHMETIC_OPERATIONS_SCALAR_OPS_ASSIGNMENT(/);
ARITHMETIC_OPERATIONS_SCALAR_OPS_ASSIGNMENT(+);
ARITHMETIC_OPERATIONS_SCALAR_OPS_ASSIGNMENT(-);

#define ARITHMETIC_OPERATIONS_RVALUE(op, otherTy)                              \
  scalar_operator scalar_operator::operator op(otherTy other) && {             \
    *this op## = other;                                                        \
    return std::move(*this);                                                   \
  }

ARITHMETIC_OPERATIONS_RVALUE(*, double);
ARITHMETIC_OPERATIONS_RVALUE(/, double);
ARITHMETIC_OPERATIONS_RVALUE(+, double);
ARITHMETIC_OPERATIONS_RVALUE(-, double);
ARITHMETIC_OPERATIONS_RVALUE(*, std::complex<double>);
ARITHMETIC_OPERATIONS_RVALUE(/, std::complex<double>);
ARITHMETIC_OPERATIONS_RVALUE(+, std::complex<double>);
ARITHMETIC_OPERATIONS_RVALUE(-, std::complex<double>);
ARITHMETIC_OPERATIONS_RVALUE(*, const scalar_operator &);
ARITHMETIC_OPERATIONS_RVALUE(/, const scalar_operator &);
ARITHMETIC_OPERATIONS_RVALUE(+, const scalar_operator &);
ARITHMETIC_OPERATIONS_RVALUE(-, const scalar_operator &);

// left-hand arithmetics

#define ARITHMETIC_OPERATIONS_REVERSE(op, otherTy)                             \
                                                                               \
  scalar_operator operator op(otherTy other, const scalar_operator &self) {    \
    if (std::holds_alternative<std::complex<double>>(self.value)) {            \
      return scalar_operator(                                                  \
          other op std::get<std::complex<double>>(self.value));                \
    }                                                                          \
    auto newGenerator =                                                        \
        [other, generator = std::get<scalar_callback>(self.value)](            \
            const std::unordered_map<std::string, std::complex<double>>        \
                &parameters) { return other op generator(parameters); };       \
    return scalar_operator(std::move(newGenerator));                           \
  }                                                                            \
                                                                               \
  scalar_operator operator op(otherTy other, scalar_operator &&self) {         \
    if (std::holds_alternative<std::complex<double>>(self.value)) {            \
      return scalar_operator(                                                  \
          other op std::get<std::complex<double>>(self.value));                \
    }                                                                          \
    auto newGenerator =                                                        \
        [other, generator = std::move(std::get<scalar_callback>(self.value))]( \
            const std::unordered_map<std::string, std::complex<double>>        \
                &parameters) { return other op generator(parameters); };       \
    self.value = std::move(newGenerator);                                      \
    return std::move(self);                                                    \
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