/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/operators.h"

#include <iostream>
#include <set>

namespace cudaq {

// read-only properties

bool scalar_operator::is_constant() const {
  return std::holds_alternative<std::complex<double>>(value);
}

const std::unordered_map<std::string, std::string> &
scalar_operator::get_parameter_descriptions() const {
  return this->param_desc;
}

// constructors and destructors

scalar_operator::scalar_operator(double value)
    : value(std::variant<std::complex<double>, scalar_callback>(
          std::complex<double>(value))) {}

scalar_operator::scalar_operator(std::complex<double> value)
    : value(std::variant<std::complex<double>, scalar_callback>(value)) {}

scalar_operator::scalar_operator(
    const scalar_callback &create,
    std::unordered_map<std::string, std::string> &&paramater_descriptions)
    : value(std::variant<std::complex<double>, scalar_callback>(create)),
      param_desc(std::move(paramater_descriptions)) {}

scalar_operator::scalar_operator(
    scalar_callback &&create,
    std::unordered_map<std::string, std::string> &&paramater_descriptions)
    : value(std::variant<std::complex<double>, scalar_callback>(
          std::move(create))),
      param_desc(std::move(paramater_descriptions)) {}

// evaluations

std::complex<double> scalar_operator::evaluate(
    const std::unordered_map<std::string, std::complex<double>> &parameters)
    const {
  if (std::holds_alternative<scalar_callback>(this->value))
    return std::get<scalar_callback>(this->value)(parameters);
  return std::get<std::complex<double>>(this->value);
}

complex_matrix scalar_operator::to_matrix(
    const std::unordered_map<std::string, std::complex<double>> &parameters)
    const {
  auto returnOperator = complex_matrix(1, 1);
  returnOperator[{0, 0}] = evaluate(parameters);
  return returnOperator;
}

std::string scalar_operator::to_string() const {
  std::stringstream sstr;
  if (std::holds_alternative<std::complex<double>>(this->value)) {
    auto value = std::get<std::complex<double>>(this->value);
    sstr << "(" << value.real() << (value.imag() < 0 ? "-" : "+")
         << std::abs(value.imag()) << "i)";
    return sstr.str();
  }
  if (this->param_desc.size() == 0)
    return "scalar";
  auto it = this->param_desc.cbegin();
  sstr << "scalar(" << it->first;
  while (++it != this->param_desc.cend())
    sstr << "," << it->first;
  sstr << ")";
  return sstr.str();
}

// comparison

bool scalar_operator::operator==(const scalar_operator &other) const {
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
