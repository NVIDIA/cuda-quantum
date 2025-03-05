/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <complex>
#include <unordered_map>
#include <vector>

#include "cudaq/operators.h"
#include "cudaq/utils/matrix.h"

#include "boson_operators.h"
#include "fermion_operators.h"
#include "matrix_operators.h"
#include "spin_operators.h"

namespace cudaq {

#if !defined(NDEBUG)
bool matrix_operator::can_be_canonicalized = false;
#endif

// tools for custom operators

std::unordered_map<std::string, Definition> matrix_operator::defined_ops = {};

template <typename T>
std::string matrix_operator::type_prefix() {
  return typeid(T).name();
}

// no need to prefix the operator id and op code with the type name for these
// (same names mean the same thing)
template <>
std::string matrix_operator::type_prefix<spin_operator>() {
  return "";
}
template <>
std::string matrix_operator::type_prefix<boson_operator>() {
  return "";
}
template <>
std::string matrix_operator::type_prefix<fermion_operator>() {
  return "";
}

void matrix_operator::define(std::string operator_id,
                             std::vector<int> expected_dimensions,
                             matrix_callback &&create) {
  auto defn = Definition(operator_id, expected_dimensions,
                         std::forward<matrix_callback>(create));
  auto result =
      matrix_operator::defined_ops.insert({operator_id, std::move(defn)});
  if (!result.second)
    throw std::runtime_error("an matrix operator with name " + operator_id +
                             "is already defined");
}

product_operator<matrix_operator>
matrix_operator::instantiate(std::string operator_id,
                             const std::vector<int> &degrees,
                             const commutation_behavior &commutation_behavior) {
  auto it = matrix_operator::defined_ops.find(operator_id);
  if (it == matrix_operator::defined_ops.end())
    throw std::range_error("not matrix operator with the name '" + operator_id +
                           "' has been defined");
  auto application_degrees = degrees;
  std::sort(application_degrees.begin(), application_degrees.end(),
            operator_handler::user_facing_order);
  if (application_degrees != degrees) {
    std::stringstream err_msg;
    err_msg << "incorrect ordering of degrees (expected order {"
            << application_degrees[0];
    for (auto i = 1; i < application_degrees.size(); ++i)
      err_msg << ", " << std::to_string(application_degrees[i]);
    err_msg << "})";
    throw std::runtime_error(err_msg.str());
  }
  return product_operator(
      matrix_operator(operator_id, degrees, commutation_behavior));
}

product_operator<matrix_operator>
matrix_operator::instantiate(std::string operator_id,
                             std::vector<int> &&degrees,
                             const commutation_behavior &commutation_behavior) {
  auto it = matrix_operator::defined_ops.find(operator_id);
  if (it == matrix_operator::defined_ops.end())
    throw std::range_error("not matrix operator with the name '" + operator_id +
                           "' has been defined");
  auto application_degrees = degrees;
  std::sort(application_degrees.begin(), application_degrees.end(),
            operator_handler::user_facing_order);
  if (application_degrees != degrees) {
    std::stringstream err_msg;
    err_msg << "incorrect ordering of degrees (expected order {"
            << application_degrees[0];
    for (auto i = 1; i < application_degrees.size(); ++i)
      err_msg << ", " << std::to_string(application_degrees[i]);
    err_msg << "})";
    throw std::runtime_error(err_msg.str());
  }
  return product_operator(
      matrix_operator(operator_id, std::move(degrees), commutation_behavior));
}

// private helpers

std::string matrix_operator::op_code_to_string(
    std::unordered_map<int, int> &dimensions) const {
  auto it = matrix_operator::defined_ops.find(this->op_code);
  assert(it != matrix_operator::defined_ops
                   .end()); // should be validated upon instantiation

  for (auto i = 0; i < this->targets.size(); ++i) {
    auto entry = dimensions.find(this->targets[i]);
    auto expected_dim = it->second.expected_dimensions[i];
    if (expected_dim <= 0) {
      if (entry == dimensions.end())
        throw std::runtime_error("missing dimension for degree " +
                                 std::to_string(this->targets[i]));
    } else {
      if (entry == dimensions.end())
        dimensions[this->targets[i]] = expected_dim;
      else if (entry->second != expected_dim)
        throw std::runtime_error(
            "invalid dimension for degree " + std::to_string(this->targets[i]) +
            ", expected dimension is " + std::to_string(expected_dim));
    }
  }
  return this->op_code;
}

// read-only properties

std::string matrix_operator::unique_id() const {
  auto it = this->targets.cbegin();
  auto str = this->op_code + std::to_string(*it);
  while (++it != this->targets.cend())
    str += "." + std::to_string(*it);
  return std::move(str);
}

std::vector<int> matrix_operator::degrees() const { return this->targets; }

// constructors

matrix_operator::matrix_operator(int degree)
    : op_code("I"), commutes(true),
      group(operator_handler::default_commutation_relations) {
  this->targets.push_back(degree);
  if (matrix_operator::defined_ops.find(this->op_code) ==
      matrix_operator::defined_ops.end()) {
    auto func =
        [](const std::vector<int> &dimensions,
           const std::unordered_map<std::string, std::complex<double>> &_none) {
          std::size_t dimension = dimensions[0];
          auto mat = complex_matrix(dimension, dimension);

          // Build up the identity matrix.
          for (std::size_t i = 0; i < dimension; i++) {
            mat[{i, i}] = std::complex<double>(1.0);
          }
          return mat;
        };
    matrix_operator::define(this->op_code, {-1}, std::move(func));
  }
}

matrix_operator::matrix_operator(
    std::string operator_id, const std::vector<int> &degrees,
    const commutation_behavior &commutation_behavior)
    : op_code(operator_id),
      commutes(commutation_behavior.commutes_across_degrees),
      group(commutation_behavior.group), targets(degrees) {
  assert(this->targets.size() > 0);
  if (!commutation_behavior.commutes_across_degrees && this->targets.size() > 1)
    // We cannot support this with the current mechanism for achieving
    // non-trivial commutation relations for operators acting on different
    // degrees. See also the comment in the `find_insert_at` template for
    // product operators. We still want to stick with that mechanism, since it
    // is more general and by far more performant than e.g. achieving
    // anti-commutation via phase operator instead. It should be fine, however,
    // for a multi-qubit operator to belong to a non-zero commutation set as
    // long as the operator itself commutes with all operators acting on
    // different degrees (as indicated by teh boolean value of
    // commutation_behavior); this effectively "marks" the degrees that the
    // operator acts on as being a certain kind of particles.
    throw std::runtime_error("non-trivial commutation behavior is not "
                             "supported for multi-target operators");
}

matrix_operator::matrix_operator(
    std::string operator_id, std::vector<int> &&degrees,
    const commutation_behavior &commutation_behavior)
    : op_code(operator_id),
      commutes(commutation_behavior.commutes_across_degrees),
      group(commutation_behavior.group), targets(std::move(degrees)) {
  assert(this->targets.size() > 0);
  if (!commutation_behavior.commutes_across_degrees && this->targets.size() > 1)
    // We cannot support this with the current mechanism for achieving
    // non-trivial commutation relations for operators acting on different
    // degrees. See also the comment in the `find_insert_at` template for
    // product operators. We still want to stick with that mechanism, since it
    // is more general and by far more performant than e.g. achieving
    // anti-commutation via phase operator instead. It should be fine, however,
    // for a multi-qubit operator to belong to a non-zero commutation set as
    // long as the operator itself commutes with all operators acting on
    // different degrees (as indicated by teh boolean value of
    // commutation_behavior); this effectively "marks" the degrees that the
    // operator acts on as being a certain kind of particles.
    throw std::runtime_error("non-trivial commutation behavior is not "
                             "supported for multi-target operators");
}

template <typename T,
          std::enable_if_t<std::is_base_of_v<operator_handler, T>, bool>>
matrix_operator::matrix_operator(const T &other)
    : matrix_operator::matrix_operator(
          other, commutation_behavior(other.commutation_group,
                                      other.commutes_across_degrees)) {}

template <typename T,
          std::enable_if_t<std::is_base_of_v<operator_handler, T>, bool>>
matrix_operator::matrix_operator(const T &other,
                                 const commutation_behavior &behavior)
    : op_code(matrix_operator::type_prefix<T>() + other.to_string(false)),
      commutes(behavior.commutes_across_degrees), group(behavior.group),
      targets(other.degrees()) {
  if (matrix_operator::defined_ops.find(this->op_code) ==
      matrix_operator::defined_ops.end()) {
    auto func = [other](const std::vector<int> &dimensions,
                        const std::unordered_map<std::string,
                                                 std::complex<double>> &_none) {
      std::unordered_map<int, int> dims;
      auto targets = other.degrees();
      for (auto i = 0; i < dimensions.size(); ++i)
        dims[targets[i]] = dimensions[i];
      return other.to_matrix(dims, std::move(_none));
    };
    // the to_matrix method on the spin op will check the dimensions, so we
    // allow arbitrary here
    std::vector<int> required_dimensions(this->targets.size(), -1);
    matrix_operator::define(this->op_code, std::move(required_dimensions),
                            func);
  }
}

template matrix_operator::matrix_operator(const spin_operator &other);
template matrix_operator::matrix_operator(const boson_operator &other);
template matrix_operator::matrix_operator(const fermion_operator &other);

template matrix_operator::matrix_operator(const spin_operator &other,
                                          const commutation_behavior &behavior);
template matrix_operator::matrix_operator(const boson_operator &other,
                                          const commutation_behavior &behavior);
template matrix_operator::matrix_operator(const fermion_operator &other,
                                          const commutation_behavior &behavior);

matrix_operator::matrix_operator(const matrix_operator &other)
    : op_code(other.op_code), commutes(other.commutes), group(other.group),
      targets(other.targets) {}

matrix_operator::matrix_operator(matrix_operator &&other)
    : op_code(other.op_code), commutes(other.commutes),
      group(std::move(other.group)), targets(std::move(other.targets)) {}

// assignments

matrix_operator &matrix_operator::operator=(matrix_operator &&other) {
  if (this != &other) {
    this->op_code = other.op_code;
    this->commutes = other.commutes;
    this->group = std::move(other.group);
    this->targets = std::move(other.targets);
  }
  return *this;
}

matrix_operator &matrix_operator::operator=(const matrix_operator &other) {
  if (this != &other) {
    this->op_code = other.op_code;
    this->commutes = other.commutes;
    this->group = other.group;
    this->targets = other.targets;
  }
  return *this;
}

template <typename T,
          std::enable_if_t<!std::is_same<T, matrix_operator>::value &&
                               std::is_base_of_v<operator_handler, T>,
                           bool>>
matrix_operator &matrix_operator::operator=(const T &other) {
  *this = matrix_operator(other);
  return *this;
}

template matrix_operator &
matrix_operator::operator=(const spin_operator &other);
template matrix_operator &
matrix_operator::operator=(const boson_operator &other);
template matrix_operator &
matrix_operator::operator=(const fermion_operator &other);

// evaluations

complex_matrix matrix_operator::to_matrix(
    std::unordered_map<int, int> &dimensions,
    const std::unordered_map<std::string, std::complex<double>> &parameters)
    const {
  auto it = matrix_operator::defined_ops.find(this->op_code);
  assert(it != matrix_operator::defined_ops
                   .end()); // should be validated upon instantiation

  std::vector<int> relevant_dimensions;
  relevant_dimensions.reserve(this->targets.size());
  for (auto i = 0; i < this->targets.size(); ++i) {
    auto entry = dimensions.find(this->targets[i]);
    auto expected_dim = it->second.expected_dimensions[i];
    if (expected_dim <= 0) {
      if (entry == dimensions.end())
        throw std::runtime_error("missing dimension for degree " +
                                 std::to_string(this->targets[i]));
      relevant_dimensions.push_back(entry->second);
    } else {
      if (entry == dimensions.end())
        dimensions[this->targets[i]] = expected_dim;
      else if (entry->second != expected_dim)
        throw std::runtime_error(
            "invalid dimension for degree " + std::to_string(this->targets[i]) +
            ", expected dimension is " + std::to_string(expected_dim));
      relevant_dimensions.push_back(expected_dim);
    }
  }

  return it->second.generate_matrix(relevant_dimensions, parameters);
}

std::string matrix_operator::to_string(bool include_degrees) const {
  if (!include_degrees)
    return this->op_code;
  else if (this->targets.size() == 0)
    return this->op_code + "()";
  auto it = this->targets.cbegin();
  std::string str = this->op_code + "(" + std::to_string(*it);
  while (++it != this->targets.cend())
    str += ", " + std::to_string(*it);
  return str + ")";
}

// comparisons

bool matrix_operator::operator==(const matrix_operator &other) const {
  return this->op_code == other.op_code && this->group == other.group &&
         // no need to compare commutes (should be determined by op_code and
         // commutation group)
         this->targets == other.targets;
}

// predefined operators

operator_sum<matrix_operator> matrix_operator::empty() {
  return operator_handler::empty<matrix_operator>();
}

product_operator<matrix_operator> matrix_operator::identity() {
  return operator_handler::identity<matrix_operator>();
}

product_operator<matrix_operator> matrix_operator::identity(int degree) {
  return product_operator(matrix_operator(degree));
}

product_operator<matrix_operator> matrix_operator::number(int degree) {
  std::string op_code = "number";
  if (matrix_operator::defined_ops.find(op_code) ==
      matrix_operator::defined_ops.end()) {
    auto func =
        [](const std::vector<int> &dimensions,
           const std::unordered_map<std::string, std::complex<double>> &_none) {
          std::size_t dimension = dimensions[0];
          auto mat = complex_matrix(dimension, dimension);
          for (std::size_t i = 0; i < dimension; i++) {
            mat[{i, i}] = std::complex<double>(i);
          }
          return mat;
        };
    matrix_operator::define(op_code, {-1}, func);
  }
  auto op = matrix_operator(op_code, {degree});
  return product_operator(std::move(op));
}

product_operator<matrix_operator> matrix_operator::parity(int degree) {
  std::string op_code = "parity";
  if (matrix_operator::defined_ops.find(op_code) ==
      matrix_operator::defined_ops.end()) {
    auto func =
        [](const std::vector<int> &dimensions,
           const std::unordered_map<std::string, std::complex<double>> &_none) {
          std::size_t dimension = dimensions[0];
          auto mat = complex_matrix(dimension, dimension);
          for (std::size_t i = 0; i < dimension; i++) {
            mat[{i, i}] = i & 1 ? -1. : 1.;
          }
          return mat;
        };
    matrix_operator::define(op_code, {-1}, func);
  }
  auto op = matrix_operator(op_code, {degree});
  return product_operator(std::move(op));
}

product_operator<matrix_operator> matrix_operator::position(int degree) {
  std::string op_code = "position";
  if (matrix_operator::defined_ops.find(op_code) ==
      matrix_operator::defined_ops.end()) {
    auto func =
        [](const std::vector<int> &dimensions,
           const std::unordered_map<std::string, std::complex<double>> &_none) {
          std::size_t dimension = dimensions[0];
          auto mat = complex_matrix(dimension, dimension);
          // position = 0.5 * (create + annihilate)
          for (std::size_t i = 0; i + 1 < dimension; i++) {
            mat[{i + 1, i}] = 0.5 * std::sqrt(static_cast<double>(i + 1));
            mat[{i, i + 1}] = 0.5 * std::sqrt(static_cast<double>(i + 1));
          }
          return mat;
        };
    matrix_operator::define(op_code, {-1}, func);
  }
  auto op = matrix_operator(op_code, {degree});
  return product_operator(std::move(op));
}

product_operator<matrix_operator> matrix_operator::momentum(int degree) {
  std::string op_code = "momentum";
  if (matrix_operator::defined_ops.find(op_code) ==
      matrix_operator::defined_ops.end()) {
    auto func =
        [](const std::vector<int> &dimensions,
           const std::unordered_map<std::string, std::complex<double>> &_none) {
          std::size_t dimension = dimensions[0];
          auto mat = complex_matrix(dimension, dimension);
          // momentum = 0.5j * (create - annihilate)
          for (std::size_t i = 0; i + 1 < dimension; i++) {
            mat[{i + 1, i}] = std::complex<double>(0., 0.5) *
                              std::sqrt(static_cast<double>(i + 1));
            mat[{i, i + 1}] = std::complex<double>(0., -0.5) *
                              std::sqrt(static_cast<double>(i + 1));
          }
          return mat;
        };
    matrix_operator::define(op_code, {-1}, func);
  }
  auto op = matrix_operator(op_code, {degree});
  return product_operator(std::move(op));
}

product_operator<matrix_operator> matrix_operator::displace(int degree) {
  std::string op_code = "displace";
  if (matrix_operator::defined_ops.find(op_code) ==
      matrix_operator::defined_ops.end()) {
    auto func = [](const std::vector<int> &dimensions,
                   const std::unordered_map<std::string, std::complex<double>>
                       &parameters) {
      std::size_t dimension = dimensions[0];
      auto entry = parameters.find("displacement");
      if (entry == parameters.end())
        throw std::runtime_error("missing value for parameter 'displacement'");
      auto displacement_amplitude = entry->second;
      auto create = complex_matrix(dimension, dimension);
      auto annihilate = complex_matrix(dimension, dimension);
      for (std::size_t i = 0; i + 1 < dimension; i++) {
        create[{i + 1, i}] = std::sqrt(static_cast<double>(i + 1));
        annihilate[{i, i + 1}] = std::sqrt(static_cast<double>(i + 1));
      }
      auto term1 = displacement_amplitude * create;
      auto term2 = std::conj(displacement_amplitude) * annihilate;
      return (term1 - term2).exponential();
    };
    matrix_operator::define(op_code, {-1}, func);
  }
  auto op = matrix_operator(op_code, {degree});
  return product_operator(std::move(op));
}

product_operator<matrix_operator> matrix_operator::squeeze(int degree) {
  std::string op_code = "squeeze";
  if (matrix_operator::defined_ops.find(op_code) ==
      matrix_operator::defined_ops.end()) {
    auto func = [](const std::vector<int> &dimensions,
                   const std::unordered_map<std::string, std::complex<double>>
                       &parameters) {
      std::size_t dimension = dimensions[0];
      auto entry = parameters.find("squeezing");
      if (entry == parameters.end())
        throw std::runtime_error("missing value for parameter 'squeezing'");
      auto squeezing = entry->second;
      auto create = complex_matrix(dimension, dimension);
      auto annihilate = complex_matrix(dimension, dimension);
      for (std::size_t i = 0; i + 1 < dimension; i++) {
        create[{i + 1, i}] = std::sqrt(static_cast<double>(i + 1));
        annihilate[{i, i + 1}] = std::sqrt(static_cast<double>(i + 1));
      }
      auto term1 = std::conj(squeezing) * annihilate.power(2);
      auto term2 = squeezing * create.power(2);
      auto difference = 0.5 * (term1 - term2);
      return difference.exponential();
    };
    matrix_operator::define(op_code, {-1}, func);
  }
  auto op = matrix_operator(op_code, {degree});
  return product_operator(std::move(op));
}

// tools for custom operators

} // namespace cudaq