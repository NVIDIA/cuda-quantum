/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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

#include "cudaq/boson_op.h"
#include "cudaq/fermion_op.h"
#include "cudaq/matrix_op.h"
#include "cudaq/spin_op.h"

namespace cudaq {

#if !defined(NDEBUG)
bool matrix_handler::can_be_canonicalized = false;
#endif

// tools for custom operators

std::unordered_map<std::string, Definition> matrix_handler::defined_ops = {};

template <typename T>
std::string matrix_handler::type_prefix() {
  return typeid(T).name();
}

// no need to prefix the operator id and op code with the type name for these
// (same names mean the same thing)
template <>
std::string matrix_handler::type_prefix<spin_handler>() {
  return "";
}
template <>
std::string matrix_handler::type_prefix<boson_handler>() {
  return "";
}
template <>
std::string matrix_handler::type_prefix<fermion_handler>() {
  return "";
}

void matrix_handler::define(
    std::string operator_id, std::vector<std::int64_t> expected_dimensions,
    matrix_callback &&create,
    std::unordered_map<std::string, std::string> &&parameter_descriptions) {
  auto defn = Definition(operator_id, std::move(expected_dimensions),
                         std::move(create), std::move(parameter_descriptions));
  auto result =
      matrix_handler::defined_ops.insert({operator_id, std::move(defn)});
  if (!result.second)
    throw std::runtime_error("a matrix operator with name " + operator_id +
                             " is already defined");
}

void matrix_handler::define(std::string operator_id,
                            std::vector<int64_t> expected_dimensions,
                            matrix_callback &&create,
                            const std::unordered_map<std::string, std::string>
                                &parameter_descriptions) {
  matrix_handler::define(
      std::move(operator_id), std::move(expected_dimensions), std::move(create),
      std::unordered_map<std::string, std::string>(parameter_descriptions));
}

void matrix_handler::define(std::string operator_id,
                            std::vector<std::int64_t> expected_dimensions,
                            matrix_callback &&create,
                            diag_matrix_callback &&diag_create,
                            const std::unordered_map<std::string, std::string>
                                &parameter_descriptions) {
  auto defn = Definition(
      operator_id, std::move(expected_dimensions), std::move(create),
      std::move(diag_create),
      std::unordered_map<std::string, std::string>(parameter_descriptions));
  auto result =
      matrix_handler::defined_ops.insert({operator_id, std::move(defn)});
  if (!result.second)
    throw std::runtime_error("a matrix operator with name " + operator_id +
                             " is already defined");
}

bool matrix_handler::remove_definition(const std::string &operator_id) {
  return matrix_handler::defined_ops.erase(operator_id);
}

product_op<matrix_handler>
matrix_handler::instantiate(std::string operator_id,
                            const std::vector<std::size_t> &degrees,
                            const commutation_behavior &commutation_behavior) {
  return matrix_handler::instantiate(
      std::move(operator_id),
      std::vector<std::size_t>(degrees.cbegin(), degrees.cend()),
      commutation_behavior);
}

product_op<matrix_handler>
matrix_handler::instantiate(std::string operator_id,
                            std::vector<std::size_t> &&degrees,
                            const commutation_behavior &commutation_behavior) {
  auto it = matrix_handler::defined_ops.find(operator_id);
  if (it == matrix_handler::defined_ops.end())
    throw std::range_error("not matrix operator with the name '" + operator_id +
                           "' has been defined");
  auto application_degrees = degrees;
  std::sort(application_degrees.begin(), application_degrees.end(),
            operator_handler::canonical_order);
  if (application_degrees != degrees) {
    std::stringstream err_msg;
    err_msg << "incorrect ordering of degrees (expected order {"
            << application_degrees[0];
    for (auto i = 1; i < application_degrees.size(); ++i)
      err_msg << ", " << std::to_string(application_degrees[i]);
    err_msg << "})";
    throw std::runtime_error(err_msg.str());
  }
  return product_op(matrix_handler(std::move(operator_id), std::move(degrees),
                                   commutation_behavior));
}

const std::unordered_map<std::string, std::string> &
matrix_handler::get_parameter_descriptions() const {
  auto it = matrix_handler::defined_ops.find(this->op_code);
  assert(it != matrix_handler::defined_ops.end());
  return it->second.parameter_descriptions;
}

const std::vector<int64_t> &matrix_handler::get_expected_dimensions() const {
  auto it = matrix_handler::defined_ops.find(this->op_code);
  assert(it != matrix_handler::defined_ops.end());
  return it->second.expected_dimensions;
}

// private helpers

std::string matrix_handler::canonical_form(
    std::unordered_map<std::size_t, std::int64_t> &dimensions,
    std::vector<std::int64_t> &relevant_dims) const {
  auto it = matrix_handler::defined_ops.find(this->op_code);
  assert(it != matrix_handler::defined_ops
                   .end()); // should be validated upon instantiation

  for (auto i = 0; i < this->targets.size(); ++i) {
    auto entry = dimensions.find(this->targets[i]);
    auto expected_dim = it->second.expected_dimensions[i];
    if (expected_dim <= 0) {
      if (entry == dimensions.end())
        throw std::runtime_error("missing dimension for degree " +
                                 std::to_string(this->targets[i]));
      relevant_dims.push_back(entry->second);
    } else {
      if (entry == dimensions.end())
        dimensions[this->targets[i]] = expected_dim;
      else if (entry->second != expected_dim)
        throw std::runtime_error(
            "invalid dimension for degree " + std::to_string(this->targets[i]) +
            ", expected dimension is " + std::to_string(expected_dim));
      relevant_dims.push_back(expected_dim);
    }
  }
  return this->op_code;
}

// read-only properties

std::string matrix_handler::unique_id() const {
  if (this->targets.size() == 0)
    return this->op_code;
  auto it = this->targets.cbegin();
  std::string str = this->op_code + "(" + std::to_string(*it);
  while (++it != this->targets.cend())
    str += "," + std::to_string(*it);
  return str + ")";
}

std::vector<std::size_t> matrix_handler::degrees() const {
  return this->targets;
}

// constructors

matrix_handler::matrix_handler(std::size_t degree)
    : op_code("I"), commutes(true),
      group(operator_handler::default_commutation_relations) {
  this->targets.push_back(degree);
  if (matrix_handler::defined_ops.find(this->op_code) ==
      matrix_handler::defined_ops.end()) {
    auto func =
        [](const std::vector<std::int64_t> &dimensions,
           const std::unordered_map<std::string, std::complex<double>> &_none) {
          std::size_t dimension = dimensions[0];
          auto mat = complex_matrix(dimension, dimension);

          // Build up the identity matrix.
          for (std::size_t i = 0; i < dimension; i++) {
            mat[{i, i}] = std::complex<double>(1.0);
          }
          return mat;
        };
    matrix_handler::define(this->op_code, {-1}, std::move(func));
  }
}

matrix_handler::matrix_handler(std::string operator_id,
                               const std::vector<std::size_t> &degrees,
                               const commutation_behavior &commutation_behavior)
    : op_code(operator_id),
      commutes(commutation_behavior.commutes_across_degrees),
      group(commutation_behavior.group), targets(degrees) {
  if (!commutation_behavior.commutes_across_degrees && this->targets.size() > 1)
    // We cannot support this with the current mechanism for achieving
    // non-trivial commutation relations for operators acting on different
    // degrees. See also the comment in the `find_insert_at` template for
    // product operators. We still want to stick with that mechanism, since it
    // is more general and by far more performant than e.g. achieving
    // anti-commutation via phase operator instead. It should be fine, however,
    // for a multi-qubit operator to belong to a non-zero commutation set as
    // long as the operator itself commutes with all operators acting on
    // different degrees (as indicated by the boolean value of
    // commutation_behavior); this effectively "marks" the degrees that the
    // operator acts on as being a certain kind of particles.
    throw std::runtime_error("non-trivial commutation behavior is not "
                             "supported for multi-target operators");
}

matrix_handler::matrix_handler(std::string operator_id,
                               std::vector<std::size_t> &&degrees,
                               const commutation_behavior &commutation_behavior)
    : op_code(operator_id),
      commutes(commutation_behavior.commutes_across_degrees),
      group(commutation_behavior.group), targets(std::move(degrees)) {
  if (!commutation_behavior.commutes_across_degrees && this->targets.size() > 1)
    // We cannot support this with the current mechanism for achieving
    // non-trivial commutation relations for operators acting on different
    // degrees. See also the comment in the `find_insert_at` template for
    // product operators. We still want to stick with that mechanism, since it
    // is more general and by far more performant than e.g. achieving
    // anti-commutation via phase operator instead. It should be fine, however,
    // for a multi-qubit operator to belong to a non-zero commutation set as
    // long as the operator itself commutes with all operators acting on
    // different degrees (as indicated by the boolean value of
    // commutation_behavior); this effectively "marks" the degrees that the
    // operator acts on as being a certain kind of particles.
    throw std::runtime_error("non-trivial commutation behavior is not "
                             "supported for multi-target operators");
}

template <typename T,
          std::enable_if_t<std::is_base_of_v<operator_handler, T>, bool>>
matrix_handler::matrix_handler(const T &other)
    : matrix_handler::matrix_handler(
          other, commutation_behavior(other.commutation_group,
                                      other.commutes_across_degrees)) {}

template <typename T,
          std::enable_if_t<std::is_base_of_v<operator_handler, T>, bool>>
matrix_handler::matrix_handler(const T &other,
                               const commutation_behavior &behavior)
    : op_code(matrix_handler::type_prefix<T>() + other.to_string(false)),
      commutes(behavior.commutes_across_degrees), group(behavior.group),
      targets(other.degrees()) {
  if (matrix_handler::defined_ops.find(this->op_code) ==
      matrix_handler::defined_ops.end()) {
    auto func = [other](const std::vector<std::int64_t> &dimensions,
                        const std::unordered_map<std::string,
                                                 std::complex<double>> &_none) {
      std::unordered_map<std::size_t, std::int64_t> dims;
      auto targets = other.degrees();
      for (auto i = 0; i < dimensions.size(); ++i)
        dims[targets[i]] = dimensions[i];
      return other.to_matrix(dims, _none);
    };
    // the to_matrix method on the spin op will check the dimensions, so we
    // allow arbitrary here
    std::vector<std::int64_t> required_dimensions(this->targets.size(), -1);

    if constexpr (std::is_base_of_v<mdiag_operator_handler, T>) {
      auto dia_func =
          [other](const std::vector<std::int64_t> &dimensions,
                  const std::unordered_map<std::string, std::complex<double>>
                      &_none) {
            std::unordered_map<std::size_t, std::int64_t> dims;
            auto targets = other.degrees();
            for (auto i = 0; i < dimensions.size(); ++i)
              dims[targets[i]] = dimensions[i];
            return other.to_diagonal_matrix(dims, _none);
          };
      matrix_handler::define(this->op_code, std::move(required_dimensions),
                             func, std::move(dia_func));
    } else {
      matrix_handler::define(this->op_code, std::move(required_dimensions),
                             func);
    }
  }
}

template matrix_handler::matrix_handler(const spin_handler &other);
template matrix_handler::matrix_handler(const boson_handler &other);
template matrix_handler::matrix_handler(const fermion_handler &other);

template matrix_handler::matrix_handler(const spin_handler &other,
                                        const commutation_behavior &behavior);
template matrix_handler::matrix_handler(const boson_handler &other,
                                        const commutation_behavior &behavior);
template matrix_handler::matrix_handler(const fermion_handler &other,
                                        const commutation_behavior &behavior);

matrix_handler::matrix_handler(const matrix_handler &other)
    : op_code(other.op_code), commutes(other.commutes), group(other.group),
      targets(other.targets) {}

matrix_handler::matrix_handler(matrix_handler &&other)
    : op_code(other.op_code), commutes(other.commutes),
      group(std::move(other.group)), targets(std::move(other.targets)) {}

// assignments

matrix_handler &matrix_handler::operator=(matrix_handler &&other) {
  if (this != &other) {
    this->op_code = other.op_code;
    this->commutes = other.commutes;
    this->group = std::move(other.group);
    this->targets = std::move(other.targets);
  }
  return *this;
}

matrix_handler &matrix_handler::operator=(const matrix_handler &other) {
  if (this != &other) {
    this->op_code = other.op_code;
    this->commutes = other.commutes;
    this->group = other.group;
    this->targets = other.targets;
  }
  return *this;
}

template <typename T,
          std::enable_if_t<!std::is_same<T, matrix_handler>::value &&
                               std::is_base_of_v<operator_handler, T>,
                           bool>>
matrix_handler &matrix_handler::operator=(const T &other) {
  *this = matrix_handler(other);
  return *this;
}

template matrix_handler &matrix_handler::operator=(const spin_handler &other);
template matrix_handler &matrix_handler::operator=(const boson_handler &other);
template matrix_handler &
matrix_handler::operator=(const fermion_handler &other);

// evaluations

complex_matrix matrix_handler::to_matrix(
    std::unordered_map<std::size_t, std::int64_t> &dimensions,
    const std::unordered_map<std::string, std::complex<double>> &parameters)
    const {
  auto it = matrix_handler::defined_ops.find(this->op_code);
  assert(it != matrix_handler::defined_ops
                   .end()); // should be validated upon instantiation

  std::vector<std::int64_t> relevant_dimensions;
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

mdiag_sparse_matrix matrix_handler::to_diagonal_matrix(
    std::unordered_map<std::size_t, std::int64_t> &dimensions,
    const std::unordered_map<std::string, std::complex<double>> &parameters)
    const {
  auto it = matrix_handler::defined_ops.find(this->op_code);
  assert(it != matrix_handler::defined_ops
                   .end()); // should be validated upon instantiation

  if (!it->second.has_dia_generator())
    return mdiag_sparse_matrix();
  std::vector<std::int64_t> relevant_dimensions;
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

  return it->second.generate_dia_matrix(relevant_dimensions, parameters);
}

std::string matrix_handler::to_string(bool include_degrees) const {
  if (include_degrees)
    return this->unique_id(); // unique id for consistency with keys in some
                              // user facing maps
  else
    return this->op_code;
}

// comparisons

bool matrix_handler::operator==(const matrix_handler &other) const {
  return this->op_code == other.op_code && this->group == other.group &&
         // no need to compare commutes (should be determined by op_code and
         // commutation group)
         this->targets == other.targets;
}

// predefined operators

matrix_handler matrix_handler::number(std::size_t degree) {
  std::string op_code = "number";
  if (matrix_handler::defined_ops.find(op_code) ==
      matrix_handler::defined_ops.end()) {
    auto func =
        [](const std::vector<std::int64_t> &dimensions,
           const std::unordered_map<std::string, std::complex<double>> &_none) {
          std::size_t dimension = dimensions[0];
          auto mat = complex_matrix(dimension, dimension);
          for (std::size_t i = 0; i < dimension; i++) {
            mat[{i, i}] = std::complex<double>(i);
          }
          return mat;
        };
    matrix_handler::define(op_code, {-1}, func);
  }
  return matrix_handler(op_code, {degree});
}

matrix_handler matrix_handler::parity(std::size_t degree) {
  std::string op_code = "parity";
  if (matrix_handler::defined_ops.find(op_code) ==
      matrix_handler::defined_ops.end()) {
    auto func =
        [](const std::vector<std::int64_t> &dimensions,
           const std::unordered_map<std::string, std::complex<double>> &_none) {
          std::size_t dimension = dimensions[0];
          auto mat = complex_matrix(dimension, dimension);
          for (std::size_t i = 0; i < dimension; i++) {
            mat[{i, i}] = i & 1 ? -1. : 1.;
          }
          return mat;
        };
    matrix_handler::define(op_code, {-1}, func);
  }
  return matrix_handler(op_code, {degree});
}

matrix_handler matrix_handler::position(std::size_t degree) {
  std::string op_code = "position";
  if (matrix_handler::defined_ops.find(op_code) ==
      matrix_handler::defined_ops.end()) {
    auto func =
        [](const std::vector<std::int64_t> &dimensions,
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
    matrix_handler::define(op_code, {-1}, func);
  }
  return matrix_handler(op_code, {degree});
}

matrix_handler matrix_handler::momentum(std::size_t degree) {
  std::string op_code = "momentum";
  if (matrix_handler::defined_ops.find(op_code) ==
      matrix_handler::defined_ops.end()) {
    auto func =
        [](const std::vector<std::int64_t> &dimensions,
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
    matrix_handler::define(op_code, {-1}, func);
  }
  return matrix_handler(op_code, {degree});
}

matrix_handler matrix_handler::displace(std::size_t degree) {
  std::string op_code = "displace";
  if (matrix_handler::defined_ops.find(op_code) ==
      matrix_handler::defined_ops.end()) {
    auto func = [](const std::vector<std::int64_t> &dimensions,
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
    matrix_handler::define(
        op_code, {-1}, func,
        {{"displacement",
          "Amplitude of the displacement operator. See also "
          "https://en.wikipedia.org/wiki/Displacement_operator."}});
  }
  return matrix_handler(op_code, {degree});
}

matrix_handler matrix_handler::squeeze(std::size_t degree) {
  std::string op_code = "squeeze";
  if (matrix_handler::defined_ops.find(op_code) ==
      matrix_handler::defined_ops.end()) {
    auto func = [](const std::vector<std::int64_t> &dimensions,
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
    matrix_handler::define(
        op_code, {-1}, func,
        {{"squeezing", "Amplitude of the squeezing operator. See also "
                       "https://en.wikipedia.org/wiki/Squeeze_operator."}});
  }
  return matrix_handler(op_code, {degree});
}

namespace operators {
product_op<matrix_handler> number(std::size_t target) {
  return product_op(matrix_handler::number(target));
}
product_op<matrix_handler> parity(std::size_t target) {
  return product_op(matrix_handler::parity(target));
}
product_op<matrix_handler> position(std::size_t target) {
  return product_op(matrix_handler::position(target));
}
product_op<matrix_handler> momentum(std::size_t target) {
  return product_op(matrix_handler::momentum(target));
}
product_op<matrix_handler> squeeze(std::size_t target) {
  return product_op(matrix_handler::squeeze(target));
}
product_op<matrix_handler> displace(std::size_t target) {
  return product_op(matrix_handler::displace(target));
}
} // namespace operators

} // namespace cudaq
