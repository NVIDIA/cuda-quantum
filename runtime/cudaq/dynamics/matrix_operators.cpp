/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/operators.h"
#include "matrix_operators.h"
#include "spin_operators.h"

#include <complex>
#include <set>

namespace cudaq {

// tools for custom operators

std::map<std::string, Definition> matrix_operator::m_ops = {};

void matrix_operator::define(std::string operator_id, std::vector<int> expected_dimensions,
            MatrixCallbackFunction &&create) {
  auto defn = Definition(operator_id, expected_dimensions, std::forward<MatrixCallbackFunction>(create));
  auto result = matrix_operator::m_ops.insert({operator_id, std::move(defn)});
  if (!result.second) {
    throw std::runtime_error("an matrix operator with name " + operator_id + "is already defined");
  }
}

product_operator<matrix_operator> matrix_operator::create(std::string operator_id, const std::vector<int> &degrees) {
  auto it = matrix_operator::m_ops.find(operator_id);
  if (it == matrix_operator::m_ops.end()) 
    throw std::range_error("not matrix operator with the name '" + operator_id + "' has been defined");
  return product_operator(matrix_operator(operator_id, degrees));
}

product_operator<matrix_operator> matrix_operator::create(std::string operator_id, std::vector<int> &&degrees) {
  auto it = matrix_operator::m_ops.find(operator_id);
  if (it == matrix_operator::m_ops.end()) 
    throw std::range_error("not matrix operator with the name '" + operator_id + "' has been defined");
  return product_operator(matrix_operator(operator_id, std::move(degrees)));
}

// read-only properties

std::vector<int> matrix_operator::degrees() const {
  return this->targets;
}

bool matrix_operator::is_identity() const {
  return this->id == "identity";
}

// constructors

matrix_operator::matrix_operator(std::string operator_id, const std::vector<int> &degrees)
  : id(operator_id), targets(degrees) {
    assert(this->targets.size() > 0);
  }

matrix_operator::matrix_operator(std::string operator_id, std::vector<int> &&degrees)
  : id(operator_id), targets(std::move(degrees)) {
    assert(this->targets.size() > 0);
  }

template<typename T, std::enable_if_t<std::is_base_of_v<operator_handler, T>, bool>>
matrix_operator::matrix_operator(const T &other) {
  this->targets = other.degrees();
  this->id = typeid(other).name() + std::to_string(this->targets.size()) + other.to_string(false);
  if (matrix_operator::m_ops.find(this->id) == matrix_operator::m_ops.end()) {
    auto func = [targets = other.degrees(), other](std::vector<int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      std::map<int, int> dims;
      for(auto i = 0; i < dimensions.size(); ++i)
        dims[targets[i]] = dimensions[i];
      return other.to_matrix(dims, std::move(_none));
    };
    // the to_matrix method on the spin op will check the dimensions, so we allow arbitrary here
    std::vector<int> required_dimensions (this->targets.size(), -1);
    matrix_operator::define(this->id, std::move(required_dimensions), func);
  }
}

template matrix_operator::matrix_operator(const spin_operator &other);

matrix_operator::matrix_operator(const matrix_operator &other)
  : targets(other.targets), id(other.id) {}

matrix_operator::matrix_operator(matrix_operator &&other) 
  : targets(std::move(other.targets)), id(other.id) {}

// assignments

matrix_operator& matrix_operator::operator=(const matrix_operator& other) {
  if (this != &other) {
    this->targets = other.targets;
    this->id = other.id;
  }
  return *this;
}

template<typename T, std::enable_if_t<!std::is_same<T, matrix_operator>::value && std::is_base_of_v<operator_handler, T>, bool>>
matrix_operator& matrix_operator::operator=(const T& other) {
  *this = matrix_operator(other);
  return *this;
}

template matrix_operator& matrix_operator::operator=(const spin_operator& other);

matrix_operator& matrix_operator::operator=(matrix_operator &&other) {
  if (this != &other) {
    this->targets = std::move(other.targets);
    this->id = other.id;  
  }
  return *this;
}

// evaluations

matrix_2 matrix_operator::to_matrix(
    std::map<int, int> &dimensions,
    std::map<std::string, std::complex<double>> parameters) const {
  auto it = matrix_operator::m_ops.find(this->id);
  if (it == matrix_operator::m_ops.end()) 
    throw std::range_error("unable to find operator");

  std::vector<int> relevant_dimensions;
  relevant_dimensions.reserve(this->targets.size());
  for (auto i = 0; i < this->targets.size(); ++i) {
    auto entry = dimensions.find(this->targets[i]);
    auto expected_dim = it->second.expected_dimensions[i];
    if (expected_dim <= 0) {
      if (entry == dimensions.end())
        throw std::runtime_error("missing dimension for degree " + std::to_string(this->targets[i]));
      relevant_dimensions.push_back(entry->second);
    } else {
      if (entry == dimensions.end())
        dimensions[this->targets[i]] = expected_dim;
      else if (entry->second != expected_dim)
        throw std::runtime_error("invalid dimension for degree " + 
                                  std::to_string(this->targets[i]) + 
                                  ", expected dimension is " + std::to_string(expected_dim));
      relevant_dimensions.push_back(expected_dim);
    }
  }

  return it->second.generate_matrix(relevant_dimensions, parameters);
}

std::string matrix_operator::to_string(bool include_degrees) const {
  if (!include_degrees) return this->id;
  else if (this->targets.size() == 0) return this->id + "()";
  auto it = this->targets.begin();
  std::string str = this->id + "(" + std::to_string(*it++);
  while (it != this->targets.end())
    str += ", " + std::to_string(*it++);
  return str + ")";
}

// comparisons

bool matrix_operator::operator==(const matrix_operator &other) const {
  return this->id == other.id && this->targets == other.targets;
}

// predefined operators

// multiplicative identity
matrix_operator matrix_operator::one(int degree) {
  std::string op_id = "identity";
  if (matrix_operator::m_ops.find(op_id) == matrix_operator::m_ops.end()) {
    auto func = [](std::vector<int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      std::size_t dimension = dimensions[0];
      auto mat = matrix_2(dimension, dimension);

      // Build up the identity matrix.
      for (std::size_t i = 0; i < dimension; i++) {
        mat[{i, i}] = 1.0 + 0.0j;
      }
      return mat;
    };
    matrix_operator::define(op_id, {-1}, std::move(func));
  }
  return matrix_operator(op_id, {degree});
}

product_operator<matrix_operator> matrix_operator::identity(int degree) {
  return product_operator(std::move(matrix_operator::one(degree)));
}

product_operator<matrix_operator> matrix_operator::annihilate(int degree) {
  std::string op_id = "annihilate";
  if (matrix_operator::m_ops.find(op_id) == matrix_operator::m_ops.end()) {
    auto func = [](std::vector<int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      std::size_t dimension = dimensions[0];
      auto mat = matrix_2(dimension, dimension);
      for (std::size_t i = 0; i + 1 < dimension; i++) {
        mat[{i, i + 1}] = std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
      }
      return mat;
    };
    matrix_operator::define(op_id, {-1}, func);
  }
  auto op = matrix_operator(op_id, {degree});
  return product_operator(std::move(op));
}

product_operator<matrix_operator> matrix_operator::create(int degree) {
  std::string op_id = "create";
  if (matrix_operator::m_ops.find(op_id) == matrix_operator::m_ops.end()) {
    auto func = [](std::vector<int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      std::size_t dimension = dimensions[0];
      auto mat = matrix_2(dimension, dimension);
      for (std::size_t i = 0; i + 1 < dimension; i++) {
        mat[{i + 1, i}] = std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
      }
      return mat;
    };
    matrix_operator::define(op_id, {-1}, func);
  }
  auto op = matrix_operator(op_id, {degree});
  return product_operator(std::move(op));
}

product_operator<matrix_operator> matrix_operator::position(int degree) {
  std::string op_id = "position";
  if (matrix_operator::m_ops.find(op_id) == matrix_operator::m_ops.end()) {
    auto func = [](std::vector<int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      std::size_t dimension = dimensions[0];
      auto mat = matrix_2(dimension, dimension);
      // position = 0.5 * (create + annihilate)
      for (std::size_t i = 0; i + 1 < dimension; i++) {
        mat[{i + 1, i}] =
            0.5 * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
        mat[{i, i + 1}] =
            0.5 * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
      }
      return mat;
    };
    matrix_operator::define(op_id, {-1}, func);
  }
  auto op = matrix_operator(op_id, {degree});
  return product_operator(std::move(op));
}

product_operator<matrix_operator> matrix_operator::momentum(int degree) {
  std::string op_id = "momentum";
  if (matrix_operator::m_ops.find(op_id) == matrix_operator::m_ops.end()) {
    auto func = [](std::vector<int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      std::size_t dimension = dimensions[0];
      auto mat = matrix_2(dimension, dimension);
      // momentum = 0.5j * (create - annihilate)
      for (std::size_t i = 0; i + 1 < dimension; i++) {
        mat[{i + 1, i}] =
            (0.5j) * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
        mat[{i, i + 1}] =
            -1. * (0.5j) * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
      }
      return mat;
    };
    matrix_operator::define(op_id, {-1}, func);
  }
  auto op = matrix_operator(op_id, {degree});
  return product_operator(std::move(op));
}

product_operator<matrix_operator> matrix_operator::number(int degree) {
  std::string op_id = "number";
  if (matrix_operator::m_ops.find(op_id) == matrix_operator::m_ops.end()) {
    auto func = [](std::vector<int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      std::size_t dimension = dimensions[0];
      auto mat = matrix_2(dimension, dimension);
      for (std::size_t i = 0; i < dimension; i++) {
        mat[{i, i}] = static_cast<double>(i) + 0.0j;
      }
      return mat;
    };
    matrix_operator::define(op_id, {-1}, func);
  }
  auto op = matrix_operator(op_id, {degree});
  return product_operator(std::move(op));
}

product_operator<matrix_operator> matrix_operator::parity(int degree) {
  std::string op_id = "parity";
  if (matrix_operator::m_ops.find(op_id) == matrix_operator::m_ops.end()) {
    auto func = [](std::vector<int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      std::size_t dimension = dimensions[0];
      auto mat = matrix_2(dimension, dimension);
      for (std::size_t i = 0; i < dimension; i++) {
        mat[{i, i}] = std::pow(-1., static_cast<double>(i)) + 0.0j;
      }
      return mat;
    };
    matrix_operator::define(op_id, {-1}, func);
  }
  auto op = matrix_operator(op_id, {degree});
  return product_operator(std::move(op));
}

product_operator<matrix_operator> matrix_operator::displace(int degree) {
  std::string op_id = "displace";
  if (matrix_operator::m_ops.find(op_id) == matrix_operator::m_ops.end()) {
    auto func = [](std::vector<int> dimensions,
                     std::map<std::string, std::complex<double>> parameters) {
      std::size_t dimension = dimensions[0];
      auto entry = parameters.find("displacement");
      if (entry == parameters.end())
          throw std::runtime_error("missing value for parameter 'displacement'");
      auto displacement_amplitude = entry->second;
      auto create = matrix_2(dimension, dimension);
      auto annihilate = matrix_2(dimension, dimension);
      for (std::size_t i = 0; i + 1 < dimension; i++) {
        create[{i + 1, i}] = std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
        annihilate[{i, i + 1}] =
            std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
      }
      auto term1 = displacement_amplitude * create;
      auto term2 = std::conj(displacement_amplitude) * annihilate;
      return (term1 - term2).exponential();
    };
    matrix_operator::define(op_id, {-1}, func);
  }
  auto op = matrix_operator(op_id, {degree});
  return product_operator(std::move(op));
}

product_operator<matrix_operator> matrix_operator::squeeze(int degree) {
  std::string op_id = "squeeze";
  if (matrix_operator::m_ops.find(op_id) == matrix_operator::m_ops.end()) {
    auto func = [](std::vector<int> dimensions,
                     std::map<std::string, std::complex<double>> parameters) {
      std::size_t dimension = dimensions[0];
      auto entry = parameters.find("squeezing");
      if (entry == parameters.end())
          throw std::runtime_error("missing value for parameter 'squeezing'");
      auto squeezing = entry->second;
      auto create = matrix_2(dimension, dimension);
      auto annihilate = matrix_2(dimension, dimension);
      for (std::size_t i = 0; i + 1 < dimension; i++) {
        create[{i + 1, i}] = std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
        annihilate[{i, i + 1}] =
            std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
      }
      auto term1 = std::conj(squeezing) * annihilate.power(2);
      auto term2 = squeezing * create.power(2);
      auto difference = 0.5 * (term1 - term2);
      return difference.exponential();
    };
    matrix_operator::define(op_id, {-1}, func);
  }
  auto op = matrix_operator(op_id, {degree});
  return product_operator(std::move(op));
}

// tools for custom operators

} // namespace cudaq