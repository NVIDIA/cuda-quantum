/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/operators.h"

#include <iostream>
#include <complex>
#include <set>

namespace cudaq {

std::map<std::string, Definition> matrix_operator::m_ops = {};

product_operator<matrix_operator> matrix_operator::identity(int degree) {
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
  auto op = matrix_operator(op_id, {degree});
  return product_operator<matrix_operator>(1., op);
}

product_operator<matrix_operator> matrix_operator::zero(int degree) {
  std::string op_id = "zero";
  if (matrix_operator::m_ops.find(op_id) == matrix_operator::m_ops.end()) {
    auto func = [](std::vector<int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      // Need to set the degree via the op itself because the
      // argument to the outer function goes out of scope when
      // the user invokes this later on via, e.g, `to_matrix()`.
      std::size_t dimension = dimensions[0];
      auto mat = matrix_2(dimension, dimension);
      return mat;
    };
    matrix_operator::define(op_id, {-1}, func);
  }
  auto op = matrix_operator(op_id, {degree});
  return product_operator<matrix_operator>(1., op);
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
  return product_operator<matrix_operator>(1., op);
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
  return product_operator<matrix_operator>(1., op);
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
  return product_operator<matrix_operator>(1., op);
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
  return product_operator<matrix_operator>(1., op);
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
  return product_operator<matrix_operator>(1., op);
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
  return product_operator<matrix_operator>(1., op);
}

product_operator<matrix_operator> matrix_operator::displace(int degree) {
  std::string op_id = "displace";
  if (matrix_operator::m_ops.find(op_id) == matrix_operator::m_ops.end()) {
    auto func = [](std::vector<int> dimensions,
                     std::map<std::string, std::complex<double>> parameters) {
      std::size_t dimension = dimensions[0];
      auto displacement_amplitude = parameters["displacement"];
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
  return product_operator<matrix_operator>(1., op);
}


product_operator<matrix_operator> matrix_operator::squeeze(int degree) {
  std::string op_id = "squeeze";
  if (matrix_operator::m_ops.find(op_id) == matrix_operator::m_ops.end()) {
    auto func = [](std::vector<int> dimensions,
                     std::map<std::string, std::complex<double>> parameters) {
      std::size_t dimension = dimensions[0];
      auto squeezing = parameters["squeezing"];
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
  return product_operator<matrix_operator>(1., op);
}


matrix_2 matrix_operator::to_matrix(
    std::map<int, int> dimensions,
    std::map<std::string, std::complex<double>> parameters) const {
  auto it = matrix_operator::m_ops.find(this->id);
  if (it != matrix_operator::m_ops.end()) {
      std::vector<int> relevant_dimensions;
      relevant_dimensions.reserve(this->degrees.size());
      for (auto d : this->degrees)
        relevant_dimensions.push_back(dimensions[d]);
      return it->second.generate_matrix(relevant_dimensions, parameters);
  }
  throw std::range_error("unable to find operator");
}

} // namespace cudaq