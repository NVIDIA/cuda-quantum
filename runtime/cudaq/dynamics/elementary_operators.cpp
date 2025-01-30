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

std::map<std::string, Definition> elementary_operator::m_ops = {};

product_operator<elementary_operator> elementary_operator::identity(int degree) {
  std::string op_id = "identity";
  auto op = elementary_operator(op_id, {degree});
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&, degree](std::map<int, int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      std::size_t dimension = dimensions[degree];
      auto mat = matrix_2(dimension, dimension);

      // Build up the identity matrix.
      for (std::size_t i = 0; i < dimension; i++) {
        mat[{i, i}] = 1.0 + 0.0j;
      }
      return mat;
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  return product_operator<elementary_operator>(1., op);
}

product_operator<elementary_operator> elementary_operator::zero(int degree) {
  std::string op_id = "zero";
  auto op = elementary_operator(op_id, {degree});
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&, degree](std::map<int, int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      // Need to set the degree via the op itself because the
      // argument to the outer function goes out of scope when
      // the user invokes this later on via, e.g, `to_matrix()`.
      std::size_t dimension = dimensions[degree];
      auto mat = matrix_2(dimension, dimension);
      return mat;
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  return product_operator<elementary_operator>(1., op);
}

product_operator<elementary_operator> elementary_operator::annihilate(int degree) {
  std::string op_id = "annihilate";
  auto op = elementary_operator(op_id, {degree});
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&, degree](std::map<int, int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      std::size_t dimension = dimensions[degree];
      auto mat = matrix_2(dimension, dimension);
      for (std::size_t i = 0; i + 1 < dimension; i++) {
        mat[{i, i + 1}] = std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
      }
      return mat;
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  return product_operator<elementary_operator>(1., op);
}

product_operator<elementary_operator> elementary_operator::create(int degree) {
  std::string op_id = "create";
  auto op = elementary_operator(op_id, {degree});
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&, degree](std::map<int, int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      std::size_t dimension = dimensions[degree];
      auto mat = matrix_2(dimension, dimension);
      for (std::size_t i = 0; i + 1 < dimension; i++) {
        mat[{i + 1, i}] = std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
      }
      return mat;
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  return product_operator<elementary_operator>(1., op);
}

product_operator<elementary_operator> elementary_operator::position(int degree) {
  std::string op_id = "position";
  auto op = elementary_operator(op_id, {degree});
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&, degree](std::map<int, int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      std::size_t dimension = dimensions[degree];
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
    op.define(op_id, op.expected_dimensions, func);
  }
  return product_operator<elementary_operator>(1., op);
}

product_operator<elementary_operator> elementary_operator::momentum(int degree) {
  std::string op_id = "momentum";
  auto op = elementary_operator(op_id, {degree});
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&, degree](std::map<int, int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      std::size_t dimension = dimensions[degree];
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
    op.define(op_id, op.expected_dimensions, func);
  }
  return product_operator<elementary_operator>(1., op);
}

product_operator<elementary_operator> elementary_operator::number(int degree) {
  std::string op_id = "number";
  auto op = elementary_operator(op_id, {degree});
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&, degree](std::map<int, int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      std::size_t dimension = dimensions[degree];
      auto mat = matrix_2(dimension, dimension);
      for (std::size_t i = 0; i < dimension; i++) {
        mat[{i, i}] = static_cast<double>(i) + 0.0j;
      }
      return mat;
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  return product_operator<elementary_operator>(1., op);
}

product_operator<elementary_operator> elementary_operator::parity(int degree) {
  std::string op_id = "parity";
  auto op = elementary_operator(op_id, {degree});
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&, degree](std::map<int, int> dimensions,
                    std::map<std::string, std::complex<double>> _none) {
      std::size_t dimension = dimensions[degree];
      auto mat = matrix_2(dimension, dimension);
      for (std::size_t i = 0; i < dimension; i++) {
        mat[{i, i}] = std::pow(-1., static_cast<double>(i)) + 0.0j;
      }
      return mat;
    };
    op.define(op_id, op.expected_dimensions, func);
  }
  return product_operator<elementary_operator>(1., op);
}

product_operator<elementary_operator> elementary_operator::displace(int degree) {
  std::string op_id = "displace";
  auto op = elementary_operator(op_id, {degree});
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&, degree](std::map<int, int> dimensions,
                     std::map<std::string, std::complex<double>> parameters) {
      std::size_t dimension = dimensions[degree];
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
    op.define(op_id, op.expected_dimensions, func);
  }
  return product_operator<elementary_operator>(1., op);
}


product_operator<elementary_operator> elementary_operator::squeeze(int degree) {
  std::string op_id = "squeeze";
  auto op = elementary_operator(op_id, {degree});
  // A dimension of -1 indicates this operator can act on any dimension.
  op.expected_dimensions[degree] = -1;
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&, degree](std::map<int, int> dimensions,
                     std::map<std::string, std::complex<double>> parameters) {
      std::size_t dimension = dimensions[degree];
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
    op.define(op_id, op.expected_dimensions, func);
  }
  return product_operator<elementary_operator>(1., op);
}


matrix_2 elementary_operator::to_matrix(
    std::map<int, int> dimensions,
    std::map<std::string, std::complex<double>> parameters) const {
  return m_ops[id].generator(dimensions, parameters);
}

} // namespace cudaq