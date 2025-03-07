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

#include "cudaq/utils/matrix.h"
#include "spin_operators.h"

namespace cudaq {

// private helpers

std::string spin_operator::op_code_to_string() const {
  if (this->op_code == 1)
    return "Z";
  if (this->op_code == 2)
    return "X";
  if (this->op_code == 3)
    return "Y";
  return "I";
}

std::string spin_operator::op_code_to_string(
    std::unordered_map<int, int> &dimensions) const {
  auto it = dimensions.find(this->degree);
  if (it == dimensions.end())
    dimensions[this->degree] = 2;
  else if (it->second != 2)
    throw std::runtime_error("dimension for spin operator must be 2");
  return this->op_code_to_string();
}

std::complex<double> spin_operator::inplace_mult(const spin_operator &other) {
  assert(this->degree == other.degree);
  std::complex<double> factor;
  if (this->op_code == 0 || other.op_code == 0 ||
      this->op_code == other.op_code)
    factor = std::complex<double>(1., 0.);
  else if (this->op_code + 1 == other.op_code ||
           this->op_code - 2 == other.op_code)
    factor = std::complex<double>(0., 1.);
  else
    factor = std::complex<double>(0., -1.);
  this->op_code ^= other.op_code;
  return factor;
}

// read-only properties

pauli spin_operator::as_pauli() const {
  if (this->op_code == 1) return pauli::Z;
  if (this->op_code == 2) return pauli::X;
  if (this->op_code == 3) return pauli::Y;
  assert(this->op_code == 0);
  return pauli::I;
}

std::string spin_operator::unique_id() const {
  return this->op_code_to_string() + std::to_string(this->degree);
}

std::vector<int> spin_operator::degrees() const { return {this->degree}; }

int spin_operator::target() const { return this->degree; }

// constructors

spin_operator::spin_operator(int target) : op_code(0), degree(target) {}

spin_operator::spin_operator(int target, int op_id)
    : op_code(op_id), degree(target) {
  assert(0 <= op_id && op_id < 4);
}

// evaluations

complex_matrix spin_operator::to_matrix(std::string pauli_word,
                                  std::complex<double> coeff,
                                  bool invert_order) {
  auto map_state = [&pauli_word](char pauli, bool state) {
    if (state) {
      if (pauli == 'Z')
        return std::make_pair(std::complex<double>(-1., 0.), bool(state));
      if (pauli == 'X')
        return std::make_pair(std::complex<double>(1., 0.), !state);
      if (pauli == 'Y')
        return std::make_pair(std::complex<double>(0., -1.), !state);
      return std::make_pair(std::complex<double>(1., 0.), bool(state));
    } else {
      if (pauli == 'Z')
        return std::make_pair(std::complex<double>(1., 0.), bool(state));
      if (pauli == 'X')
        return std::make_pair(std::complex<double>(1., 0.), !state);
      if (pauli == 'Y')
        return std::make_pair(std::complex<double>(0., 1.), !state);
      return std::make_pair(std::complex<double>(1., 0.), bool(state));
    }
  };

  auto dim = 1 << pauli_word.size();
  auto nr_deg = pauli_word.size();

  complex_matrix matrix(dim, dim);
  for (std::size_t old_state = 0; old_state < dim; ++old_state) {
    std::size_t new_state = 0;
    std::complex<double> entry = 1.;
    for (auto degree = 0; degree < nr_deg; ++degree) {
      auto canon_degree = degree;
      auto state = (old_state & (1 << canon_degree)) >> canon_degree;
      auto op = pauli_word[invert_order ? nr_deg - 1 - degree : degree];
      auto mapped = map_state(op, state);
      entry *= mapped.first;
      new_state |= (mapped.second << canon_degree);
    }
    matrix[{new_state, old_state}] = coeff * entry;
  }
  return std::move(matrix);
}

complex_matrix spin_operator::to_matrix(
    std::unordered_map<int, int> &dimensions,
    const std::unordered_map<std::string, std::complex<double>> &parameters)
    const {
  auto it = dimensions.find(this->degree);
  if (it == dimensions.end())
    dimensions[this->degree] = 2;
  else if (it->second != 2)
    throw std::runtime_error("dimension for spin operator must be 2");
  return spin_operator::to_matrix(this->op_code_to_string());
}

std::string spin_operator::to_string(bool include_degrees) const {
  if (include_degrees)
    return this->op_code_to_string() + "(" + std::to_string(this->degree) + ")";
  else
    return this->op_code_to_string();
}

// comparisons

bool spin_operator::operator==(const spin_operator &other) const {
  return this->op_code == other.op_code && this->degree == other.degree;
}

// defined operators

operator_sum<spin_operator> spin_operator::empty() {
  return operator_handler::empty<spin_operator>();
}

product_operator<spin_operator> spin_operator::identity() {
  return operator_handler::identity<spin_operator>();
}

product_operator<spin_operator> spin_operator::i(int degree) {
  return product_operator(spin_operator(degree));
}

product_operator<spin_operator> spin_operator::z(int degree) {
  return product_operator(spin_operator(degree, 1));
}

product_operator<spin_operator> spin_operator::x(int degree) {
  return product_operator(spin_operator(degree, 2));
}

product_operator<spin_operator> spin_operator::y(int degree) {
  return product_operator(spin_operator(degree, 3));
}

} // namespace cudaq