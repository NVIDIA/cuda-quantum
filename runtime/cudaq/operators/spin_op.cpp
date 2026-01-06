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

#include "common/EigenSparse.h"
#include "cudaq/operators.h"
#include "cudaq/utils/matrix.h"

#include "cudaq/spin_op.h"

namespace cudaq {

// private helpers

std::string spin_handler::op_code_to_string() const {
  if (this->op_code == 1)
    return "Z";
  if (this->op_code == 2)
    return "X";
  if (this->op_code == 3)
    return "Y";
  return "I";
}

// Used internally for canonical evaluation -
// use a single char for representing the operator.
// Relevant dimensions is not used but only exists
// for consistency with other operator classes.
std::string spin_handler::canonical_form(
    std::unordered_map<std::size_t, std::int64_t> &dimensions,
    std::vector<std::int64_t> &relevant_dims) const {
  auto it = dimensions.find(this->degree);
  if (it == dimensions.end())
    dimensions[this->degree] = 2;
  else if (it->second != 2)
    throw std::runtime_error("dimension for spin operator must be 2");
  return this->op_code_to_string();
}

std::complex<double> spin_handler::inplace_mult(const spin_handler &other) {
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

pauli spin_handler::as_pauli() const {
  if (this->op_code == 1)
    return pauli::Z;
  if (this->op_code == 2)
    return pauli::X;
  if (this->op_code == 3)
    return pauli::Y;
  assert(this->op_code == 0);
  return pauli::I;
}

std::string spin_handler::unique_id() const {
  return this->op_code_to_string() + std::to_string(this->degree);
}

std::vector<std::size_t> spin_handler::degrees() const {
  return {this->degree};
}

std::size_t spin_handler::target() const { return this->degree; }

// constructors

spin_handler::spin_handler(std::size_t target) : op_code(0), degree(target) {}

spin_handler::spin_handler(pauli p, std::size_t target)
    : op_code(0), degree(target) {
  if (p == pauli::Z)
    this->op_code = 1;
  else if (p == pauli::X)
    this->op_code = 2;
  else if (p == pauli::Y)
    this->op_code = 3;
}

spin_handler::spin_handler(std::size_t target, int op_id)
    : op_code(op_id), degree(target) {
  assert(0 <= op_id && op_id < 4);
}

// evaluations

void spin_handler::create_matrix(
    const std::string &pauli_word,
    const std::function<void(std::size_t, std::size_t, std::complex<double>)>
        &process_element,
    bool invert_order) {
  auto map_state = [](char pauli, bool state) {
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

  auto dim = 1ul << pauli_word.size();
  auto nr_deg = pauli_word.size();

  for (std::size_t old_state = 0; old_state < dim; ++old_state) {
    std::size_t new_state = 0;
    std::complex<double> entry = 1.;
    for (std::size_t degree = 0; degree < nr_deg; ++degree) {
      auto state = (old_state & (1ul << degree)) >> degree;
      auto op = pauli_word[invert_order ? nr_deg - 1 - degree : degree];
      auto mapped = map_state(op, state);
      entry *= mapped.first;
      new_state |= (mapped.second << degree);
    }
    process_element(new_state, old_state, entry);
  }
}

cudaq::detail::EigenSparseMatrix
spin_handler::to_sparse_matrix(const std::string &pauli_word,
                               std::complex<double> coeff, bool invert_order) {
  auto dim = 1ul << pauli_word.size();
  return cudaq::detail::create_sparse_matrix(
      dim, coeff,
      [&pauli_word, invert_order](
          const std::function<void(std::size_t, std::size_t,
                                   std::complex<double>)> &process_entry) {
        create_matrix(pauli_word, process_entry, invert_order);
      });
}

cudaq::detail::EigenSparseMatrix
spin_handler::to_sparse_matrix(const std::string &pauli,
                               const std::vector<std::int64_t> &dimensions,
                               std::complex<double> coeff, bool invert_order) {
  // private method, so we only assert dimensions
  assert(std::find_if(dimensions.cbegin(), dimensions.cend(),
                      [](std::int64_t d) { return d != 2; }) ==
         dimensions.cend());
  return to_sparse_matrix(pauli, coeff, invert_order);
}

complex_matrix spin_handler::to_matrix(const std::string &pauli_word,
                                       std::complex<double> coeff,
                                       bool invert_order) {
  auto dim = 1ul << pauli_word.size();
  return cudaq::detail::create_matrix(
      dim, coeff,
      [&pauli_word, invert_order](
          const std::function<void(std::size_t, std::size_t,
                                   std::complex<double>)> &process_entry) {
        create_matrix(pauli_word, process_entry, invert_order);
      });
}

complex_matrix
spin_handler::to_matrix(const std::string &pauli_word,
                        const std::vector<std::int64_t> &dimensions,
                        std::complex<double> coeff, bool invert_order) {
  // private method, so we only assert dimensions
  assert(std::find_if(dimensions.cbegin(), dimensions.cend(),
                      [](std::int64_t d) { return d != 2; }) ==
         dimensions.cend());
  return to_matrix(pauli_word, coeff, invert_order);
}

complex_matrix spin_handler::to_matrix(
    std::unordered_map<std::size_t, std::int64_t> &dimensions,
    const std::unordered_map<std::string, std::complex<double>> &parameters)
    const {
  std::vector<std::int64_t> rel_dims;
  return spin_handler::to_matrix(this->canonical_form(dimensions, rel_dims));
}

mdiag_sparse_matrix spin_handler::to_diagonal_matrix(
    std::unordered_map<std::size_t, std::int64_t> &dimensions,
    const std::unordered_map<std::string, std::complex<double>> &parameters)
    const {
  std::vector<std::int64_t> rel_dims;
  return spin_handler::to_diagonal_matrix(
      this->canonical_form(dimensions, rel_dims));
}

mdiag_sparse_matrix spin_handler::to_diagonal_matrix(
    const std::string &pauli_word, const std::vector<std::int64_t> &dimensions,
    std::complex<double> coeff, bool invert_order) {
  auto dim = 1ul << pauli_word.size();
  return cudaq::detail::create_mdiag_sparse_matrix(
      dim, coeff,
      [&pauli_word, invert_order](
          const std::function<void(std::size_t, std::size_t,
                                   std::complex<double>)> &process_entry) {
        create_matrix(pauli_word, process_entry, invert_order);
      });
}

std::string spin_handler::to_string(bool include_degrees) const {
  if (include_degrees)
    return this->unique_id(); // unique id for consistency with keys in some
                              // user facing maps
  else
    return this->op_code_to_string();
}

// comparisons

bool spin_handler::operator==(const spin_handler &other) const {
  return this->op_code == other.op_code && this->degree == other.degree;
}

// defined operators

spin_handler spin_handler::z(std::size_t degree) {
  return spin_handler(degree, 1);
}

spin_handler spin_handler::x(std::size_t degree) {
  return spin_handler(degree, 2);
}

spin_handler spin_handler::y(std::size_t degree) {
  return spin_handler(degree, 3);
}

namespace spin {
product_op<spin_handler> i(std::size_t target) {
  return product_op(spin_handler(target));
}
product_op<spin_handler> x(std::size_t target) {
  return product_op(spin_handler::x(target));
}
product_op<spin_handler> y(std::size_t target) {
  return product_op(spin_handler::y(target));
}
product_op<spin_handler> z(std::size_t target) {
  return product_op(spin_handler::z(target));
}
sum_op<spin_handler> plus(std::size_t target) {
  return sum_op<spin_handler>(0.5 * x(target),
                              std::complex<double>(0., 0.5) * y(target));
}
sum_op<spin_handler> minus(std::size_t target) {
  return sum_op<spin_handler>(0.5 * x(target),
                              std::complex<double>(0., -0.5) * y(target));
}
} // namespace spin

} // namespace cudaq
