/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <algorithm>
#include <cmath>
#include <complex>
#include <unordered_map>
#include <vector>

#include "common/EigenSparse.h"
#include "cudaq/operators.h"
#include "cudaq/utils/matrix.h"

#include "cudaq/fermion_op.h"

namespace cudaq {

// private helpers

#if !defined(NDEBUG)
void fermion_handler::validate_opcode() const {
  std::vector<int> valid_op_codes = {0, 1, 2, 4, 8, 9};
  assert(std::find(valid_op_codes.cbegin(), valid_op_codes.cend(),
                   this->op_code) != valid_op_codes.cend());
  assert(this->commutes_across_degrees ==
         (!(this->op_code & 2) && !(this->op_code & 4)));
}
#endif

// used internally and externally to identity the operator -
// use a human friendly string code to make it more comprehensible
std::string fermion_handler::op_code_to_string() const {
  // Note that we can (and should) have the same op codes across boson, fermion,
  // and spin ops, since individual operators with the same op codes are
  // actually equal. Note that the matrix definition for creation, annihilation
  // and number operators are equal despite the different
  // commutation/anticommutation relations; what makes them behave differently
  // is effectively the "finite size effects" for fermions. Specifically, if we
  // were to blindly multiply the matrices for d=2 for bosons, we would get the
  // same behavior as we have for a single fermion due to the finite size of the
  // matrix. To avoid this, we ensure that we reorder the operators for bosons
  // appropriately as part of the in-place multiplication, whereas for fermions,
  // this effect is desired/correct.
  if (this->op_code == 0)
    return "0";
  if (this->op_code & 1)
    return "(1-N)";
  if (this->op_code & 2)
    return "A";
  if (this->op_code & 4)
    return "Ad";
  if (this->op_code & 8)
    return "N";
  return "I";
}

// Used internally for canonical evaluation -
// use a single char for representing the operator.
// Relevant dimensions is not used but only exists
// for consistency with other operator classes.
std::string fermion_handler::canonical_form(
    std::unordered_map<std::size_t, std::int64_t> &dimensions,
    std::vector<std::int64_t> &relevant_dims) const {
  auto it = dimensions.find(this->degree);
  if (it == dimensions.end())
    dimensions[this->degree] = 2;
  else if (it->second != 2)
    throw std::runtime_error("dimension for fermion operator must be 2");
  return std::to_string(this->op_code);
}

void fermion_handler::inplace_mult(const fermion_handler &other) {
#if !defined(NDEBUG)
  other.validate_opcode();
#endif

  // The below code is just a bitwise implementation of a matrix multiplication;
  // Multiplication becomes a bitwise and, addition becomes an exclusive or.
  auto get_entry = [](const fermion_handler &op, int quadrant) {
    return (op.op_code & (1 << quadrant)) >> quadrant;
  };

  auto res00 = (get_entry(*this, 0) & get_entry(other, 0)) ^
               (get_entry(*this, 1) & get_entry(other, 2));
  auto res01 = (get_entry(*this, 0) & get_entry(other, 1)) ^
               (get_entry(*this, 1) & get_entry(other, 3));
  auto res10 = (get_entry(*this, 2) & get_entry(other, 0)) ^
               (get_entry(*this, 3) & get_entry(other, 2));
  auto res11 = (get_entry(*this, 2) & get_entry(other, 1)) ^
               (get_entry(*this, 3) & get_entry(other, 3));

  this->op_code = res00 ^ (res01 << 1) ^ (res10 << 2) ^ (res11 << 3);
  this->commutes = !(this->op_code & 2) && !(this->op_code & 4);
#if !defined(NDEBUG)
  this->validate_opcode();
#endif
}

// read-only properties

std::string fermion_handler::unique_id() const {
  return this->op_code_to_string() + std::to_string(this->degree);
}

std::vector<std::size_t> fermion_handler::degrees() const {
  return {this->degree};
}

std::size_t fermion_handler::target() const { return this->degree; }

// constructors

fermion_handler::fermion_handler(std::size_t target)
    : degree(target), op_code(9), commutes(true) {}

fermion_handler::fermion_handler(std::size_t target, int op_id)
    : degree(target), op_code(9), commutes(true) {
  assert(0 <= op_id && op_id < 4);
  if (op_id == 1) { // create
    this->op_code = 4;
    this->commutes = false;
  } else if (op_id == 2) { // annihilate
    this->op_code = 2;
    this->commutes = false;
  } else if (op_id == 3) // number
    this->op_code = 8;
}

fermion_handler::fermion_handler(const fermion_handler &other)
    : op_code(other.op_code), commutes(other.commutes), degree(other.degree) {}

// assignments

fermion_handler &fermion_handler::operator=(const fermion_handler &other) {
  if (this != &other) {
    this->op_code = other.op_code;
    this->commutes = other.commutes;
    this->degree = other.degree;
  }
  return *this;
}

// evaluations

void fermion_handler::create_matrix(
    const std::string &fermi_word,
    const std::function<void(std::size_t, std::size_t, std::complex<double>)>
        &process_element,
    bool invert_order) {

  // check if the operator quenches all states
  auto it = std::find(fermi_word.cbegin(), fermi_word.cend(), '0');
  if (it != fermi_word.cend())
    return;

  auto map_state = [](char encoding, bool state) {
    if (state) {
      if (encoding == '4' || encoding == '1') // zeros the state
        return std::pair<double, bool>{0., state};
      if (encoding == '8')
        return std::pair<double, bool>{1., state};
      if (encoding == '2')
        return std::pair<double, bool>{1., !state};
      else {
        assert(encoding == '9');
        return std::pair<double, bool>{1., state};
      }
    } else {
      if (encoding == '2' || encoding == '8') // zeros the state
        return std::pair<double, bool>{0., state};
      if (encoding == '1')
        return std::pair<double, bool>{1., state};
      if (encoding == '4')
        return std::pair<double, bool>{1., !state};
      else {
        assert(encoding == '9');
        return std::pair<double, bool>{1., state};
      }
    }
  };

  auto dim = 1 << fermi_word.size();
  auto nr_deg = fermi_word.size();

  for (std::size_t old_state = 0; old_state < dim; ++old_state) {
    std::size_t new_state = 0;
    std::complex<double> entry = 1.;
    for (std::size_t degree = 0; degree < nr_deg; ++degree) {
      auto state = (old_state & (1 << degree)) >> degree;
      auto op = fermi_word[invert_order ? nr_deg - 1 - degree : degree];
      auto mapped = map_state(op, state);
      entry *= mapped.first;
      new_state |= (mapped.second << degree);
    }
    process_element(new_state, old_state, entry);
  }
}

cudaq::detail::EigenSparseMatrix fermion_handler::to_sparse_matrix(
    const std::string &fermi_word, const std::vector<std::int64_t> &dimensions,
    std::complex<double> coeff, bool invert_order) {
  // private method, so we only assert dimensions
  assert(std::find_if(dimensions.cbegin(), dimensions.cend(),
                      [](std::int64_t d) { return d != 2; }) ==
         dimensions.cend());
  auto dim = 1 << fermi_word.size();
  return cudaq::detail::create_sparse_matrix(
      dim, coeff,
      [&fermi_word, invert_order](
          const std::function<void(std::size_t, std::size_t,
                                   std::complex<double>)> &process_entry) {
        create_matrix(fermi_word, process_entry, invert_order);
      });
}

complex_matrix
fermion_handler::to_matrix(const std::string &fermi_word,
                           const std::vector<std::int64_t> &dimensions,
                           std::complex<double> coeff, bool invert_order) {
  // private method, so we only assert dimensions
  assert(std::find_if(dimensions.cbegin(), dimensions.cend(),
                      [](std::int64_t d) { return d != 2; }) ==
         dimensions.cend());
  auto dim = 1 << fermi_word.size();
  return cudaq::detail::create_matrix(
      dim, coeff,
      [&fermi_word, invert_order](
          const std::function<void(std::size_t, std::size_t,
                                   std::complex<double>)> &process_entry) {
        create_matrix(fermi_word, process_entry, invert_order);
      });
}

complex_matrix fermion_handler::to_matrix(
    std::unordered_map<std::size_t, std::int64_t> &dimensions,
    const std::unordered_map<std::string, std::complex<double>> &parameters)
    const {
  std::vector<std::int64_t> relevant_dims;
  return fermion_handler::to_matrix(
      this->canonical_form(dimensions, relevant_dims));
}

mdiag_sparse_matrix fermion_handler::to_diagonal_matrix(
    std::unordered_map<std::size_t, std::int64_t> &dimensions,
    const std::unordered_map<std::string, std::complex<double>> &parameters)
    const {
  std::vector<std::int64_t> relevant_dims;
  return fermion_handler::to_diagonal_matrix(
      this->canonical_form(dimensions, relevant_dims));
}

mdiag_sparse_matrix fermion_handler::to_diagonal_matrix(
    const std::string &fermi_word, const std::vector<std::int64_t> &dimensions,
    std::complex<double> coeff, bool invert_order) {
  auto dim = 1 << fermi_word.size();
  return cudaq::detail::create_mdiag_sparse_matrix(
      dim, coeff,
      [&fermi_word, invert_order](
          const std::function<void(std::size_t, std::size_t,
                                   std::complex<double>)> &process_entry) {
        create_matrix(fermi_word, process_entry, invert_order);
      });
}

std::string fermion_handler::to_string(bool include_degrees) const {
  if (include_degrees)
    return this->unique_id(); // unique id for consistency with keys in some
                              // user facing maps
  else
    return this->op_code_to_string();
}

// comparisons

bool fermion_handler::operator==(const fermion_handler &other) const {
  return this->degree == other.degree &&
         this->op_code == other.op_code; // no need to compare commutes (is
                                         // determined by op_code)
}

// defined operators

fermion_handler fermion_handler::create(std::size_t degree) {
  return fermion_handler(degree, 1);
}

fermion_handler fermion_handler::annihilate(std::size_t degree) {
  return fermion_handler(degree, 2);
}

fermion_handler fermion_handler::number(std::size_t degree) {
  return fermion_handler(degree, 3);
}

namespace fermion {
product_op<fermion_handler> create(std::size_t target) {
  return product_op(fermion_handler::create(target));
}
product_op<fermion_handler> annihilate(std::size_t target) {
  return product_op(fermion_handler::annihilate(target));
}
product_op<fermion_handler> number(std::size_t target) {
  return product_op(fermion_handler::number(target));
}
} // namespace fermion

} // namespace cudaq
