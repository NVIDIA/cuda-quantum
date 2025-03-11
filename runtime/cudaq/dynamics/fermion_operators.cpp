/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
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

#include "cudaq/operators.h"
#include "cudaq/utils/matrix.h"
#include "fermion_operators.h"

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

std::string fermion_handler::op_code_to_string(
    std::unordered_map<int, int> &dimensions) const {
  auto it = dimensions.find(this->degree);
  if (it == dimensions.end())
    dimensions[this->degree] = 2;
  else if (it->second != 2)
    throw std::runtime_error("dimension for fermion operator must be 2");
  return this->op_code_to_string();
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

std::vector<int> fermion_handler::degrees() const { return {this->degree}; }

// constructors

fermion_handler::fermion_handler(int target)
    : degree(target), op_code(9), commutes(true) {}

fermion_handler::fermion_handler(int target, int op_id)
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

complex_matrix fermion_handler::to_matrix(
    std::unordered_map<int, int> &dimensions,
    const std::unordered_map<std::string, std::complex<double>> &parameters)
    const {
  auto it = dimensions.find(this->degree);
  if (it == dimensions.end())
    dimensions[this->degree] = 2;
  else if (it->second != 2)
    throw std::runtime_error("dimension for fermion operator must be 2");

#if !defined(NDEBUG)
  this->validate_opcode();
#endif

  auto mat = complex_matrix(2, 2);
  if (this->op_code & 1)
    mat[{0, 0}] = 1.;
  if (this->op_code & 2)
    mat[{0, 1}] = 1.;
  if (this->op_code & 4)
    mat[{1, 0}] = 1.;
  if (this->op_code & 8)
    mat[{1, 1}] = 1.;
  return std::move(mat);
}

std::string fermion_handler::to_string(bool include_degrees) const {
  if (include_degrees)
    return this->op_code_to_string() + "(" + std::to_string(this->degree) + ")";
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

product_op<fermion_handler> fermion_handler::create(int degree) {
  return product_op(fermion_handler(degree, 1));
}

product_op<fermion_handler> fermion_handler::annihilate(int degree) {
  return product_op(fermion_handler(degree, 2));
}

product_op<fermion_handler> fermion_handler::number(int degree) {
  return product_op(fermion_handler(degree, 3));
}

} // namespace cudaq