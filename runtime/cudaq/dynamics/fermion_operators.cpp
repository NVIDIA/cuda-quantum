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

#include "cudaq/utils/tensor.h"
#include "cudaq/operators.h"
#include "fermion_operators.h"

namespace cudaq {

#if !defined(NDEBUG)
bool fermion_operator::can_be_canonicalized = false;
#endif

// private helpers

#if !defined(NDEBUG)
void fermion_operator::validate_opcode() const {
  std::vector<int> valid_op_codes = {0, 1, 2, 4, 8, 9};
  assert(std::find(valid_op_codes.cbegin(), valid_op_codes.cend(), std::abs(this->op_code)) != valid_op_codes.cend());
  // While -8 should be fine for all of the implemented logic, I believe it can never
  // occur within this representation. A value of -9 on the other hand should also never
  // occur, and if it does then something is wrong with the code and/or my reasoning.
  assert(this->op_code != -9 && this->op_code != -8);
}
#endif

std::string fermion_operator::op_code_to_string() const {
  // Note that we can (and should) have the same op codes across boson, fermion, and spin ops, 
  // since individual operators with the same op codes are actually equal.
  // Note that the matrix definition for creation, annihilation and number operators are 
  // equal despite the different commutation/anticommutation relations; what makes them 
  // behave differently is effectively the "finite size effects" for fermions. Specifically,
  // if we were to blindly multiply the matrices for d=2 for bosons, we would get the same
  // behavior as we have for a single fermion due to the finite size of the matrix. 
  // To avoid this, we ensure that we reorder the operators for bosons appropriately as part of
  // the in-place multiplication, whereas for fermions, this effect is desired/correct. 
  if (this->op_code == 0) return "0";
  if (this->op_code & 1) {
    if (this->op_code < 0) return "(N-1)";
    else return "(1-N)";
  } 
  if (this->op_code & 2) {
    if (this->op_code < 0) return "AZ";
    else return "A";
  }
  if (this->op_code & 4) {
    if (this->op_code < 0) return "ZAd";
    else return "Ad";
  }
  if (this->op_code & 8) {
    assert(this->op_code > 0); // should never be negative
    return "N";
  }
  return "I";
}

std::string fermion_operator::op_code_to_string(std::unordered_map<int, int> &dimensions) const {
  auto it = dimensions.find(this->target);
  if (it == dimensions.end())
    dimensions[this->target] = 2;
  else if (it->second != 2)
    throw std::runtime_error("dimension for fermion operator must be 2");
  return this->op_code_to_string();
}

void fermion_operator::inplace_mult(const fermion_operator &other) {
#if !defined(NDEBUG)
  other.validate_opcode();
#endif

  // The below code is just a bitwise implementation of a matrix multiplication;
  // Multiplication becomes a bitwise and, addition becomes an exclusive or.
  auto get_entry = [](const fermion_operator &op, int quadrant) {
    return (op.op_code & (1 << quadrant)) >> quadrant;
  };

  auto res00 = (get_entry(*this, 0) & get_entry(other, 0)) ^ (get_entry(*this, 1) & get_entry(other, 2));
  auto res01 = (get_entry(*this, 0) & get_entry(other, 1)) ^ (get_entry(*this, 1) & get_entry(other, 3));
  auto res10 = (get_entry(*this, 2) & get_entry(other, 0)) ^ (get_entry(*this, 3) & get_entry(other, 2));
  auto res11 = (get_entry(*this, 2) & get_entry(other, 1)) ^ (get_entry(*this, 3) & get_entry(other, 3));

  this->op_code = res00 ^ (res01 << 1) ^ (res10 << 2) ^ (res11 << 3);
  if ((this->op_code < 0) ^ (other.op_code < 0)) this->op_code = -this->op_code;
#if !defined(NDEBUG)
  this->validate_opcode();
#endif
}

// read-only properties

std::string fermion_operator::unique_id() const {
  return this->op_code_to_string() + std::to_string(target);
}

std::vector<int> fermion_operator::degrees() const {
  return {this->target};
}

// constructors

fermion_operator::fermion_operator(int target) 
  : target(target), op_code(9) {}

fermion_operator::fermion_operator(int target, int op_id) 
  : target(target), op_code(9) {
    assert(0 <= op_id < 4);
    if (op_id == 1) // create
      this->op_code = 4;
    else if (op_id == 2) // annihilate
      this->op_code = 2;
    else if (op_id == 3) // number
      this->op_code = 8;
}

// evaluations

matrix_2 fermion_operator::to_matrix(std::unordered_map<int, int> &dimensions,
                                   const std::unordered_map<std::string, std::complex<double>> &parameters) const {
  auto it = dimensions.find(this->target);
  if (it == dimensions.end())
    dimensions[this->target] = 2;
  else if (it->second != 2)
    throw std::runtime_error("dimension for fermion operator must be 2");

#if !defined(NDEBUG)
  this->validate_opcode();
#endif

  auto mat = matrix_2(2, 2);
  auto value = this->op_code < 0 ? -1. : 1.;
  if (this->op_code & 1) mat[{0, 0}] = value;
  if (this->op_code & 2) mat[{0, 1}] = value;
  if (this->op_code & 4) mat[{1, 0}] = value;
  if (this->op_code & 8) mat[{1, 1}] = value;
  return std::move(mat);
}

std::string fermion_operator::to_string(bool include_degrees) const {
  if (include_degrees) return this->op_code_to_string() + "(" + std::to_string(target) + ")";
  else return this->op_code_to_string();
}

// comparisons

bool fermion_operator::operator==(const fermion_operator &other) const {
  return this->target == other.target &&
         this->op_code == other.op_code;
}

// defined operators

operator_sum<fermion_operator> fermion_operator::empty() {
  return operator_handler::empty<fermion_operator>();
}

product_operator<fermion_operator> fermion_operator::identity() {
  return operator_handler::identity<fermion_operator>();
}

product_operator<fermion_operator> fermion_operator::identity(int degree) {
  return product_operator(fermion_operator(degree));
}

product_operator<fermion_operator> fermion_operator::create(int degree) {
  return product_operator(fermion_operator(degree, 1));
}

product_operator<fermion_operator> fermion_operator::annihilate(int degree) {
  return product_operator(fermion_operator(degree, 2));
}

product_operator<fermion_operator> fermion_operator::number(int degree) {
  return product_operator(fermion_operator(degree, 3));
}

} // namespace cudaq