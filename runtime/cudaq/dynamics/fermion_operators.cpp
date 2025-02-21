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

// private helpers

std::string fermion_operator::op_code_to_string() const {
  // FIXME: WHY WOULD C=A Cd=Ad if commutation relations are different??
  // see matrix computation below;
  if (this->quenched) return "O";
  if (this->additional_term > 0) {
    assert(this->number_factor == 0);
    if (this->phase) return "ZCd";
    else return "Cd";
  }
  if (this->additional_term < 0) {
    assert(this->number_factor == 0);
    if (this->phase) return "CZ";
    else return "C";
  }
  if (this->number_factor > 0 ) {
    assert(!this->phase);
    return "Nf";
  }
  if (this->number_factor < 0) {
    if (this->phase) return "(Nf-1)";
    else return "(1-Nf)";
  }
  assert(!this->phase);
  return "I";
}

void fermion_operator::inplace_mult(const fermion_operator &other) {
  this->quenched = this->quenched || other.quenched;
  if (this->quenched) return;

  auto add_number_factor = [this](int8_t new_number_factor) {

  };

  auto new_number_factor = this->additional_term & 1 ? -other.number_factor : other.number_factor;
  add_number_factor(new_number_factor);

  if (this->additional_term > 0) {
    // ad a
    assert(!this->phase || !other.phase); // FIXME: REALLY?
    add_number_factor(1.);
  } else if (this->additional_term < 0) {
    // a ad
    //assert()
    this->phase = this->phase || other.phase;
    add_number_factor(-1.);
  } else {
    assert(!this->phase || !other.phase); // FIXME: REALLY?? -> COULD HAVE A PHASE FROM NR
    this->phase = this->phase || other.phase;
  }

  this->additional_term += other.additional_term;
  if (this->additional_term & 2) this->quenched = true;
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
  : target(target), additional_term(0), number_factor(0), phase(false), quenched(false) {}

fermion_operator::fermion_operator(int target, int op_id) 
  : target(target), additional_term(0), number_factor(0), phase(false), quenched(false) {
    assert(0 <= op_id < 4);
    if (op_id == 1) // create
      this->additional_term = 1;
    else if (op_id == 2) // annihilate
      this->additional_term = -1;
    else if (op_id == 3) // number
      this->number_factor = 1;
}

// evaluations

matrix_2 fermion_operator::to_matrix(std::unordered_map<int, int> &dimensions,
                                   const std::unordered_map<std::string, std::complex<double>> &parameters) const {
  auto it = dimensions.find(this->target);
  if (it == dimensions.end())
    dimensions[this->target] = 2;
  else if (it->second != 2)
    throw std::runtime_error("dimension for fermion operator must be 2");

  auto mat = matrix_2(2, 2);
  if (this->quenched) return std::move(mat);

  assert(std::abs(this->additional_term) <= 1);
  assert(std::abs(this->number_factor) <= 1);
  assert(!this->phase || this->number_factor <= 0);

  // Note that we could apply the phase globally, but we do it in-place instead.
  auto value = this->phase ? -1. : 1.;
  if (this->additional_term > 0) {
    // We can only have either a Cd or a Z*Cd term, since (1-Nf)*Cd = 0.
    assert(this->number_factor == 0); // Nf*Cd = Cd
    mat[{1, 0}] = value;
  } else if (this->additional_term < 0) {
    // We can only have either a C or a C*Z term, since Nf*C = 0.
    assert(this->number_factor == 0); // (1-Nf)*C = C
    mat[{0, 1}] = value;
  } else {
    mat[{0, 0}] = this->number_factor <= 0 ? value : 0.;
    mat[{1, 1}] = this->number_factor >= 0 ? value : 0.;
    assert(this->number_factor & 1 || !this->phase);
  }
  return std::move(mat);
}

std::string fermion_operator::to_string(bool include_degrees) const {
  if (include_degrees) return this->op_code_to_string() + "(" + std::to_string(target) + ")";
  else return this->op_code_to_string();
}

// comparisons

bool fermion_operator::operator==(const fermion_operator &other) const {
  if (this->quenched && other.quenched) return true;
  return this->target == other.target &&
         this->additional_term == other.additional_term && 
         this->number_factor == other.number_factor &&
         this->phase == other.phase;
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