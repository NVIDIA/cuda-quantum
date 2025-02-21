/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cmath>
#include <complex>
#include <unordered_map>
#include <vector>

#include "boson_operators.h"
#include "cudaq/operators.h"
#include "cudaq/utils/tensor.h"

namespace cudaq {

// private helpers

std::string boson_operator::op_code_to_string() const {
  if (this->additional_terms == 0 && this->number_offsets.size() == 0)
    return "I";
  std::string str;
  for (auto offset : this->number_offsets) {
    if (offset == 0)
      str += "N";
    else if (offset > 0)
      str += "(N+" + std::to_string(offset) + ")";
    else
      str += "(N" + std::to_string(offset) + ")";
  }
  for (auto i = 0; i < this->additional_terms; ++i)
    str += "Ad";
  for (auto i = 0; i > this->additional_terms; --i)
    str += "A";
  return std::move(str);
}

void boson_operator::inplace_mult(const boson_operator &other) {
  this->number_offsets.reserve(this->number_offsets.size() +
                               other.number_offsets.size());

  // first permute all number operators of RHS to the left; for x = #
  // permutations, if we have "unpaired" creation operators, the number operator
  // becomes (N + x), if we have "unpaired" annihilation operators, the number
  // operator becomes (N - x).
  for (auto offset : other.number_offsets)
    this->number_offsets.push_back(offset - this->additional_terms);

  // now we can combine the creation and annihilation operators;
  if (this->additional_terms > 0) { // we have "unpaired" creation operators
    // using ad*a = N and ad*N = (N - 1)*ad, each created number operator has an
    // offset of -(x - 1 - i), where x is the number of creation operators, and
    // i is the number of creation operators we already combined
    for (auto i = 1;
         i <= this->additional_terms && i <= -other.additional_terms; ++i)
      this->number_offsets.push_back(i - this->additional_terms);
  } else if (this->additional_terms <
             0) { // we have "unpaired" annihilation operators
    // using a*ad = (N + 1) and a*N = (N + 1)*a, each created number operator
    // has an offset of (x - i), where x is the number of annihilation
    // operators, and i is the number of annihilation operators we already
    // combined
    for (auto i = 0; i > this->additional_terms && i > -other.additional_terms;
         --i)
      this->number_offsets.push_back(i - this->additional_terms);
  }

  // finally, we update the number of remaining unpaired operators
  this->additional_terms += other.additional_terms;
}

// read-only properties

std::string boson_operator::unique_id() const {
  return this->op_code_to_string() + std::to_string(target);
}

std::vector<int> boson_operator::degrees() const { return {this->target}; }

// constructors

boson_operator::boson_operator(int target)
    : target(target), additional_terms(0) {}

boson_operator::boson_operator(int target, int op_id)
    : target(target), additional_terms(0) {
  assert(0 <= op_id < 4);
  if (op_id == 1) // create
    this->additional_terms = 1;
  else if (op_id == 2) // annihilate
    this->additional_terms = -1;
  else if (op_id == 3) // number
    this->number_offsets.push_back(0);
}

// evaluations

matrix_2 boson_operator::to_matrix(
    std::unordered_map<int, int> &dimensions,
    const std::unordered_map<std::string, std::complex<double>> &parameters)
    const {
  auto it = dimensions.find(this->target);
  if (it == dimensions.end())
    throw std::runtime_error("missing dimension for degree " +
                             std::to_string(this->target));
  auto dim = it->second;

  auto mat = matrix_2(dim, dim);
  if (this->additional_terms > 0) {
    for (std::size_t column = 0; column + this->additional_terms < dim;
         column++) {
      auto row = column + this->additional_terms;
      mat[{row, column}] = 1.;
      for (auto offset : this->number_offsets)
        mat[{row, column}] *= (row + offset);
      for (auto offset = this->additional_terms; offset > 0; --offset)
        mat[{row, column}] *= std::sqrt(column + offset);
    }
  } else if (this->additional_terms < 0) {
    for (std::size_t row = 0; row - this->additional_terms < dim; row++) {
      auto column = row - this->additional_terms;
      mat[{row, column}] = 1.;
      for (auto offset : this->number_offsets)
        mat[{row, column}] *= (row + offset);
      for (auto offset = -this->additional_terms; offset > 0; --offset)
        mat[{row, column}] *= std::sqrt(row + offset);
    }
  } else {
    for (std::size_t i = 0; i < dim; i++) {
      mat[{i, i}] = 1.;
      for (auto offset : this->number_offsets)
        mat[{i, i}] *= (i + offset);
    }
  }
  return std::move(mat);
}

std::string boson_operator::to_string(bool include_degrees) const {
  if (include_degrees)
    return this->op_code_to_string() + "(" + std::to_string(target) + ")";
  else
    return this->op_code_to_string();
}

// comparisons

bool boson_operator::operator==(const boson_operator &other) const {
  return this->additional_terms == other.additional_terms &&
         this->number_offsets == other.number_offsets &&
         this->target == other.target;
}

// defined operators

operator_sum<boson_operator> boson_operator::empty() {
  return operator_handler::empty<boson_operator>();
}

product_operator<boson_operator> boson_operator::identity() {
  return operator_handler::identity<boson_operator>();
}

product_operator<boson_operator> boson_operator::identity(int degree) {
  return product_operator(boson_operator(degree));
}

product_operator<boson_operator> boson_operator::create(int degree) {
  return product_operator(boson_operator(degree, 1));
}

product_operator<boson_operator> boson_operator::annihilate(int degree) {
  return product_operator(boson_operator(degree, 2));
}

product_operator<boson_operator> boson_operator::number(int degree) {
  return product_operator(boson_operator(degree, 3));
}

operator_sum<boson_operator> boson_operator::position(int degree) {
  return 0.5 *
         (boson_operator::create(degree) + boson_operator::annihilate(degree));
}

operator_sum<boson_operator> boson_operator::momentum(int degree) {
  return 0.5j *
         (boson_operator::create(degree) - boson_operator::annihilate(degree));
}

} // namespace cudaq