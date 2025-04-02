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

#include "cudaq/operators.h"
#include "cudaq/utils/matrix.h"

#include "boson_operators.h"

namespace cudaq {

// private helpers

std::string boson_handler::op_code_to_string() const {
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

std::string boson_handler::op_code_to_string(
    std::unordered_map<int, int> &dimensions) const {
  auto it = dimensions.find(this->degree);
  if (it == dimensions.end())
    throw std::runtime_error("missing dimension for degree " +
                             std::to_string(this->degree));
  return this->op_code_to_string();
}

void boson_handler::inplace_mult(const boson_handler &other) {
  this->number_offsets.reserve(this->number_offsets.size() +
                               other.number_offsets.size());

  // first permute all number operators of RHS to the left; for x = #
  // permutations, if we have "unpaired" creation operators, the number operator
  // becomes (N + x), if we have "unpaired" annihilation operators, the number
  // operator becomes (N - x).
  auto it = this->number_offsets
                .cbegin(); // we will sort the offsets from biggest to smallest
  for (auto offset : other.number_offsets) {
    while (it != this->number_offsets.cend() &&
           *it >= offset - this->additional_terms)
      ++it;
    this->number_offsets.insert(it, offset - this->additional_terms);
  }

  // now we can combine the creation and annihilation operators;
  if (this->additional_terms > 0) { // we have "unpaired" creation operators
    // using ad*a = N and ad*N = (N - 1)*ad, each created number operator has an
    // offset of -(x - 1 - i), where x is the number of creation operators, and
    // i is the number of creation operators we already combined
    it = this->number_offsets.cbegin();
    for (auto i = std::min(this->additional_terms, -other.additional_terms);
         i > 0; --i) {
      // we make sure to have offsets get smaller as we go to keep the sorting
      // cheap
      while (it != this->number_offsets.cend() &&
             *it >= i - this->additional_terms)
        ++it;
      this->number_offsets.insert(it, i - this->additional_terms);
    }
  } else if (this->additional_terms <
             0) { // we have "unpaired" annihilation operators
    // using a*ad = (N + 1) and a*N = (N + 1)*a, each created number operator
    // has an offset of (x - i), where x is the number of annihilation
    // operators, and i is the number of annihilation operators we already
    // combined
    it = this->number_offsets.cbegin();
    for (auto i = 0; i > this->additional_terms && i > -other.additional_terms;
         --i) {
      // we make sure to have offsets get smaller as we go to keep the sorting
      // cheap
      while (it != this->number_offsets.cend() &&
             *it >= i - this->additional_terms)
        ++it;
      this->number_offsets.insert(it, i - this->additional_terms);
    }
  }

  // finally, we update the number of remaining unpaired operators
  this->additional_terms += other.additional_terms;

#if !defined(NDEBUG)
  // we sort the number offsets, such that the equality comparison and the
  // operator id perfectly reflects the mathematical evaluation of the operator
  auto sorted_offsets = this->number_offsets;
  std::sort(sorted_offsets.begin(), sorted_offsets.end(), std::greater<int>());
  assert(sorted_offsets == this->number_offsets);
#endif
}

// read-only properties

std::string boson_handler::unique_id() const {
  return this->op_code_to_string() + std::to_string(this->degree);
}

std::vector<int> boson_handler::degrees() const { return {this->degree}; }

// constructors

boson_handler::boson_handler(int target)
    : degree(target), additional_terms(0) {}

boson_handler::boson_handler(int target, int op_id)
    : degree(target), additional_terms(0) {
  assert(0 <= op_id && op_id < 4);
  if (op_id == 1) // create
    this->additional_terms = 1;
  else if (op_id == 2) // annihilate
    this->additional_terms = -1;
  else if (op_id == 3) // number
    this->number_offsets.push_back(0);
}

// evaluations

complex_matrix boson_handler::to_matrix(
    std::unordered_map<int, int> &dimensions,
    const std::unordered_map<std::string, std::complex<double>> &parameters)
    const {
  auto it = dimensions.find(this->degree);
  if (it == dimensions.end())
    throw std::runtime_error("missing dimension for degree " +
                             std::to_string(this->degree));
  auto dim = it->second;

  auto mat = complex_matrix(dim, dim);
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

std::string boson_handler::to_string(bool include_degrees) const {
  if (include_degrees)
    return this->op_code_to_string() + "(" + std::to_string(this->degree) + ")";
  else
    return this->op_code_to_string();
}

// comparisons

bool boson_handler::operator==(const boson_handler &other) const {
  return this->degree == other.degree &&
         this->additional_terms == other.additional_terms &&
         this->number_offsets == other.number_offsets;
}

// defined operators

product_op<boson_handler> boson_handler::create(int degree) {
  return product_op(boson_handler(degree, 1));
}

product_op<boson_handler> boson_handler::annihilate(int degree) {
  return product_op(boson_handler(degree, 2));
}

product_op<boson_handler> boson_handler::number(int degree) {
  return product_op(boson_handler(degree, 3));
}

sum_op<boson_handler> boson_handler::position(int degree) {
  return 0.5 *
         (boson_handler::create(degree) + boson_handler::annihilate(degree));
}

sum_op<boson_handler> boson_handler::momentum(int degree) {
  return std::complex<double>(0., 0.5) *
         (boson_handler::create(degree) - boson_handler::annihilate(degree));
}

} // namespace cudaq