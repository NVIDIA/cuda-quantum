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

#include "boson_operators.h"
#include "cudaq/operators.h"
#include "cudaq/utils/tensor.h"

namespace cudaq {

// FIXME: GET RID OF THIS AND INSTEAD MAKE SURE WE AGGREGATE TERMS PROPERLY
#if !defined(NDEBUG)
bool boson_operator::can_be_canonicalized = false;
#endif

// private helpers

std::string boson_operator::op_code_to_string() const {
  if (this->op_code == 1)
    return "Ad";
  else if (this->op_code == 2)
    return "A";
  else if (this->op_code == 3)
    return "AdA";
  else
    return "I";
}

// read-only properties

std::string boson_operator::unique_id() const {
  return this->op_code_to_string() + std::to_string(target);
}

std::vector<int> boson_operator::degrees() const { return {this->target}; }

// constructors

boson_operator::boson_operator(int target) : op_code(0), target(target) {}

boson_operator::boson_operator(int target, int op_id)
    : op_code(op_id), target(target) {
  assert(0 <= op_id < 4);
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
  if (this->op_code == 1) { // create
    for (std::size_t i = 0; i + 1 < dim; i++)
      mat[{i + 1, i}] = std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
  } else if (this->op_code == 2) { // annihilate
    for (std::size_t i = 0; i + 1 < dim; i++)
      mat[{i, i + 1}] = std::sqrt(static_cast<double>(i + 1)) + 0.0j;
  } else if (this->op_code == 3) { // number
    for (std::size_t i = 0; i < dim; i++)
      mat[{i, i}] = static_cast<double>(i) + 0.0j;
  } else { // id
    mat[{0, 0}] = 1.0;
    mat[{1, 1}] = 1.0;
  }
  return mat;
}

std::string boson_operator::to_string(bool include_degrees) const {
  if (include_degrees)
    return this->op_code_to_string() + "(" + std::to_string(target) + ")";
  else
    return this->op_code_to_string();
}

// comparisons

bool boson_operator::operator==(const boson_operator &other) const {
  return this->op_code == other.op_code && this->target == other.target;
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

// FIXME: add position, momentum, others?

} // namespace cudaq