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

#include "cudaq/utils/tensor.h"
#include "cudaq/operators.h"
#include "boson_operators.h"

namespace cudaq {

// read-only properties

std::vector<int> boson_operator::degrees() const {
  return {this->target};
}

// constructors

boson_operator::boson_operator(int target) 
  : id(0), target(target) {}

boson_operator::boson_operator(int target, int op_id) 
  : id(op_id), target(target) {
    assert(0 <= op_id < 4);
}

// evaluations

matrix_2 boson_operator::to_matrix(std::unordered_map<int, int> &dimensions,
                                   const std::unordered_map<std::string, std::complex<double>> &parameters) const {
  auto it = dimensions.find(this->target);
  if (it == dimensions.end())
    throw std::runtime_error("missing dimension for degree " + std::to_string(this->target));
  auto dim = it->second;

  auto mat = matrix_2(dim, dim);
  if (this->id == 1) { // create
    for (std::size_t i = 0; i + 1 < dim; i++)
      mat[{i + 1, i}] = std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
  } else if (this->id == 2) { // annihilate
    for (std::size_t i = 0; i + 1 < dim; i++)
        mat[{i, i + 1}] = std::sqrt(static_cast<double>(i + 1)) + 0.0j;
  } else if (this->id == 3) { // number
    for (std::size_t i = 0; i < dim; i++)
      mat[{i, i}] = static_cast<double>(i) + 0.0j;
  } else { // id
    mat[{0, 0}] = 1.0;
    mat[{1, 1}] = 1.0;
  }
  return mat;
}

std::string boson_operator::to_string(bool include_degrees) const {
  std::string op_str;
  if (this->id == 1) op_str = "create";
  else if (this->id == 2) op_str = "annihilate";
  else if (this->id == 3) op_str = "number";
  else op_str = "identity";
  if (include_degrees) return op_str + "(" + std::to_string(target) + ")";
  else return op_str;
}

// comparisons

bool boson_operator::operator==(const boson_operator &other) const {
  return this->id == other.id && this->target == other.target;
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