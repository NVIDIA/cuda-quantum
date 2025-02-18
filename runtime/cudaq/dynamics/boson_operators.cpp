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

#if !defined(NDEBUG)
bool boson_operator::can_be_canonicalized = false;
#endif

// private helpers

std::string boson_operator::op_code_to_string() const {
  if (this->ad == 0 && this->a == 0) return "I";
  std::string str;
  for (auto i = 0; i < ad; ++i)
    str += "Ad";
  for (auto i = 0; i < a; ++i)
    str += "A";
  return std::move(str);
}

bool boson_operator::inplace_mult(const boson_operator &other) {
  if (this->a != 0 && other.ad != 0) return false;
  this->a += other.a;
  this->ad += other.ad;
  return true;
}

// read-only properties

std::string boson_operator::unique_id() const {
  return this->op_code_to_string() + std::to_string(target);
}

std::vector<int> boson_operator::degrees() const {
  return {this->target};
}

// constructors

boson_operator::boson_operator(int target) 
  : ad(0), a(0), target(target) {}

boson_operator::boson_operator(int target, int op_id) 
  : ad(op_id & 1), a((op_id & 2) >> 1), target(target) {
    assert(0 <= op_id < 4);
}

// evaluations

matrix_2 boson_operator::to_matrix(std::unordered_map<int, int> &dimensions,
                                   const std::unordered_map<std::string, std::complex<double>> &parameters) const {
  auto it = dimensions.find(this->target);
  if (it == dimensions.end())
    throw std::runtime_error("missing dimension for degree " + std::to_string(this->target));
  auto dim = it->second;

  // fixme: make matrix computation more efficient
  auto mat = matrix_2(dim, dim);
  mat[{0, 0}] = 1.0;
  mat[{1, 1}] = 1.0;
  if (this->ad == 0 && this->a == 0)
    return std::move(mat);

  auto create_mat = matrix_2(dim, dim);
  for (std::size_t i = 0; i + 1 < dim; i++)
    create_mat[{i + 1, i}] = std::sqrt(static_cast<double>(i + 1)) + 0.0j;
  auto annihilate_mat = matrix_2(dim, dim);
  for (std::size_t i = 0; i + 1 < dim; i++)
    annihilate_mat[{i, i + 1}] = std::sqrt(static_cast<double>(i + 1)) + 0.0j;
  for (auto i = 0; i < ad; ++i)
    mat *= create_mat;
  for (auto i = 0; i < a; ++i) 
    mat *= annihilate_mat;
  return std::move(mat);
}

std::string boson_operator::to_string(bool include_degrees) const {
  if (include_degrees) return this->op_code_to_string() + "(" + std::to_string(target) + ")";
  else return this->op_code_to_string();
}

// comparisons

bool boson_operator::operator==(const boson_operator &other) const {
  return this->ad == other.ad && this->a == other.a && this->target == other.target;
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