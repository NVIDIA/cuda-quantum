/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <set>
#include <vector>
#include "cudaq/utils/tensor.h"
#include "operator_leafs.h"

namespace cudaq {

// implementation for operator_handler::matrix_evaluation

const std::vector<int>& operator_handler::matrix_evaluation::degrees() const {
  return this->targets;
}

const matrix_2& operator_handler::matrix_evaluation::matrix() const {
  return this->value;
}

operator_handler::matrix_evaluation::matrix_evaluation() = default;

operator_handler::matrix_evaluation::matrix_evaluation(std::vector<int> &&degrees, matrix_2 &&matrix)
: targets(std::move(degrees)), value(std::move(matrix)) {
#if !defined(NDEBUG)
  std::set<int> unique_degrees;
  for (auto d : this->targets)
  unique_degrees.insert(d);
  assert(unique_degrees.size() == this->targets.size());
#endif
}

operator_handler::matrix_evaluation::matrix_evaluation(matrix_evaluation &&other)
: targets(std::move(other.targets)), value(std::move(other.value)) {}

operator_handler::matrix_evaluation& operator_handler::matrix_evaluation::operator=(matrix_evaluation &&other) {
  if (this != &other) {
    this->targets = std::move(other.targets);
    this->value = std::move(other.value);
  }
  return *this;
}

// implementation for operator_handler::canonical_evaluation

const std::vector<std::pair<std::complex<double>, std::string>>& operator_handler::canonical_evaluation::get_terms() {
  return this->terms;
}

operator_handler::canonical_evaluation::canonical_evaluation() = default;

operator_handler::canonical_evaluation::canonical_evaluation(canonical_evaluation &&other) 
  : terms(std::move(other.terms)) {}

operator_handler::canonical_evaluation& operator_handler::canonical_evaluation::operator=(canonical_evaluation &&other) {
  if (this != &other)
    this->terms = std::move(other.terms);
  return *this;
}

void operator_handler::canonical_evaluation::push_back(std::pair<std::complex<double>, std::string> &&term) {
  this->terms.push_back(term);
}

void operator_handler::canonical_evaluation::push_back(const std::string &op) {
  assert(this->terms.size() != 0);
  this->terms.back().second.append(op);
}

} // namespace cudaq