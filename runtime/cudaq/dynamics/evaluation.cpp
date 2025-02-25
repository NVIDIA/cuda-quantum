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

operator_handler::matrix_evaluation::matrix_evaluation() = default;

operator_handler::matrix_evaluation::matrix_evaluation(std::vector<int> &&degrees, matrix_2 &&matrix)
: degrees(std::move(degrees)), matrix(std::move(matrix)) {
#if !defined(NDEBUG)
  std::set<int> unique_degrees;
  for (auto d : this->degrees)
  unique_degrees.insert(d);
  assert(unique_degrees.size() == this->degrees.size());
#endif
}

operator_handler::matrix_evaluation::matrix_evaluation(matrix_evaluation &&other)
: degrees(std::move(other.degrees)), matrix(std::move(other.matrix)) {}

operator_handler::matrix_evaluation& operator_handler::matrix_evaluation::operator=(matrix_evaluation &&other) {
  if (this != &other) {
    this->degrees = std::move(other.degrees);
    this->matrix = std::move(other.matrix);
  }
  return *this;
}

// implementation for operator_handler::canonical_evaluation

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