/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/utils/matrix.h"
#include "operator_leafs.h"
#include <complex>
#include <set>
#include <unordered_map>
#include <vector>

namespace cudaq {

// commutation_relations

std::unordered_map<uint, std::complex<double>>
    commutation_relations::exchange_factors = {
        {-1, 1.},  // default relation
        {-2, -1.}, // fermion relation
};

void commutation_relations::define(uint group_id,
                                   std::complex<double> exchange_factor) {
  auto result = commutation_relations::exchange_factors.insert(
      {group_id, exchange_factor});
  if (!result.second)
    throw std::invalid_argument("commutation relations for group id '" +
                                std::to_string(group_id) +
                                "' are already defined");
}

std::complex<double> commutation_relations::commutation_factor() const {
  auto it = commutation_relations::exchange_factors.find(id);
  assert(it != commutation_relations::exchange_factors.cend());
  return it->second;
}

bool commutation_relations::operator==(
    const commutation_relations &other) const {
  return this->id == other.id;
}

// operator_handler

commutation_relations operator_handler::custom_commutation_relations(uint id) {
  auto it = commutation_relations::exchange_factors.find(id);
  if (it == commutation_relations::exchange_factors.cend())
    throw std::range_error("no commutation relations with id '" +
                           std::to_string(id) + "' has been defined");
  return commutation_relations(id);
}

// operator_handler::matrix_evaluation

operator_handler::matrix_evaluation::matrix_evaluation() = default;

operator_handler::matrix_evaluation::matrix_evaluation(
    std::vector<int> &&degrees, complex_matrix &&matrix)
    : degrees(std::move(degrees)), matrix(std::move(matrix)) {
#if !defined(NDEBUG)
  std::set<int> unique_degrees;
  for (auto d : this->degrees)
    unique_degrees.insert(d);
  assert(unique_degrees.size() == this->degrees.size());
#endif
}

operator_handler::matrix_evaluation::matrix_evaluation(
    matrix_evaluation &&other)
    : degrees(std::move(other.degrees)), matrix(std::move(other.matrix)) {}

operator_handler::matrix_evaluation &
operator_handler::matrix_evaluation::operator=(matrix_evaluation &&other) {
  if (this != &other) {
    this->degrees = std::move(other.degrees);
    this->matrix = std::move(other.matrix);
  }
  return *this;
}

// operator_handler::canonical_evaluation

operator_handler::canonical_evaluation::canonical_evaluation() = default;

operator_handler::canonical_evaluation::canonical_evaluation(
    canonical_evaluation &&other)
    : terms(std::move(other.terms)) {}

operator_handler::canonical_evaluation &
operator_handler::canonical_evaluation::operator=(
    canonical_evaluation &&other) {
  if (this != &other)
    this->terms = std::move(other.terms);
  return *this;
}

void operator_handler::canonical_evaluation::push_back(
    std::pair<std::complex<double>, std::string> &&term) {
  this->terms.push_back(term);
}

void operator_handler::canonical_evaluation::push_back(const std::string &op) {
  assert(this->terms.size() != 0);
  this->terms.back().second.append(op);
}

} // namespace cudaq