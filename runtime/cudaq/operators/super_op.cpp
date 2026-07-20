/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/operators.h"

namespace cudaq {
super_op &super_op::operator+=(const super_op &superOp) {
  m_terms.insert(m_terms.end(), superOp.m_terms.begin(), superOp.m_terms.end());
  return *this;
}

super_op
super_op::left_multiply(const cudaq::product_op<cudaq::matrix_handler> &op) {
  return super_op(std::make_pair(
      op, std::optional<cudaq::product_op<cudaq::matrix_handler>>{}));
}

super_op
super_op::right_multiply(const cudaq::product_op<cudaq::matrix_handler> &op) {
  return super_op(std::make_pair(
      std::optional<cudaq::product_op<cudaq::matrix_handler>>{}, op));
}

super_op super_op::left_right_multiply(
    const cudaq::product_op<cudaq::matrix_handler> &leftOp,
    const cudaq::product_op<cudaq::matrix_handler> &rightOp) {
  return super_op(std::make_pair(leftOp, rightOp));
}

super_op
super_op::left_multiply(const cudaq::sum_op<cudaq::matrix_handler> &op) {
  std::vector<term> productTerms;
  productTerms.reserve(op.num_terms());
  for (const cudaq::product_op<cudaq::matrix_handler> &prodOp : op) {
    productTerms.emplace_back(std::make_pair(
        prodOp, std::optional<cudaq::product_op<cudaq::matrix_handler>>{}));
  }
  return super_op(std::move(productTerms));
}

super_op
super_op::right_multiply(const cudaq::sum_op<cudaq::matrix_handler> &op) {
  std::vector<term> productTerms;
  productTerms.reserve(op.num_terms());
  for (const cudaq::product_op<cudaq::matrix_handler> &prodOp : op) {
    productTerms.emplace_back(std::make_pair(
        std::optional<cudaq::product_op<cudaq::matrix_handler>>{}, prodOp));
  }
  return super_op(std::move(productTerms));
}

super_op super_op::left_right_multiply(
    const cudaq::sum_op<cudaq::matrix_handler> &leftOp,
    const cudaq::sum_op<cudaq::matrix_handler> &rightOp) {
  std::vector<term> productTerms;
  productTerms.reserve(leftOp.num_terms() * rightOp.num_terms());
  for (const cudaq::product_op<cudaq::matrix_handler> &leftProdOp : leftOp) {
    for (const cudaq::product_op<cudaq::matrix_handler> &rightProdOp :
         rightOp) {
      productTerms.emplace_back(std::make_pair(leftProdOp, rightProdOp));
    }
  }
  return super_op(std::move(productTerms));
}

super_op::const_iterator super_op::begin() const { return m_terms.cbegin(); }

super_op::const_iterator super_op::end() const { return m_terms.cend(); }

super_op::super_op(term &&term) : m_terms({std::move(term)}) {}

super_op::super_op(std::vector<term> &&terms) : m_terms(std::move(terms)) {}

} // namespace cudaq
