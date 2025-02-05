/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/operators.h"
#include "spin_operators.h"
#include <vector>

namespace cudaq {

// read-only properties

std::vector<int> spin_operator::degrees() const {
  return {this->target};
}

// constructors

spin_operator::spin_operator(int op_id, int target) 
  : id(op_id), target(target) {
    assert(0 <= op_id < 4);
}

bool spin_operator::is_identity() const {
    return this->id == 0;
}

// evaluations

matrix_2 spin_operator::to_matrix(std::map<int, int> &dimensions,
                                  std::map<std::string, std::complex<double>> parameters) const {
  auto it = dimensions.find(this->target);
  if (it == dimensions.end())
    dimensions[this->target] = 2;
  else if (it->second != 2)
    throw std::runtime_error("dimension for spin operator must be 2");

  // FIXME: CHECK CONVENTIONS
  auto mat = matrix_2(2, 2);
  if (this->id == 1) { // Z
    mat[{0, 0}] = 1.0;
    mat[{1, 1}] = -1.0;
  } else if (this->id == 2) { // X
    mat[{0, 1}] = 1.0;
    mat[{1, 0}] = 1.0;
  } else if (this->id == 3) { // Y
    mat[{0, 1}] = -1.0j;
    mat[{1, 0}] = 1.0j;
  } else { // I
    mat[{0, 0}] = 1.0;
    mat[{1, 1}] = 1.0;
  }
  return mat;
}

// comparisons

bool spin_operator::operator==(const spin_operator &other) const {
  return this->id == other.id && this->target == other.target;
}

// defined operators

spin_operator spin_operator::identity(int degree) {
  return spin_operator(0, degree);
}

product_operator<spin_operator> spin_operator::i(int degree) {
  return product_operator<spin_operator>(1., spin_operator(0, degree));
}

product_operator<spin_operator> spin_operator::z(int degree) {
  return product_operator<spin_operator>(1., spin_operator(1, degree));
}

product_operator<spin_operator> spin_operator::x(int degree) {
  return product_operator<spin_operator>(1., spin_operator(2, degree));
}

product_operator<spin_operator> spin_operator::y(int degree) {
  return product_operator<spin_operator>(1., spin_operator(3, degree));
}

} // namespace cudaq