/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <complex>
#include <unordered_map>
#include <vector>

#include "cudaq/operators.h"
#include "cudaq/utils/tensor.h"

namespace cudaq {

// FIXME: rename?
class boson_operator : public operator_handler {
  template <typename T>
  friend class product_operator;

private:
  // Each boson operator is represented as number operators along with an
  // offset to add to each number operator, as well as an integer indicating
  // how many creation or annihilation terms follow the number operators.
  // See the implementation of the in-place multiplication to understand
  // the meaning and purpose of this representation. In short, this
  // representation allows us to perform a perfect in-place multiplication.
  int additional_terms;
  std::vector<int> number_offsets;
  int target;

  // 0 = I, Ad = 1, A = 2, AdA = 3
  boson_operator(int target, int op_code);

  std::string op_code_to_string() const;
  virtual std::string
  op_code_to_string(std::unordered_map<int, int> &dimensions) const override;

  void inplace_mult(const boson_operator &other);

public:
  // read-only properties

  virtual std::string unique_id() const override;

  virtual std::vector<int> degrees() const override;

  // constructors and destructors

  boson_operator(int target);

  ~boson_operator() = default;

  // evaluations

  /// @brief Return the matrix representation of the operator in the eigenbasis
  /// of the number operator.
  /// @arg  `dimensions` : A map specifying the dimension, that is the number of
  /// eigenstates, for each degree of freedom.
  virtual matrix_2
  to_matrix(std::unordered_map<int, int> &dimensions,
            const std::unordered_map<std::string, std::complex<double>>
                &parameters = {}) const override;

  virtual std::string to_string(bool include_degrees) const override;

  // comparisons

  /// @returns True if, and only if, the two operators have the same effect on
  /// any state.
  bool operator==(const boson_operator &other) const;

  // defined operators

  static operator_sum<boson_operator> empty();
  static product_operator<boson_operator> identity();

  static product_operator<boson_operator> identity(int degree);
  static product_operator<boson_operator> create(int degree);
  static product_operator<boson_operator> annihilate(int degree);
  static product_operator<boson_operator> number(int degree);

  static operator_sum<boson_operator> position(int degree);
  static operator_sum<boson_operator> momentum(int degree);
};

} // namespace cudaq