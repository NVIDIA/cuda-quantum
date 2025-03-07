/*******************************************************************************
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

// FIXME: rename to spin ...
class spin_operator : public operator_handler {
  template <typename T>
  friend class product_operator;

private:
  // I = 0, Z = 1, X = 2, Y = 3
  int op_code;
  int target;

  spin_operator(int target, int op_code);

  // private helpers

  std::string op_code_to_string() const;
  std::string
  op_code_to_string(std::unordered_map<int, int> &dimensions) const override;

  std::complex<double> inplace_mult(const spin_operator &other);

public:
  // read-only properties

  virtual std::string unique_id() const override;

  virtual std::vector<int> degrees() const override;

  // constructors and destructors

  spin_operator(int target);

  ~spin_operator() = default;

  // evaluations

  /// @brief Computes the matrix representation of a Pauli string.
  /// By default, the ordering of the matrix matches the ordering of the Pauli
  /// string,
  static matrix_2 to_matrix(std::string pauli, std::complex<double> coeff = 1.,
                            bool invert_order = false);

  /// @brief Return the `matrix_operator` as a matrix.
  /// @arg  `dimensions` : A map specifying the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0 : 2, 1 : 2}`.
  virtual matrix_2
  to_matrix(std::unordered_map<int, int> &dimensions,
            const std::unordered_map<std::string, std::complex<double>>
                &parameters = {}) const override;

  virtual std::string to_string(bool include_degrees) const override;

  // comparisons

  bool operator==(const spin_operator &other) const;

  // defined operators

  static operator_sum<spin_operator> empty();
  static product_operator<spin_operator> identity();

  static product_operator<spin_operator> i(int degree);
  static product_operator<spin_operator> z(int degree);
  static product_operator<spin_operator> x(int degree);
  static product_operator<spin_operator> y(int degree);
};

} // namespace cudaq