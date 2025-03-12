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
#include "cudaq/spin_op.h" // for pauli
#include "cudaq/utils/matrix.h"

namespace cudaq {

class spin_handler : public operator_handler {
  template <typename T>
  friend class product_op;

private:
  // I = 0, Z = 1, X = 2, Y = 3
  int op_code;
  int degree;

  spin_handler(int target, int op_code);

  // private helpers

  std::string op_code_to_string() const;
  virtual std::string
  op_code_to_string(std::unordered_map<int, int> &dimensions) const override;

  std::complex<double> inplace_mult(const spin_handler &other);

public:
  // read-only properties

  pauli as_pauli() const;

  virtual std::string unique_id() const override;

  virtual std::vector<int> degrees() const override;

  int target() const;

  // constructors and destructors

  spin_handler(int target);

  spin_handler(pauli p, int target);

  ~spin_handler() = default;

  // evaluations

  /// @brief Computes the matrix representation of a Pauli string.
  /// By default, the ordering of the matrix matches the ordering of the Pauli
  /// string,
  static complex_matrix to_matrix(std::string pauli,
                                  std::complex<double> coeff = 1.,
                                  bool invert_order = false);

  /// @brief Return the `matrix_handler` as a matrix.
  /// @arg  `dimensions` : A map specifying the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0 : 2, 1 : 2}`.
  virtual complex_matrix
  to_matrix(std::unordered_map<int, int> &dimensions,
            const std::unordered_map<std::string, std::complex<double>>
                &parameters = {}) const override;

  virtual std::string to_string(bool include_degrees) const override;

  // comparisons

  bool operator==(const spin_handler &other) const;

  // defined operators

  static product_op<spin_handler> i(int degree);
  static product_op<spin_handler> z(int degree);
  static product_op<spin_handler> x(int degree);
  static product_op<spin_handler> y(int degree);
  static sum_op<spin_handler> plus(int degree);
  static sum_op<spin_handler> minus(int degree);
};

} // namespace cudaq