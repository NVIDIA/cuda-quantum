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

template <typename HandlerTy>
class product_operator;

// FIXME: rename?
class boson_operator : public operator_handler {
  template <typename T>
  friend class product_operator;

private:
  // 0 = I, 1 = Ad (create), 2 = A (annihilate), 3 = AdA (number)
  int op_code;
  int target;

  boson_operator(int target, int op_code);

  std::string op_code_to_string() const;

public:
// FIXME: GET RID OF THIS
#if !defined(NDEBUG)
  static bool
      can_be_canonicalized; // needs to be false; no canonical order can be
                            // defined for matrix operator expressions
#endif

  // read-only properties

  virtual std::string unique_id() const;

  /// @brief The degrees of freedom that the operator acts on in canonical
  /// order.
  virtual std::vector<int> degrees() const;

  // constructors and destructors

  boson_operator(int target);

  ~boson_operator() = default;

  // evaluations

  /// @brief Return the `matrix_operator` as a matrix.
  /// @arg  `dimensions` : A map specifying the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0 : 2, 1 : 2}`.
  virtual matrix_2
  to_matrix(std::unordered_map<int, int> &dimensions,
            const std::unordered_map<std::string, std::complex<double>>
                &parameters = {}) const;

  virtual std::string to_string(bool include_degrees) const;

  // comparisons

  bool operator==(const boson_operator &other) const;

  // defined operators

  static operator_sum<boson_operator> empty();
  static product_operator<boson_operator> identity();

  static product_operator<boson_operator> identity(int degree);
  static product_operator<boson_operator> create(int degree);
  static product_operator<boson_operator> annihilate(int degree);
  static product_operator<boson_operator> number(int degree);
};

} // namespace cudaq