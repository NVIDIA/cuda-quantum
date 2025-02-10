/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <complex>
#include <map>
#include <vector>

#include "cudaq/utils/tensor.h"
#include "cudaq/operators.h"

namespace cudaq {

template <typename HandlerTy> 
class product_operator;

// FIXME: rename?
class boson_operator : operator_handler{

private:

  // ...
  int id;
  int target;

  boson_operator(int op, int target);

public:

  // read-only properties

  /// @brief The degrees of freedom that the operator acts on in canonical
  /// order.
  virtual std::vector<int> degrees() const;

  virtual bool is_identity() const;

  // constructors and destructors

  ~boson_operator() = default;

  // assignments

  // evaluations

  /// @brief Return the `matrix_operator` as a matrix.
  /// @arg  `dimensions` : A map specifying the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0 : 2, 1 : 2}`.
  virtual matrix_2 to_matrix(std::map<int, int> &dimensions,
                             std::map<std::string, std::complex<double>> parameters = {}) const;

  virtual std::string to_string(bool include_degrees) const;

  // comparisons

  bool operator==(const boson_operator &other) const;

  // defined operators

  // multiplicative identity
  static boson_operator one(int degree);
  static product_operator<boson_operator> identity(int degree);
  static product_operator<boson_operator> create(int degree);
  static product_operator<boson_operator> annihilate(int degree);
  static product_operator<boson_operator> number(int degree);
};

} // namespace cudaq