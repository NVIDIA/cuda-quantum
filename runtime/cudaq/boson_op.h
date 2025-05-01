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

#include "cudaq/operators/operator_leafs.h"
#include "cudaq/utils/matrix.h"

namespace cudaq {
class boson_handler : public operator_handler {
  template <typename T>
  friend class product_op;

private:
  // Each boson operator is represented as number operators along with an
  // offset to add to each number operator, as well as an integer indicating
  // how many creation or annihilation terms follow the number operators.
  // See the implementation of the in-place multiplication to understand
  // the meaning and purpose of this representation. In short, this
  // representation allows us to perform a perfect in-place multiplication.
  int additional_terms;
  std::vector<int> number_offsets;
  std::size_t degree;

  // 0 = I, Ad = 1, A = 2, AdA = 3
  boson_handler(std::size_t target, int op_code);

  std::string op_code_to_string() const;
  virtual std::string op_code_to_string(
      std::unordered_map<std::size_t, std::int64_t> &dimensions) const override;

  void inplace_mult(const boson_handler &other);

public:
  // read-only properties

  virtual std::string unique_id() const override;

  virtual std::vector<std::size_t> degrees() const override;

  std::size_t target() const;

  // constructors and destructors

  boson_handler(std::size_t target);

  ~boson_handler() = default;

  // evaluations

  /// @brief Return the matrix representation of the operator in the eigenbasis
  /// of the number operator.
  /// @param  `dimensions` : A map specifying the dimension, that is the number
  /// of eigenstates, for each degree of freedom.
  virtual complex_matrix
  to_matrix(std::unordered_map<std::size_t, std::int64_t> &dimensions,
            const std::unordered_map<std::string, std::complex<double>>
                &parameters = {}) const override;

  virtual std::string to_string(bool include_degrees) const override;

  // comparisons

  /// @returns True if, and only if, the two operators have the same effect on
  /// any state.
  bool operator==(const boson_handler &other) const;

  // defined operators

  static boson_handler create(std::size_t degree);
  static boson_handler annihilate(std::size_t degree);
  static boson_handler number(std::size_t degree);
};
} // namespace cudaq

// needs to be down here such that the handler is defined
// before we include the template declarations that depend on it
#include "cudaq/operators.h"

namespace cudaq::boson {
product_op<boson_handler> create(std::size_t target);
product_op<boson_handler> annihilate(std::size_t target);
product_op<boson_handler> number(std::size_t target);
sum_op<boson_handler> position(std::size_t target);
sum_op<boson_handler> momentum(std::size_t target);
} // namespace cudaq::boson
