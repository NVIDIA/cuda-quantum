/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq.h"
#include <Eigen/Dense>
#include <map>
#include <operator_arithmetics.h>
#include <tuple>

namespace cudaq {
class Evaluated {
public:
  Evaluated(const std::vector<int> &degrees,
            const cudaq::complex_matrix &matrix);
  const std::vector<int> &get_degrees() const;
  const cudaq::complex_matrix &get_matrix() const;

private:
  std::vector<int> _degrees;
  cudaq::complex_matrix _matrix;
};

class matrix_arithmetics : public operator_arithmetics<Evaluated> {
public:
  matrix_arithmetics(const std::map<int, int> &dimensions);

  Evaluated evaluate(const Operator &op) const override;
  // std::shared_ptr<Evaluated> add(const std::shared_ptr<Evaluated> &op1, const
  // std::shared_ptr<Evaluated> &op2) override; std::shared_ptr<Evaluated>
  // mul(const Evaluated &op1, const Evaluated &op2) override;
  // std::shared_ptr<Evaluated> tensor(const Evaluated &op1, const Evaluated
  // &op2) override;

  std::vector<int> compute_permutation(const std::vector<int> &op_degrees,
                                       const std::vector<int> &canon_degrees);
  std::pair<cudaq::complex_matrix, std::vector<int>>
  canonicalize(const cudaq::complex_matrix &op_matrix,
               const std::vector<int> &op_degrees);

private:
  std::map<int, int> dimensions;
};
} // namespace cudaq
