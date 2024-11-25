/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <complex>
#include <cudensitymat.h>
#include <memory>
#include <operators.h>
#include <variant>
#include <vector>

namespace cudaq {
class elementary_operator;
class scalar_operator;
class product_operator;
class operator_sum;

template <typename TEval>
class operator_arithmetics {
public:
  virtual ~operator_arithmetics() = default;

  virtual TEval evaluate(const Operator &op) const = 0;
  // virtual TEval add(const TEval &val1, const TEval &val2) = 0;
  // virtual TEval mul(const TEval &val1, const TEval &val2) = 0;
  // virtual TEval tensor(const TEval &val1, const TEval &val2) = 0;
};
} // namespace cudaq
