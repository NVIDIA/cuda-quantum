/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/algorithms/optimizer.h"

namespace cudaq::optimizers {
struct max_eval;
struct initial_parameters;

class base_nlopt : public cudaq::optimizer {
public:
  std::optional<int> max_eval;
  std::optional<std::vector<double>> initial_parameters;
  std::optional<std::vector<double>> lower_bounds;
  std::optional<std::vector<double>> upper_bounds;
  std::optional<double> f_tol;
};

#define CUDAQ_NLOPT_ALGORITHM_TYPE(NAME, NEEDS_GRADIENTS)                      \
  class NAME : public base_nlopt {                                             \
  public:                                                                      \
    NAME();                                                                    \
    virtual ~NAME();                                                           \
    bool requiresGradients() override { return NEEDS_GRADIENTS; }              \
    optimization_result                                                        \
    optimize(const int dim, optimizable_function &&opt_function) override;     \
  };

CUDAQ_NLOPT_ALGORITHM_TYPE(cobyla, false)
CUDAQ_NLOPT_ALGORITHM_TYPE(neldermead, false)

} // namespace cudaq::optimizers
