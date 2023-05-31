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

class BaseEnsmallen : public cudaq::optimizer {
protected:
  void validate(optimizable_function &optFunction);

public:
  std::optional<std::size_t> max_eval;
  std::optional<std::vector<double>> initial_parameters;
  std::optional<std::vector<double>> lower_bounds;
  std::optional<std::vector<double>> upper_bounds;
  std::optional<double> f_tol;
  std::optional<double> step_size;
};

#define CUDAQ_ENSMALLEN_ALGORITHM_TYPE(NAME, NEEDS_GRADIENTS, EXTRA_PARAMS)    \
  class NAME : public BaseEnsmallen {                                          \
  public:                                                                      \
    NAME() = default;                                                          \
    bool requiresGradients() override { return NEEDS_GRADIENTS; }              \
    optimization_result                                                        \
    optimize(const int dim, optimizable_function &&opt_function) override;     \
    EXTRA_PARAMS                                                               \
  };

CUDAQ_ENSMALLEN_ALGORITHM_TYPE(lbfgs, true, )
CUDAQ_ENSMALLEN_ALGORITHM_TYPE(spsa, false, std::optional<double> alpha;
                               std::optional<double> gamma;
                               std::optional<double> eval_step_size;)

CUDAQ_ENSMALLEN_ALGORITHM_TYPE(adam, true,
                               std::optional<std::size_t> batch_size;
                               std::optional<double> beta1;
                               std::optional<double> beta2;
                               std::optional<double> eps;)

CUDAQ_ENSMALLEN_ALGORITHM_TYPE(gradient_descent, true, )
CUDAQ_ENSMALLEN_ALGORITHM_TYPE(sgd, true,
                               std::optional<std::size_t> batch_size;)

} // namespace cudaq::optimizers
