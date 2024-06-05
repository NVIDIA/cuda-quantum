/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
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

  virtual std::string serialize() const = 0;
  static BaseEnsmallen deserialize(const std::string &serialized_data);
};

class Adam : public BaseEnsmallen {
public:
  Adam() = default;
  bool requiresGradients() override { return true; }
  optimization_result optimize(const int dim, optimizable_function &&opt_function) override;
  std::string serialize() const override;
};

std::string Adam::serialize() const {
  std::ostringstream oss;

  if (max_eval.has_value()) {
    oss << "max_eval:" << max_eval.value() << ";";
  }

  if (initial_parameters.has_value()) {
    oss << "initial_parameters";
    for (double param : initial_parameters.value()) {
      oss << param << ",";
    }
    oss << ";";
  }

  if (lower_bounds.has_value()) {
    oss << "lower_bounds";
    for (double param : lower_bounds.value()) {
      oss << param << ",";
    }
    oss << ";";
  }

  if (upper_bounds.has_value()) {
    oss << "upper_bounds";
    for (double param : upper_bounds.value()) {
      oss << param << ",";
    }
    oss << ";";
  }

  if (f_tol.has_value()) {
    oss << "f_tol:" << f_tol.value() << ";";
  }

  if (step_size.has_value()) {
    oss << "step_size:" << step_size.value() << ";";
  }

  return oss.str();
}

Adam deserialize(const std::string &serialized_data) {
  Adam obj;

  std::istringstream iss(serialized_data);
  char delimiter = ';';
  std::string field;

  while (iss >> field >> delimiter) {
    if (field == "max_eval:") {
      iss >> *obj.max_eval;
    } else if (field == "initial_parameters:") {
      std::string params_str;
      iss >> params_str;
      std::istringstream params_ss(params_str);
      double param;
      while (params_ss >> param) {
        obj.initial_parameters->push_back(param);
        if (params_ss.peek() == ',')
          params_ss.ignore();
      }
    } else if (field == "step_size:") {
      iss >> *obj.step_size;
    } else if (field == "lower_bounds:") {
      std::string params_str;
      iss >> params_str;
      std::istringstream params_ss(params_str);
      double param;
      while (params_ss >> param) {
        obj.lower_bounds->push_back(param);
        if (params_ss.peek() == ',')
          params_ss.ignore();
      }
    } else if (field == "upper_bounds:") {
      std::string params_str;
      iss >> params_str;
      std::istringstream params_ss(params_str);
      double param;
      while (params_ss >> param) {
        obj.upper_bounds->push_back(param);
        if (params_ss.peek() == ',')
          params_ss.ignore();
      }
    } else if (field == "f_tol:") {
      iss >> *obj.f_tol;
    }
  } 

  return obj;
}

#define CUDAQ_ENSMALLEN_ALGORITHM_TYPE(NAME, NEEDS_GRADIENTS, EXTRA_PARAMS)    \
  class NAME : public BaseEnsmallen {                                          \
  public:                                                                      \
    NAME() = default;                                                          \
    bool requiresGradients() override { return NEEDS_GRADIENTS; }              \
    optimization_result                                                        \
    optimize(const int dim, optimizable_function &&opt_function) override;     \
    EXTRA_PARAMS                                                               \
  };

CUDAQ_ENSMALLEN_ALGORITHM_TYPE(
    lbfgs, true, std::optional<std::size_t> max_line_search_trials;)
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
