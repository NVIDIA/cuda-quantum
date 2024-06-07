/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/algorithms/optimizer.h"
#include <optional>

namespace cudaq::optimizers {
struct max_eval;
struct initial_parameters;

class BaseEnsmallen : public cudaq::optimizer {
public:
  std::optional<std::size_t> max_eval;
  std::optional<std::vector<double>> initial_parameters;
  std::optional<std::vector<double>> lower_bounds;
  std::optional<std::vector<double>> upper_bounds;
  std::optional<double> f_tol;
  std::optional<double> step_size;

  std::string __getstate__() const { return serialize(); }

  void __setstate__(const std::string &data) { deserialize(data); }

  std::string serialize() const {
    std::ostringstream oss;
    serializeOptional(oss, max_eval);
    serializeOptional(oss, initial_parameters);
    serializeOptional(oss, lower_bounds);
    serializeOptional(oss, upper_bounds);
    serializeOptional(oss, f_tol);
    serializeOptional(oss, step_size);
    return oss.str();
  }

  void deserialize(const std::string &data) {
    std::istringstream iss(data);
    deserializeOptional(iss, max_eval);
    deserializeOptional(iss, initial_parameters);
    deserializeOptional(iss, lower_bounds);
    deserializeOptional(iss, upper_bounds);
    deserializeOptional(iss, f_tol);
    deserializeOptional(iss, step_size);
  }

protected:
  void validate(optimizable_function &optFunction);

  template <typename T>
  void serializeOptional(std::ostringstream &oss,
                         const std::optional<T> &opt) const {
    if (opt.has_value()) {
      oss << "1 ";
      serialize(oss, opt.value());
    } else {
      oss << "0 ";
    }
  }

  template <typename T>
  void deserializeOptional(std::istringstream &iss, std::optional<T> &opt) {
    int has_value;
    iss >> has_value;
    if (has_value) {
      T value;
      deserialize(iss, value);
      if constexpr (std::is_fundamental_v<T>) {
        opt = value;
      } else {
        opt.emplace(std::move(value));
      }
    } else {
      opt.reset();
    }
  }

  void serialize(std::ostringstream &oss,
                 const std::vector<double> &vec) const {
    oss << vec.size() << " ";
    for (const auto &val : vec) {
      oss << val << " ";
    }
  }

  void deserialize(std::istringstream &iss, std::vector<double> &vec) {
    size_t size;
    iss >> size;
    vec.resize(size);
    for (auto &val : vec) {
      iss >> val;
    }
  }

  void serialize(std::ostringstream &oss, const std::size_t &value) const {
    oss << value << " ";
  }

  void deserialize(std::istringstream &iss, std::size_t &value) {
    iss >> value;
  }

  void serialize(std::ostringstream &oss, const double &value) const {
    oss << value << " ";
  }

  void deserialize(std::istringstream &iss, double &value) { iss >> value; }
};

#define CUDAQ_ENSMALLEN_ALGORITHM_TYPE(                                        \
    NAME, NEEDS_GRADIENTS, EXTRA_PARAMS_DECL, EXTRA_PARAMS_SERIALIZE,          \
    EXTRA_PARAMS_DESERIALIZE)                                                  \
  class NAME : public BaseEnsmallen {                                          \
  public:                                                                      \
    NAME() = default;                                                          \
    bool requiresGradients() override { return NEEDS_GRADIENTS; }              \
    optimization_result                                                        \
    optimize(const int dim, optimizable_function &&opt_function) override;     \
                                                                               \
    std::string serialize() const {                                            \
      std::ostringstream oss;                                                  \
      oss << BaseEnsmallen::serialize();                                       \
      EXTRA_PARAMS_SERIALIZE                                                   \
      return oss.str();                                                        \
    }                                                                          \
                                                                               \
    void deserialize(const std::string &data) {                                \
      std::istringstream iss(data);                                            \
      BaseEnsmallen::deserialize(data);                                        \
      EXTRA_PARAMS_DESERIALIZE                                                 \
    }                                                                          \
    EXTRA_PARAMS_DECL                                                          \
  };

#define SERIALIZE_PARAM(oss, param) serializeOptional(oss, param);
#define DESERIALIZE_PARAM(iss, param) deserializeOptional(iss, param);

CUDAQ_ENSMALLEN_ALGORITHM_TYPE(
    lbfgs, true, std::optional<std::size_t> max_line_search_trials;
    , SERIALIZE_PARAM(oss, max_line_search_trials),
    DESERIALIZE_PARAM(iss, max_line_search_trials))
CUDAQ_ENSMALLEN_ALGORITHM_TYPE(spsa, false, std::optional<double> alpha;
                               std::optional<double> gamma;
                               std::optional<double> eval_step_size;
                               ,
                               SERIALIZE_PARAM(oss, alpha)
                                   SERIALIZE_PARAM(oss, gamma)
                                       SERIALIZE_PARAM(oss, eval_step_size),
                               DESERIALIZE_PARAM(iss, alpha)
                                   DESERIALIZE_PARAM(iss, gamma)
                                       DESERIALIZE_PARAM(iss, eval_step_size))
CUDAQ_ENSMALLEN_ALGORITHM_TYPE(
    adam, true, std::optional<std::size_t> batch_size;
    std::optional<double> beta1; std::optional<double> beta2;
    std::optional<double> eps;
    ,
    SERIALIZE_PARAM(oss, batch_size) SERIALIZE_PARAM(oss, beta1)
        SERIALIZE_PARAM(oss, beta2) SERIALIZE_PARAM(oss, eps),
    DESERIALIZE_PARAM(iss, batch_size) DESERIALIZE_PARAM(iss, beta1)
        DESERIALIZE_PARAM(iss, beta2) DESERIALIZE_PARAM(iss, eps))
CUDAQ_ENSMALLEN_ALGORITHM_TYPE(gradient_descent, true, , , )
CUDAQ_ENSMALLEN_ALGORITHM_TYPE(sgd, true, std::optional<std::size_t> batch_size;
                               , SERIALIZE_PARAM(oss, batch_size),
                               DESERIALIZE_PARAM(iss, batch_size))

#undef SERIALIZE_PARAM
#undef DESERIALIZE_PARAM

} // namespace cudaq::optimizers
